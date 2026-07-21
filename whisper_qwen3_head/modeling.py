"""Whisper encoder → Qwen3 backbone + Option A's p(eot) classification head (Option C).

This is a third way to get ``p(eot)`` out of an audio-native backbone, combining pieces of
the other two already in this repo:

- ``semvad/`` (Option A): ``Qwen2-Audio-7B`` backbone (a native multimodal checkpoint) +
  ``EoTHead`` on the pooled last hidden state, audio tower frozen, LoRA over the LLM trunk.
- ``semantic_vad/training/`` (Option B): Whisper encoder → linear projection → **Qwen3**
  (bolted together — Qwen3 has no native audio checkpoint), full fine-tune, ``<|eot|>``/
  ``<|hold|>`` marker-token read (next-token prediction), flash-attn varlen packing.

Option C takes Option B's smaller from-scratch backbone (Whisper encoder + projection +
Qwen3, no 7B multimodal checkpoint required) and reads ``p(eot)`` the way Option A does — a
dedicated ``EoTHead`` (reused as-is from ``semvad/modeling.py``) on the pooled last hidden
state, no marker token, no vocabulary projection. Like Option B (and unlike Option A's LoRA),
this is a **full fine-tune**: encoder + projection + Qwen3 trunk + head are all trainable.

Because each training example is scored independently (a classification label, not a
generated sequence), there's no benefit to flash-attention varlen packing here — batches are
built as a normal padded ``[batch, seq]`` tensor with ``attention_mask`` (see ``data.py``),
same as Option A's collator.
"""

from __future__ import annotations

import os
import tempfile
from typing import Optional

import numpy as np
import torch
from torch import nn
from transformers import Qwen3Model, WhisperConfig, WhisperModel
from transformers.models.whisper.modeling_whisper import WhisperEncoder

from semvad.modeling import EoTHead, EoTHeadConfig, EoTOutput
from whisper_qwen3_head.prompt import EOT_INSTRUCTION, SpecialIds, build_prompt_ids, num_audio_tokens

DEFAULT_QWEN3_NAME = "Qwen/Qwen3-0.6B"
DEFAULT_WHISPER_NAME = "openai/whisper-base"


class WhisperQwen3EoTClassifier(nn.Module):
    """Whisper encoder → linear projection → Qwen3 trunk (no ``lm_head``) → ``EoTHead``."""

    def __init__(self, encoder: WhisperEncoder, projection: nn.Linear, backbone: Qwen3Model, head: EoTHead):
        super().__init__()
        self.encoder = encoder
        self.projection = projection
        self.backbone = backbone
        self.head = head
        # Resolved from a tokenizer via `set_special_ids` -- not known until the caller has
        # added `whisper_qwen3_head.prompt.SPECIAL_TOKENS` (see `from_pretrained`'s docstring).
        self.audio_token_id: Optional[int] = None

    def set_special_ids(self, ids: SpecialIds) -> None:
        """Record the ``<|AUDIO|>`` placeholder id `forward` scatters audio embeddings into.

        Call once after adding `whisper_qwen3_head.prompt.SPECIAL_TOKENS` to the tokenizer and
        resizing `self.backbone`'s embeddings (see ``train.py``) -- and again after
        `from_checkpoint`, since that id isn't itself persisted in the checkpoint.
        """
        self.audio_token_id = ids.audio

    # ---- construction -----------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        qwen3_name: str = DEFAULT_QWEN3_NAME,
        whisper_name: str = DEFAULT_WHISPER_NAME,
        *,
        dtype: torch.dtype = torch.bfloat16,
        freeze_encoder: bool = False,
        head_config: Optional[EoTHeadConfig] = None,
        attn_implementation: str = "sdpa",
    ) -> "WhisperQwen3EoTClassifier":
        """Build the model. Does **not** touch a tokenizer -- callers must add
        ``whisper_qwen3_head.prompt.SPECIAL_TOKENS`` to their tokenizer and call
        ``backbone.resize_token_embeddings(len(tokenizer), ...)`` afterwards (see
        ``train.py``), mirroring how ``semantic_vad.training.train`` handles it for Option B.
        """
        backbone = Qwen3Model.from_pretrained(qwen3_name, dtype=dtype, attn_implementation=attn_implementation)

        whisper_config = WhisperConfig.from_pretrained(whisper_name)
        encoder = WhisperEncoder(whisper_config).to(dtype=dtype)
        whisper = WhisperModel.from_pretrained(whisper_name, dtype=dtype)
        encoder.load_state_dict(whisper.get_encoder().state_dict())
        del whisper
        if freeze_encoder:
            encoder.requires_grad_(False)

        projection = nn.Linear(whisper_config.d_model, backbone.config.hidden_size, bias=False).to(dtype=dtype)
        head = EoTHead(backbone.config.hidden_size, head_config).to(dtype=torch.float32)
        return cls(encoder, projection, backbone, head)

    # ---- audio branch (mirrors semantic_vad.training.modeling.WhisperQwen3) ----------------

    def get_audio_features(self, input_features: torch.Tensor, feature_attention_mask: torch.Tensor) -> torch.Tensor:
        """Encode mel features and return the valid projected frames, flattened over clips.

        ``input_features``: ``[n_clips, n_mels, 3000]``. ``feature_attention_mask``:
        ``[n_clips, 3000]`` marking real mel frames. Returns ``[total_audio_tokens, hidden]``
        where the per-clip valid counts (``num_audio_tokens``) sum to the number of
        ``<|AUDIO|>`` placeholders in the batch.
        """
        input_features = input_features.to(dtype=self.encoder.dtype, device=self.encoder.device)
        encoder_out = self.encoder(input_features).last_hidden_state  # [n_clips, 1500, d_model]
        feats = self.projection(encoder_out)  # [n_clips, 1500, hidden]

        lengths = (feature_attention_mask.sum(-1) - 1) // 2 + 1  # [n_clips]
        max_frames = feats.shape[1]
        keep = torch.arange(max_frames, device=feats.device)[None, :] < lengths[:, None]
        return feats[keep]  # clip-major order, matches placeholder order in input_ids

    @staticmethod
    def _pool_last_token(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Gather each row's last non-pad hidden state. Requires right-padding."""
        last_idx = attention_mask.sum(dim=1) - 1
        batch_idx = torch.arange(hidden_states.size(0), device=hidden_states.device)
        return hidden_states[batch_idx, last_idx]

    # ---- forward ------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        input_features: Optional[torch.FloatTensor] = None,
        feature_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        example_weight: Optional[torch.Tensor] = None,
        **_unused,
    ) -> EoTOutput:
        inputs_embeds = self.backbone.get_input_embeddings()(input_ids)
        if input_features is not None:
            if self.audio_token_id is None:
                raise RuntimeError("call set_special_ids() before running forward() with audio input")
            audio_embeds = self.get_audio_features(input_features, feature_attention_mask)
            audio_positions = input_ids == self.audio_token_id
            inputs_embeds = inputs_embeds.clone()  # avoid in-place write on a graph leaf
            inputs_embeds[audio_positions] = audio_embeds.to(inputs_embeds.dtype)

        outputs = self.backbone(inputs_embeds=inputs_embeds, attention_mask=attention_mask, use_cache=False)
        pooled = self._pool_last_token(outputs.last_hidden_state, attention_mask)
        logits = self.head(pooled.to(self.head.fc1.weight.dtype))

        loss = None
        if labels is not None:
            # `example_weight` defaults to 1/num_spans_in_row (see semvad/data.py), so a turn
            # with several mid-turn hesitations doesn't drown out single-pause turns.
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits, labels.to(logits.dtype), weight=example_weight.to(logits.dtype) if example_weight is not None else None
            )
        return EoTOutput(logits=logits, p_eot=torch.sigmoid(logits.float()), loss=loss)

    # ---- params / checkpointing --------------------------------------------

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def count_parameters(self, trainable_only: bool = False) -> int:
        params = self.trainable_parameters() if trainable_only else list(self.parameters())
        return sum(p.numel() for p in params)

    def save_pretrained(self, save_dir: str) -> None:
        """Full fine-tune (no LoRA) -- persist every trainable piece: the Whisper encoder,
        the Qwen3 backbone, the projection, and the head. Unlike ``semvad``'s
        ``save_adapter`` (which skips the frozen 7B backbone), everything here was actually
        trained, so everything is saved."""
        os.makedirs(save_dir, exist_ok=True)
        self.encoder.save_pretrained(os.path.join(save_dir, "encoder"))
        self.backbone.save_pretrained(os.path.join(save_dir, "backbone"))
        torch.save(self.projection.state_dict(), os.path.join(save_dir, "projection.pt"))
        torch.save(self.head.state_dict(), os.path.join(save_dir, "eot_head.pt"))

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str,
        *,
        dtype: torch.dtype = torch.bfloat16,
        attn_implementation: str = "sdpa",
        head_config: Optional[EoTHeadConfig] = None,
        revision: Optional[str] = None,
        token: Optional[str] = None,
    ) -> "WhisperQwen3EoTClassifier":
        """Load a checkpoint written by ``save_pretrained`` -- either a local directory or a
        ``push_to_hub`` repo id (``"org/model-name"``); ``revision``/``token`` are ignored for
        a local directory. The tokenizer/feature extractor are saved alongside it (see
        ``train.py`` / ``push_to_hub``) but loaded separately by the caller.
        """
        is_local = os.path.isdir(checkpoint)
        hub_kwargs = {} if is_local else {"revision": revision, "token": token}

        encoder = WhisperEncoder.from_pretrained(
            checkpoint if not is_local else os.path.join(checkpoint, "encoder"),
            dtype=dtype,
            subfolder="encoder" if not is_local else "",
            **hub_kwargs,
        )
        backbone = Qwen3Model.from_pretrained(
            checkpoint if not is_local else os.path.join(checkpoint, "backbone"),
            dtype=dtype,
            attn_implementation=attn_implementation,
            subfolder="backbone" if not is_local else "",
            **hub_kwargs,
        )

        if is_local:
            projection_path = os.path.join(checkpoint, "projection.pt")
            head_path = os.path.join(checkpoint, "eot_head.pt")
        else:
            from huggingface_hub import hf_hub_download

            projection_path = hf_hub_download(checkpoint, filename="projection.pt", **hub_kwargs)
            head_path = hf_hub_download(checkpoint, filename="eot_head.pt", **hub_kwargs)

        projection = nn.Linear(encoder.config.d_model, backbone.config.hidden_size, bias=False).to(dtype=dtype)
        projection.load_state_dict(torch.load(projection_path, map_location="cpu"))
        head = EoTHead(backbone.config.hidden_size, head_config).to(dtype=torch.float32)
        head.load_state_dict(torch.load(head_path, map_location="cpu"))
        return cls(encoder, projection, backbone, head)

    _MODEL_CARD_TEMPLATE = """\
---
library_name: transformers
tags:
- end-of-turn
- semantic-vad
- whisper
- qwen3
---

# WhisperQwen3EoTClassifier

Whisper encoder → linear projection → Qwen3 backbone (no `lm_head`) → a small binary
classification head (`LayerNorm → Linear → GELU → Linear → 1 logit`) predicting `p(eot)` --
whether a speaker has finished their conversational turn. Full fine-tune (no LoRA); see the
`semantic-vad` repo's `whisper_qwen3_head/modeling.py` for the reference implementation.

## Load

```python
from transformers import AutoTokenizer, WhisperFeatureExtractor
from whisper_qwen3_head.modeling import WhisperQwen3EoTClassifier
from whisper_qwen3_head.prompt import SpecialIds, AUDIO_BOS_TOKEN, AUDIO_EOS_TOKEN, AUDIO_TOKEN

model = WhisperQwen3EoTClassifier.from_checkpoint("{repo_id}")
tokenizer = AutoTokenizer.from_pretrained("{repo_id}")
feature_extractor = WhisperFeatureExtractor.from_pretrained("{repo_id}")
special_ids = SpecialIds(
    audio_bos=tokenizer.convert_tokens_to_ids(AUDIO_BOS_TOKEN),
    audio_eos=tokenizer.convert_tokens_to_ids(AUDIO_EOS_TOKEN),
    audio=tokenizer.convert_tokens_to_ids(AUDIO_TOKEN),
)
model.set_special_ids(special_ids)
model.eval()

p_eot = model.predict_p_eot(tokenizer, feature_extractor, special_ids, audio_array, sampling_rate)
```
"""

    def push_to_hub(
        self,
        repo_id: str,
        *,
        tokenizer=None,
        feature_extractor=None,
        commit_message: str = "Upload WhisperQwen3EoTClassifier",
        private: bool = False,
        token: Optional[str] = None,
        revision: Optional[str] = None,
        create_model_card: bool = True,
    ) -> str:
        """Push this checkpoint (encoder + backbone + projection + head, and optionally the
        tokenizer/feature extractor) to the Hugging Face Hub, mirroring
        ``PreTrainedModel.push_to_hub``: writes everything to a local temp dir via
        ``save_pretrained`` first, then uploads that dir as one commit.

        Pass ``tokenizer``/``feature_extractor`` (the same objects used for training/
        inference) so the pushed repo is self-contained and reloadable with
        ``from_checkpoint(repo_id)`` alone -- otherwise only the raw model weights are
        pushed. Returns the commit URL.
        """
        from huggingface_hub import create_repo, upload_folder

        create_repo(repo_id, token=token, private=private, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmp_dir:
            self.save_pretrained(tmp_dir)
            if tokenizer is not None:
                tokenizer.save_pretrained(tmp_dir)
            if feature_extractor is not None:
                feature_extractor.save_pretrained(tmp_dir)

            readme_path = os.path.join(tmp_dir, "README.md")
            if create_model_card and not os.path.exists(readme_path):
                with open(readme_path, "w") as f:
                    f.write(self._MODEL_CARD_TEMPLATE.format(repo_id=repo_id))

            commit = upload_folder(
                repo_id=repo_id,
                folder_path=tmp_dir,
                commit_message=commit_message,
                token=token,
                revision=revision,
            )
        return commit.commit_url if hasattr(commit, "commit_url") else str(commit)

    # ---- single-clip convenience path (benchmark script / future eot-harness adapter) -----

    @torch.inference_mode()
    def predict_p_eot(
        self,
        tokenizer,
        feature_extractor,
        special_ids: SpecialIds,
        audio: np.ndarray,
        sampling_rate: int,
        instruction: str = EOT_INSTRUCTION,
    ) -> float:
        """Score one causal audio prefix as p(eot) via a single forward pass + head."""
        self.set_special_ids(special_ids)
        target_sr = feature_extractor.sampling_rate
        if sampling_rate != target_sr:
            import librosa

            audio = librosa.resample(np.asarray(audio, dtype=np.float32), orig_sr=sampling_rate, target_sr=target_sr)

        feat = feature_extractor(
            audio, sampling_rate=target_sr, return_attention_mask=True, padding="max_length", return_tensors="np"
        )
        device = next(self.parameters()).device
        input_features = torch.from_numpy(feat["input_features"][0].astype(np.float32))[None].to(device)
        feature_attention_mask = torch.from_numpy(np.asarray(feat["attention_mask"][0], dtype=np.int64))[None].to(device)
        n_audio = num_audio_tokens(int(feature_attention_mask.sum().item()))

        input_ids = build_prompt_ids(
            lambda text: tokenizer.encode(text, add_special_tokens=False),
            special_ids,
            n_audio=n_audio,
            instruction=instruction,
        )
        input_ids_t = torch.tensor(input_ids, dtype=torch.long, device=device)[None]
        attention_mask = torch.ones_like(input_ids_t)

        output = self.forward(
            input_ids=input_ids_t,
            attention_mask=attention_mask,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
        )
        return output.p_eot.item()
