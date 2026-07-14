"""``WhisperQwen3`` — the semantic-VAD model: Whisper encoder → linear adapter → Qwen3.

Architecture (README's "the branch we build")::

    audio ─► Whisper encoder ─► Linear projection ─► Qwen3 (causal LM) ─► marker logits

The Whisper encoder turns a mel spectrogram into frame embeddings; a single ``nn.Linear``
(no bias) projects them into Qwen3's embedding space; those vectors are scattered into the
input-embedding sequence wherever the tokenizer placed an ``<|AUDIO|>`` placeholder — the
Qwen2-Audio / SALMONN recipe, with **Qwen3** in place of Qwen2. There is **no LoRA** (full
fine-tune) and **no separate EOT head**: the end-of-turn signal is the logprob of the
``<|eot|>`` / ``<|hold|>`` marker token, read at serving time.

Attention uses the **flash-attention varlen** path: the model is loaded with
``attn_implementation="flash_attention_2"`` (or an fa3 kernel), examples are packed into one
sequence (batch dim 1), and ``cu_seq_lens_q``/``cu_seq_lens_k`` + ``max_length_q``/
``max_length_k`` (built by the collator) flow through ``**kwargs`` into the Qwen3 attention.
``attention_mask`` is therefore ``None`` — we do **not** build an SDPA block-diagonal mask.
"""

from __future__ import annotations

import torch
from torch import nn
from transformers import Qwen3ForCausalLM, WhisperConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.whisper.modeling_whisper import WhisperEncoder


def _make_fused_linear_ce():
    """Liger fused-linear cross-entropy (``reduction="sum"``) if available, else ``None``.

    Fusing the ``lm_head`` matmul with the softmax avoids materializing the full
    ``[tokens, vocab]`` logits — important with a large vocab and long packed sequences.
    """
    try:
        from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss

        return LigerFusedLinearCrossEntropyLoss(reduction="sum")
    except Exception:  # pragma: no cover - optional dependency
        return None


class WhisperQwen3(Qwen3ForCausalLM):
    """Qwen3 causal LM augmented with a Whisper audio encoder + projection adapter.

    ``config`` is a ``Qwen3Config`` carrying three extra attributes (set in ``train.py`` and
    persisted to ``config.json``):

    - ``audio_encoder_config`` — the Whisper encoder config (``dict`` or ``WhisperConfig``).
    - ``audio_token_index`` — id of the ``<|AUDIO|>`` placeholder token.
    - ``eot_token_index`` / ``hold_token_index`` — marker ids (for the serving read).
    """

    def __init__(self, config):
        super().__init__(config)
        enc_cfg = config.audio_encoder_config
        if isinstance(enc_cfg, dict):
            enc_cfg = WhisperConfig(**enc_cfg)
        self.audio_encoder_config = enc_cfg
        # Built here so from_pretrained(<our checkpoint>) has a module to load encoder
        # weights into; train.py overwrites this with the pretrained Whisper on a fresh run.
        self.encoder = WhisperEncoder(enc_cfg)
        self.projection = nn.Linear(enc_cfg.d_model, config.hidden_size, bias=False)
        self._fused_ce = _make_fused_linear_ce()

    # -- audio branch -----------------------------------------------------------------
    def freeze_encoder(self) -> None:
        """Freeze the Whisper encoder (projection + Qwen3 stay trainable). Optional."""
        for p in self.encoder.parameters():
            p.requires_grad_(False)

    def get_audio_features(self, input_features, feature_attention_mask):
        """Encode mel features and return the valid projected frames, flattened over clips.

        ``input_features``: ``[n_clips, n_mels, 3000]`` (padded to Whisper's fixed 30 s
        window). ``feature_attention_mask``: ``[n_clips, 3000]`` marking real mel frames.
        Returns ``[total_audio_tokens, hidden]`` where the per-clip valid counts
        (``num_audio_tokens``) sum to the number of ``<|AUDIO|>`` placeholders in the batch.
        """
        input_features = input_features.to(dtype=self.encoder.dtype, device=self.encoder.device)
        encoder_out = self.encoder(input_features).last_hidden_state  # [n_clips, 1500, d_model]
        feats = self.projection(encoder_out)  # [n_clips, 1500, hidden]

        # Whisper conv2 (stride 2) halves the frame count: n_out = (n_mel - 1)//2 + 1.
        lengths = (feature_attention_mask.sum(-1) - 1) // 2 + 1  # [n_clips]
        max_frames = feats.shape[1]
        keep = torch.arange(max_frames, device=feats.device)[None, :] < lengths[:, None]
        return feats[keep]  # clip-major order, matches placeholder order in the packed ids

    # -- forward ----------------------------------------------------------------------
    def forward(
        self,
        input_ids=None,
        position_ids=None,
        labels=None,
        input_features=None,
        feature_attention_mask=None,
        num_items_in_batch=None,
        inputs_embeds=None,
        **kwargs,  # carries cu_seq_lens_q/k + max_length_q/k for the flash-attn varlen path
    ):
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if input_features is not None:
            audio_embeds = self.get_audio_features(input_features, feature_attention_mask)
            audio_positions = input_ids == self.config.audio_token_index
            inputs_embeds = inputs_embeds.clone()  # avoid in-place on a graph leaf
            inputs_embeds[audio_positions] = audio_embeds.to(inputs_embeds.dtype)

        outputs = self.model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attention_mask=None,  # varlen is driven by cu_seq_lens in **kwargs, not a mask
            use_cache=False,
            **kwargs,
        )
        hidden = outputs.last_hidden_state

        if labels is None:
            return CausalLMOutputWithPast(logits=self.lm_head(hidden))

        loss = self._loss(hidden, labels, num_items_in_batch)
        return {"loss": loss}

    def _loss(self, hidden, labels, num_items_in_batch):
        # Shift over the whole packed sequence. This is safe across example boundaries
        # because every example begins with prompt tokens whose label is IGNORE_INDEX, so
        # the one cross-boundary prediction target is always masked out.
        shift_hidden = hidden[:, :-1].reshape(-1, hidden.shape[-1])
        shift_labels = labels[:, 1:].reshape(-1).to(shift_hidden.device)

        if self._fused_ce is not None:
            loss = self._fused_ce(self.lm_head.weight, shift_hidden, shift_labels)
        else:
            logits = self.lm_head(shift_hidden).float()
            loss = nn.functional.cross_entropy(
                logits, shift_labels, ignore_index=-100, reduction="sum"
            )

        # Normalize by the global count of supervised tokens (Trainer supplies this across
        # the grad-accum window; fall back to this micro-batch if it doesn't).
        denom = num_items_in_batch
        if denom is None:
            denom = (shift_labels != -100).sum().clamp(min=1)
        if torch.is_tensor(denom):
            denom = denom.to(loss.device)
        return loss / denom
