"""Qwen2-Audio backbone + a lightweight classification head for p(end-of-turn).

`Qwen2AudioForConditionalGeneration` ships an `lm_head` of
`Linear(hidden_size, vocab_size, bias=False)` -- for Qwen2-Audio-7B-Instruct that's
`Linear(4096, 156032)`, ~640M parameters, purely to score a 156k-token vocabulary we
never need for a binary decision. This module loads only `Qwen2AudioModel` (the
audio tower + multimodal projector + language-model trunk, with no head at all --
see `Qwen2AudioPreTrainedModel.base_model_prefix = "model"`, which is why
`Qwen2AudioModel.from_pretrained(...)` loads a `Qwen2AudioForConditionalGeneration`
checkpoint's backbone weights and simply drops the unused `lm_head.*` keys instead
of ever allocating them) and replaces the head with a ~1M-parameter MLP.

One forward pass through the backbone + head yields `p(eot)` directly -- no
autoregressive decoding, no vocabulary projection.
"""

from __future__ import annotations

import dataclasses
import os
from typing import Optional

import numpy as np
import torch
from torch import nn
from transformers import Qwen2AudioModel
from transformers.utils import ModelOutput

DEFAULT_MODEL_NAME = "Qwen/Qwen2-Audio-7B-Instruct"

# Qwen2-Audio's chat template wraps every audio clip in these three tokens; the
# processor expands `<|AUDIO|>` into as many placeholder tokens as the audio
# tower actually produces for the given clip length.
AUDIO_PROMPT_PREFIX = "<|audio_bos|><|AUDIO|><|audio_eos|>"


@dataclasses.dataclass
class EoTHeadConfig:
    head_hidden_size: int = 256
    dropout: float = 0.1


class EoTHead(nn.Module):
    """Binary classification head: LayerNorm -> Linear -> GELU -> Linear -> 1 logit.

    ~(hidden_size * head_hidden_size + head_hidden_size) params -- for a 4096-wide
    backbone and the default `head_hidden_size=256` that's ~1.05M params, versus the
    ~640M-param `lm_head` it replaces.
    """

    def __init__(self, hidden_size: int, config: Optional[EoTHeadConfig] = None):
        super().__init__()
        config = config or EoTHeadConfig()
        self.norm = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size, config.head_hidden_size)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)
        self.fc2 = nn.Linear(config.head_hidden_size, 1)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        x = self.norm(pooled)
        x = self.act(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x).squeeze(-1)  # (batch,) logits


@dataclasses.dataclass
class EoTOutput(ModelOutput):
    """Dict- and attribute-accessible, like any HF model output -- lets this plug
    straight into `Trainer.compute_loss`, which does `outputs["loss"]`."""

    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    p_eot: Optional[torch.Tensor] = None


class Qwen2AudioEoTClassifier(nn.Module):
    """Qwen2-Audio backbone (no lm_head) + a binary end-of-turn head."""

    def __init__(self, backbone: Qwen2AudioModel, head: EoTHead):
        super().__init__()
        self.backbone = backbone
        self.head = head

    # ---- construction -----------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = DEFAULT_MODEL_NAME,
        *,
        dtype: torch.dtype = torch.bfloat16,
        freeze_audio_tower: bool = True,
        head_config: Optional[EoTHeadConfig] = None,
        **kwargs,
    ) -> "Qwen2AudioEoTClassifier":
        backbone = Qwen2AudioModel.from_pretrained(model_name, dtype=dtype, **kwargs)
        if freeze_audio_tower:
            backbone.audio_tower.requires_grad_(False)
        hidden_size = backbone.config.text_config.hidden_size
        head = EoTHead(hidden_size, head_config).to(dtype=torch.float32)
        return cls(backbone, head)

    def apply_lora(self, **lora_kwargs) -> "Qwen2AudioEoTClassifier":
        """Wrap the LLM trunk in a LoRA adapter (peft); audio tower/projector untouched."""
        from peft import LoraConfig, get_peft_model

        lora_kwargs.setdefault("r", 16)
        lora_kwargs.setdefault("lora_alpha", 32)
        lora_kwargs.setdefault("lora_dropout", 0.05)
        lora_kwargs.setdefault(
            "target_modules", ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        lora_kwargs.setdefault("bias", "none")
        config = LoraConfig(**lora_kwargs)
        self.backbone.language_model = get_peft_model(self.backbone.language_model, config)
        return self

    # ---- forward ------------------------------------------------------------

    @staticmethod
    def _pool_last_token(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Gather each row's last non-pad hidden state. Requires right-padding."""
        last_idx = attention_mask.sum(dim=1) - 1
        batch_idx = torch.arange(hidden_states.size(0), device=hidden_states.device)
        return hidden_states[batch_idx, last_idx]

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
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            use_cache=False,
        )
        # `outputs.attention_mask` stays aligned to `last_hidden_state`'s sequence
        # length even when the audio-token merge changes shapes; always pool with it,
        # not the mask that was passed in.
        pooled = self._pool_last_token(outputs.last_hidden_state, outputs.attention_mask)
        logits = self.head(pooled.to(self.head.fc1.weight.dtype))
        loss = None
        if labels is not None:
            # `example_weight` defaults to 1/num_spans_in_row (README §3 "class
            # balance"), so a turn with several mid-turn hesitations doesn't drown
            # out single-pause turns in the `hold` class.
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

    def save_adapter(self, save_dir: str) -> None:
        """Persist only the trainable bits: the EoT head, and the LoRA adapter if present."""
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.head.state_dict(), os.path.join(save_dir, "eot_head.pt"))
        lm = self.backbone.language_model
        if hasattr(lm, "save_pretrained") and hasattr(lm, "peft_config"):
            lm.save_pretrained(os.path.join(save_dir, "lora"))

    def load_adapter(self, save_dir: str, is_trainable: bool = False) -> "Qwen2AudioEoTClassifier":
        """Load a checkpoint written by `save_adapter`. Call this directly on a
        plain (non-LoRA-wrapped) model -- do NOT call `apply_lora()` first: the
        LoRA weights here are attached via `PeftModel.from_pretrained`, which
        wraps whatever `self.backbone.language_model` currently is, so wrapping
        an already-LoRA model here nests two adapters and the saved state dict's
        keys no longer match the nested module paths, so the checkpoint's LoRA
        weights silently fail to load (peft only warns).

        `is_trainable` must be True to continue training the loaded adapter --
        `PeftModel.from_pretrained` defaults to `is_trainable=False` (inference
        mode, `requires_grad=False` on every LoRA param).
        """
        head_path = os.path.join(save_dir, "eot_head.pt")
        if os.path.exists(head_path):
            self.head.load_state_dict(torch.load(head_path, map_location="cpu"))
        lora_path = os.path.join(save_dir, "lora")
        if os.path.isdir(lora_path):
            from peft import PeftModel

            self.backbone.language_model = PeftModel.from_pretrained(
                self.backbone.language_model, lora_path, is_trainable=is_trainable
            )
        return self

    # ---- single-clip convenience path (used by the eot-harness adapter and the
    # Gradio demo) --------------------------------------------------------------

    @torch.inference_mode()
    def predict_p_eot(
        self,
        processor,
        audio: np.ndarray,
        sampling_rate: int,
        prior_text: str = "",
    ) -> float:
        """Score one causal audio prefix (+ optional transcript-so-far) as p(eot)."""
        target_sr = processor.feature_extractor.sampling_rate
        if sampling_rate != target_sr:
            import librosa

            audio = librosa.resample(np.asarray(audio, dtype=np.float32), orig_sr=sampling_rate, target_sr=target_sr)
        processor.tokenizer.padding_side = "right"
        prompt = AUDIO_PROMPT_PREFIX + (prior_text or "")
        inputs = processor(
            text=prompt, audio=audio, sampling_rate=target_sr, return_tensors="pt", padding=True
        )
        device = next(self.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        output = self.forward(**inputs)
        return output.p_eot.item()
