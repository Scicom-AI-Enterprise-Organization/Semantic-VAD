"""Batch collation for `WhisperQwen3EoTClassifier` (Option C).

Reuses Option A's causal example construction wholesale (`semvad.data`): per-span
truncation offsets (`0.0/0.2/0.6/1.2s` into each pause), `1/num_spans` example weighting,
singleton downsampling, and the telephony channel augmentation (`semvad.degrade`) are all
architecture-agnostic -- they just produce `{audio, sampling_rate, label, weight, ...}`
dicts, same as they do for the Qwen2-Audio classifier. Only *collation* differs: instead of
a single Qwen2-Audio `processor` call, we run Whisper's feature extractor + a Qwen3
tokenizer separately and pad both by hand (see `whisper_qwen3_head/prompt.py`).

No flash-attn varlen packing here (unlike `semantic_vad/training/data.py`): each example is
one independent classification label, so a normal padded `[batch, seq]` + `attention_mask`
is the right shape, exactly like Option A's collator.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import torch

from whisper_qwen3_head.prompt import EOT_INSTRUCTION, SpecialIds, build_prompt_ids, num_audio_tokens

# Re-exported for convenience -- callers only need to import from this module for the full
# training-data pipeline (dataset loading lives in `semvad.data`, unchanged).
from semvad.data import TRUNCATION_OFFSETS, iter_causal_examples, load_causal_dataset  # noqa: F401


@dataclasses.dataclass
class EoTCollator:
    """Builds `WhisperQwen3EoTClassifier`-ready batches from `semvad.data`-expanded rows."""

    tokenizer: Any
    feature_extractor: Any
    special_ids: SpecialIds
    max_audio_seconds: float = 16.0  # bounded causal window, matches semvad's collator
    instruction: str = EOT_INSTRUCTION

    def __post_init__(self) -> None:
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _load_audio(self, example: dict[str, Any]) -> np.ndarray:
        audio = np.asarray(example["audio"], dtype=np.float32)
        sr = example["sampling_rate"]
        target_sr = self.feature_extractor.sampling_rate
        if sr != target_sr:
            import librosa

            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        max_len = int(self.max_audio_seconds * sr)
        if len(audio) > max_len:
            # keep the *most recent* audio -- a voice agent never needs more than a bounded
            # trailing window to decide whether a turn just ended.
            audio = audio[-max_len:]
        return audio

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        audios = [self._load_audio(ex) for ex in examples]
        feat = self.feature_extractor(
            audios,
            sampling_rate=self.feature_extractor.sampling_rate,
            return_attention_mask=True,
            padding="max_length",
            return_tensors="np",
        )
        input_features = torch.from_numpy(feat["input_features"].astype(np.float32))
        feature_attention_mask = torch.from_numpy(np.asarray(feat["attention_mask"], dtype=np.int64))

        encode = lambda text: self.tokenizer.encode(text, add_special_tokens=False)  # noqa: E731
        all_ids = [
            build_prompt_ids(
                encode,
                self.special_ids,
                n_audio=num_audio_tokens(int(mask.sum().item())),
                instruction=self.instruction,
            )
            for mask in feature_attention_mask
        ]

        # Right-padding keeps `attention_mask.sum(1) - 1` a valid "last real token" index for
        # every row -- see `WhisperQwen3EoTClassifier._pool_last_token`.
        max_len = max(len(ids) for ids in all_ids)
        input_ids = torch.full((len(all_ids), max_len), self.tokenizer.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(all_ids), max_len), dtype=torch.long)
        for i, ids in enumerate(all_ids):
            input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            attention_mask[i, : len(ids)] = 1

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "input_features": input_features,
            "feature_attention_mask": feature_attention_mask,
            "labels": torch.tensor([ex["label"] for ex in examples], dtype=torch.float32),
            "example_weight": torch.tensor([ex["weight"] for ex in examples], dtype=torch.float32),
        }
