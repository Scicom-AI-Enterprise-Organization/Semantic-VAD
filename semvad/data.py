"""Causal example construction + collation for `Scicom-intl/semantic-vad-eot`.

See README.md §3 ("Causal example construction") for the rationale: a training
example must be truncated the same way the model gets queried at eval/serving
time, or we train/test-mismatch on exactly the thing that matters -- whether
p(eot) rises correctly as a mid-turn silence gets longer.

No transcript text is fed to the model. The objective (README §1) is audio-native
EoT detection, and the dataset's `messages` field holds the *whole* turn's
transcript -- feeding it as context would leak words the model hasn't "heard" yet
at the causal cut point. `words` could be filtered to a causal prefix (the way
`eot-harness` itself does for other adapters), but that's a deliberately deferred
enhancement, not required for a first model.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Iterator, Optional

import numpy as np
import torch

from semvad.modeling import AUDIO_PROMPT_PREFIX

EOT_INSTRUCTION = "Has the speaker finished their turn? Answer yes or no."

# Seconds into each silence span at which to cut the audio. Multiple offsets per
# span teach the model that confidence should rise monotonically the longer a
# `hold` pause lasts -- which is exactly what eot-harness probes by scoring every
# `inference_interval` (default 100ms) across the whole span at eval time. A model
# trained only at offset 0 is untested (and likely miscalibrated) 800ms into a pause.
TRUNCATION_OFFSETS: tuple[float, ...] = (0.0, 0.2, 0.6, 1.2)


def iter_causal_examples(
    row: dict[str, Any],
    offsets: tuple[float, ...] = TRUNCATION_OFFSETS,
) -> Iterator[dict[str, Any]]:
    """One dataset row (one full user turn) -> zero or more causal training
    examples, one per (silence_span, truncation_offset). The last span (sorted by
    `start`) is `eot`; every earlier span is `hold`."""
    spans = sorted(row["silence_spans"], key=lambda s: s["start"])
    if not spans:
        return
    audio = row["audio"]["array"]
    sr = row["audio"]["sampling_rate"]
    n_spans = len(spans)
    weight = 1.0 / n_spans  # a turn with many hesitations shouldn't dominate `hold`
    for idx, span in enumerate(spans):
        label = 1.0 if idx == n_spans - 1 else 0.0
        seen_cuts = set()
        for offset in offsets:
            cut_time = min(span["start"] + offset, span["end"])
            cut_sample = int(cut_time * sr)
            if cut_sample <= 0 or cut_sample in seen_cuts:
                continue  # dedupe: short spans collapse several offsets to the same cut
            seen_cuts.add(cut_sample)
            yield {
                "audio": audio[:cut_sample],
                "sampling_rate": sr,
                "label": label,
                "weight": weight,
                "language": row["language"],
            }


def expand_to_causal_examples(batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
    """`datasets.Dataset.map(..., batched=True, remove_columns=...)` callback.

    Rows-out != rows-in (each input turn fans out into several span examples),
    which `datasets` batched map supports natively for both `Dataset` and
    `IterableDataset`.
    """
    out: dict[str, list[Any]] = {"audio": [], "sampling_rate": [], "label": [], "weight": [], "language": []}
    n = len(batch["id"])
    for i in range(n):
        row = {key: batch[key][i] for key in batch}
        for example in iter_causal_examples(row):
            for key, value in example.items():
                out[key].append(value)
    return out


@dataclasses.dataclass
class EoTCollator:
    """Builds `Qwen2AudioEoTClassifier`-ready batches from `expand_to_causal_examples` rows."""

    processor: Any
    max_audio_seconds: float = 16.0  # bounded causal window, README §6

    def __post_init__(self) -> None:
        # Right-padding keeps `attention_mask.sum(1) - 1` a valid "last real
        # token" index for every row -- see `Qwen2AudioEoTClassifier._pool_last_token`.
        self.processor.tokenizer.padding_side = "right"

    def _load_audio(self, example: dict[str, Any]) -> np.ndarray:
        audio = np.asarray(example["audio"], dtype=np.float32)
        sr = example["sampling_rate"]
        target_sr = self.processor.feature_extractor.sampling_rate
        if sr != target_sr:
            import librosa

            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        max_len = int(self.max_audio_seconds * sr)
        if len(audio) > max_len:
            # keep the *most recent* audio -- a voice agent never needs more than
            # a bounded trailing window to decide whether a turn just ended.
            audio = audio[-max_len:]
        return audio

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        audios = [self._load_audio(ex) for ex in examples]
        texts = [AUDIO_PROMPT_PREFIX + EOT_INSTRUCTION] * len(examples)
        batch = self.processor(
            text=texts,
            audio=audios,
            sampling_rate=self.processor.feature_extractor.sampling_rate,
            return_tensors="pt",
            padding=True,
        )
        batch["labels"] = torch.tensor([ex["label"] for ex in examples], dtype=torch.float32)
        batch["example_weight"] = torch.tensor([ex["weight"] for ex in examples], dtype=torch.float32)
        return dict(batch)


def load_causal_dataset(
    path: str,
    name: Optional[str],
    split: str,
    streaming: bool = True,
    num_proc: Optional[int] = None,
):
    """Load `Scicom-intl/semantic-vad-eot` (or a compatible dataset) and expand it
    into per-span causal examples. Streaming avoids materializing the ~150GB
    `all` config; use `streaming=False` + a small `name`/split for fast iteration."""
    from datasets import load_dataset

    dataset = load_dataset(path, name=name, split=split, streaming=streaming)
    map_kwargs = {"batched": True, "remove_columns": dataset.column_names}
    if not streaming and num_proc:
        map_kwargs["num_proc"] = num_proc
    return dataset.map(expand_to_causal_examples, **map_kwargs)
