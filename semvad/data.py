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
import hashlib
import os
from typing import Any, Iterator, Optional

import numpy as np
import torch

from semvad.degrade import TelephonyDegrader
from semvad.modeling import AUDIO_PROMPT_PREFIX

EOT_INSTRUCTION = "Has the speaker finished their turn? Answer yes or no."


def _stable_unit_interval(key: str) -> float:
    """Deterministic pseudo-random value in [0, 1) derived from `key`.

    Unlike `random.random()`, this is stable across processes/workers/map-shards --
    the same turn is always kept or dropped the same way no matter which
    dataloader worker or `.map()` shard happens to process it.
    """
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64

# Seconds into each silence span at which to cut the audio. Multiple offsets per
# span teach the model that confidence should rise monotonically the longer a
# `hold` pause lasts -- which is exactly what eot-harness probes by scoring every
# `inference_interval` (default 100ms) across the whole span at eval time. A model
# trained only at offset 0 is untested (and likely miscalibrated) 800ms into a pause.
TRUNCATION_OFFSETS: tuple[float, ...] = (0.0, 0.2, 0.6, 1.2)


def iter_causal_examples(
    row: dict[str, Any],
    offsets: tuple[float, ...] = TRUNCATION_OFFSETS,
    singleton_keep_prob: float = 1.0,
    degrader: Optional[TelephonyDegrader] = None,
) -> Iterator[dict[str, Any]]:
    """One dataset row (one full user turn) -> zero or more causal training
    examples, one per (silence_span, truncation_offset). The last span (sorted by
    `start`) is `eot`; every earlier span is `hold`.

    `singleton_keep_prob` downsamples turns with exactly one silence span --
    i.e. turns that contribute an `eot` example and *zero* `hold` examples.
    These are the majority of turns (~56% in en/train) and, because `weight =
    1/n_spans` below gives every turn equal total loss weight regardless of span
    count, they alone push the corpus's effective eot:hold loss-weight ratio to
    roughly 75:25 -- well past the raw ~60:40 span-count split, and the opposite
    direction from eval benchmarks that skew hold-majority. Dropping a fixed
    fraction of singleton turns (keyed by `row["id"]`, not `random`, so it's
    reproducible regardless of worker/shard) corrects that bias without
    touching any multi-span turn's hold examples.

    `degrader`, if given, runs the deployment-matching call-centre channel
    simulation (see `semvad/degrade.py`) once on the *whole turn's* audio before
    any span is cut -- a real phone channel's distortion profile is constant
    for the duration of a call, so every causal example derived from this turn
    shares one degraded realization instead of each getting an independent one.
    """
    spans = sorted(row["silence_spans"], key=lambda s: s["start"])
    if not spans:
        return
    if len(spans) == 1 and singleton_keep_prob < 1.0 and _stable_unit_interval(row["id"]) >= singleton_keep_prob:
        return
    audio = row["audio"]["array"]
    sr = row["audio"]["sampling_rate"]
    if degrader is not None:
        audio = degrader.degrade(audio, sr)
    n_spans = len(spans)
    weight = 1.0 / n_spans  # a turn with many hesitations shouldn't dominate `hold`
    for idx, span in enumerate(spans):
        label = 1.0 if idx == n_spans - 1 else 0.0
        seen_cuts = set()
        cut_samples = []
        for offset in offsets:
            cut_time = min(span["start"] + offset, span["end"])
            cut_sample = int(cut_time * sr)
            if cut_sample <= 0 or cut_sample in seen_cuts:
                continue  # dedupe: short spans collapse several offsets to the same cut
            seen_cuts.add(cut_sample)
            cut_samples.append(cut_sample)
        if not cut_samples:
            continue
        # Split the span's weight evenly across however many offsets survived
        # dedup, so a short span (fewer distinct cuts) contributes the same total
        # loss mass as a long one instead of being shortchanged relative to it.
        example_weight = weight / len(cut_samples)
        for cut_sample in cut_samples:
            yield {
                "audio": audio[:cut_sample],
                "sampling_rate": sr,
                "label": label,
                "weight": example_weight,
                "language": row["language"],
            }


def expand_to_causal_examples(
    batch: dict[str, list[Any]],
    singleton_keep_prob: float = 1.0,
    degrader: Optional[TelephonyDegrader] = None,
) -> dict[str, list[Any]]:
    """`datasets.Dataset.map(..., batched=True, remove_columns=...)` callback.

    Rows-out != rows-in (each input turn fans out into several span examples),
    which `datasets` batched map supports natively for both `Dataset` and
    `IterableDataset`. `singleton_keep_prob` and `degrader` are forwarded to
    `iter_causal_examples` -- see its docstring.
    """
    out: dict[str, list[Any]] = {"audio": [], "sampling_rate": [], "label": [], "weight": [], "language": []}
    n = len(batch["id"])
    for i in range(n):
        row = {key: batch[key][i] for key in batch}
        for example in iter_causal_examples(row, singleton_keep_prob=singleton_keep_prob, degrader=degrader):
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
    singleton_keep_prob: float = 1.0,
    degrader: Optional[TelephonyDegrader] = None,
):
    """Load `Scicom-intl/semantic-vad-eot` (or a compatible dataset) and expand it
    into per-span causal examples. Streaming avoids materializing the ~150GB
    `all` config; use `streaming=False` + a small `name`/split for fast iteration.

    `singleton_keep_prob` (default 1.0, i.e. no-op) downsamples single-silence-span
    turns -- see `iter_causal_examples`'s docstring. `degrader`, if given, runs the
    call-centre channel simulation (`semvad/degrade.py`) on every turn's audio.
    Leave both at their no-op defaults for eval datasets, which should reflect the
    natural, undistorted distribution.

    `num_proc` controls `.map()` multiprocessing for non-streaming datasets: the
    default `None` auto-picks `os.cpu_count() // 2` worker processes, `0` disables
    multiprocessing (single process), and a positive int uses that many workers.
    Ignored when `streaming=True` (`datasets` doesn't support `num_proc` for
    `IterableDataset.map()`)."""
    from datasets import load_dataset

    dataset = load_dataset(path, name=name, split=split, streaming=streaming)
    map_kwargs = {
        "batched": True,
        "remove_columns": dataset.column_names,
        "fn_kwargs": {"singleton_keep_prob": singleton_keep_prob, "degrader": degrader},
    }
    if not streaming:
        resolved_num_proc = (os.cpu_count() or 2) // 2 if num_proc is None else num_proc
        if resolved_num_proc:
            map_kwargs["num_proc"] = resolved_num_proc
        # `audio` is a list-of-float32 column and pyarrow's ListArray offsets are
        # int32. Causal examples aren't length-capped until collate time, so a
        # writer buffer holding the default 1000 rows of (possibly long) audio
        # can overflow 2**31 total samples before it flushes. Flush much more
        # often to keep any single buffer well under that limit.
        map_kwargs["writer_batch_size"] = 64
        map_kwargs["batch_size"] = 64
    return dataset.map(expand_to_causal_examples, **map_kwargs)
