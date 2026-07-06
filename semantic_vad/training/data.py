"""Dataset + varlen packing collator that feed :class:`~.modeling.WhisperQwen3`.

The input is an eot-bench-compatible parquet built by :mod:`semantic_vad.build` (audio as
WAV/MP3 bytes, ``words``, ``silence_spans`` with the EOT positionally last). Each **turn
row expands into several training examples** (see :func:`plan_examples`):

- one ``<|eot|>`` example from the full clip, and
- one ``<|hold|>`` example per internal hold span — the audio truncated at the end of that
  pause, with only the words spoken before it.

Featurization/tokenization happen on the fly in ``__getitem__`` (clips are short, mel is
cheap); the collator then multipacks a micro-batch into a single flash-attn varlen batch.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
import soundfile as sf
import torch

from .packing import build_cu_seqlens
from .prompt import DEFAULT_SYSTEM, SpecialIds, build_example, num_audio_tokens

TARGET_SR = 16000
_MIN_SAMPLES = int(0.02 * TARGET_SR)  # skip sub-20 ms fragments


@dataclass
class _Example:
    """A planned (row, decision-point) pair, resolved to audio+tokens in ``__getitem__``."""

    row_idx: int
    end_sec: float  # slice the clip to [0, end_sec]
    marker: str  # "eot" | "hold"
    transcript: str  # words spoken up to the decision point


def plan_examples(
    silence_spans_col: list[list[dict]],
    words_col: list[list[dict]],
    durations_col: list[float],
    *,
    include_holds: bool = True,
) -> list[_Example]:
    """Expand each turn row into its ``eot`` (+ optional ``hold``) examples.

    A hold example ends the audio at the pause's ``end`` and transcribes only the words that
    finished at/before the pause's ``start`` (later words are not yet spoken at that point).
    """
    plan: list[_Example] = []
    for i, (spans, words, duration) in enumerate(
        zip(silence_spans_col, words_col, durations_col)
    ):
        if not spans:
            continue
        full_transcript = " ".join(w["word"] for w in words).strip()
        if include_holds:
            for span in spans[:-1]:  # every span except the last (the EOT) is a hold
                said = [w["word"] for w in words if w["end"] <= span["start"] + 1e-6]
                plan.append(
                    _Example(
                        row_idx=i,
                        end_sec=float(span["end"]),
                        marker="hold",
                        transcript=" ".join(said).strip(),
                    )
                )
        plan.append(
            _Example(row_idx=i, end_sec=float(duration), marker="eot", transcript=full_transcript)
        )
    return plan


class SemanticVADDataset(torch.utils.data.Dataset):
    """Map-style dataset over the expanded (turn → decision-point) examples."""

    def __init__(
        self,
        parquet_files,
        tokenizer,
        feature_extractor,
        special_ids: SpecialIds,
        marker_ids: dict[str, int],
        *,
        block_size: int = 4096,
        include_holds: bool = True,
        supervise_transcript: bool = True,
        system: str = DEFAULT_SYSTEM,
    ):
        from datasets import Audio, load_dataset

        if isinstance(parquet_files, str):
            parquet_files = [parquet_files]
        ds = load_dataset("parquet", data_files=list(parquet_files), split="train")
        # decode=False keeps audio as raw {bytes, path}; we decode with soundfile, so no
        # torch/torchcodec is dragged in by the datasets Audio feature (repo convention).
        self.ds = ds.cast_column("audio", Audio(decode=False))

        self.tokenizer = tokenizer
        self.fe = feature_extractor
        self.ids = special_ids
        self.marker_ids = marker_ids
        self.block_size = block_size
        self.supervise_transcript = supervise_transcript
        self.system = system

        self.plan = plan_examples(
            self.ds["silence_spans"],
            self.ds["words"],
            self.ds["duration"],
            include_holds=include_holds,
        )

    def __len__(self) -> int:
        return len(self.plan)

    def _encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def __getitem__(self, i: int):
        ex = self.plan[i]
        row = self.ds[ex.row_idx]

        array, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
        if array.ndim > 1:
            array = array.mean(axis=1)
        end = int(round(ex.end_sec * sr))
        if end > 0:
            array = array[:end]
        if len(array) < _MIN_SAMPLES:
            return None

        feat = self.fe(
            array,
            sampling_rate=sr,
            return_attention_mask=True,
            padding="max_length",
            return_tensors="np",
        )
        input_features = feat["input_features"][0].astype(np.float32)  # [n_mels, 3000]
        feature_attention_mask = np.asarray(feat["attention_mask"][0], dtype=np.int64)
        n_audio = num_audio_tokens(int(feature_attention_mask.sum()))
        if n_audio <= 0:
            return None

        tokens = build_example(
            self._encode,
            self.ids,
            marker_id=self.marker_ids[ex.marker],
            n_audio=n_audio,
            transcript=ex.transcript,
            system=self.system,
            supervise_transcript=self.supervise_transcript,
        )
        if len(tokens["input_ids"]) > self.block_size:
            return None  # too long to pack; drop rather than truncate audio placeholders

        return {
            "input_ids": np.asarray(tokens["input_ids"], dtype=np.int64),
            "labels": np.asarray(tokens["labels"], dtype=np.int64),
            "position_ids": np.asarray(tokens["position_ids"], dtype=np.int64),
            "input_features": input_features,
            "feature_attention_mask": feature_attention_mask,
        }


def collate_packed(batch):
    """Multipack a micro-batch into one flash-attn *varlen* sequence.

    Concatenates all examples' tokens into ``[1, total_len]`` and builds the cumulative
    sequence-length boundaries the varlen kernel needs — mirroring ``qwen3_adamw.py``. The
    per-clip mel features are stacked on dim 0 in the same order the ``<|AUDIO|>``
    placeholders appear, so the model's flattened audio embeddings line up with them.
    """
    batch = [b for b in batch if b is not None and len(b["input_ids"]) > 0]
    if not batch:
        return None

    input_ids = np.concatenate([b["input_ids"] for b in batch])
    labels = np.concatenate([b["labels"] for b in batch])
    position_ids = np.concatenate([b["position_ids"] for b in batch])
    lengths = [len(b["input_ids"]) for b in batch]
    cu, max_len = build_cu_seqlens(lengths)

    input_features = np.stack([b["input_features"] for b in batch])  # [n_clips, n_mels, 3000]
    feature_attention_mask = np.stack([b["feature_attention_mask"] for b in batch])

    return {
        "input_ids": torch.from_numpy(input_ids)[None],  # [1, total_len]
        "labels": torch.from_numpy(labels)[None],
        "position_ids": torch.from_numpy(position_ids)[None],
        "input_features": torch.from_numpy(input_features),
        "feature_attention_mask": torch.from_numpy(feature_attention_mask),
        "cu_seq_lens_q": torch.tensor(cu, dtype=torch.int32),
        "cu_seq_lens_k": torch.tensor(cu, dtype=torch.int32),
        "max_length_q": int(max_len),
        "max_length_k": int(max_len),
    }
