"""`eot-harness` batch adapter for the Qwen2-Audio classification-head EoT model.

See https://github.com/livekit/eot-bench (Adapter Contracts section). Usage:

    eot-harness predict --path Scicom-intl/semantic-vad-eot --name en --split test \\
      --adapter semvad.eot_adapter:Qwen2AudioEoTAdapter --output-dir output
"""

from __future__ import annotations

import math
import os
from typing import Optional

import torch
from eot_harness.io import (
    DEFAULT_INFERENCE_INTERVAL,
    DEFAULT_MIN_SILENCE,
    DEFAULT_TRANSCRIPT_LAG,
    _time_grid,
    build_messages,
    decode_audio,
)
from eot_harness.streaming_stt import build_event_prediction_rows
from transformers import AutoProcessor

from semvad.modeling import DEFAULT_MODEL_NAME, Qwen2AudioEoTClassifier

# Languages we have training data for (see README §3) -- restrict harness runs to
# these unless a checkpoint explicitly supports more.
TRAINED_LANGUAGES = {"de", "en", "es", "fr", "it", "ja", "ko", "pt", "tr", "zh"}

_EPS = 1e-6


class Qwen2AudioEoTAdapter:
    adapter_id = "qwen2audio-eot-head"
    score_point = 0.2

    def __init__(
        self,
        checkpoint_dir: Optional[str] = None,
        model_name: str = DEFAULT_MODEL_NAME,
        device: Optional[str] = None,
        dtype: torch.dtype = torch.bfloat16,
        supported_languages: Optional[set] = None,
        transcript_lag: float = DEFAULT_TRANSCRIPT_LAG,
    ):
        checkpoint_dir = checkpoint_dir or os.environ.get("EOT_CHECKPOINT_DIR")
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = Qwen2AudioEoTClassifier.from_pretrained(model_name, dtype=dtype)
        if checkpoint_dir:
            self.model.load_adapter(checkpoint_dir)
        self.model.to(device)
        self.model.eval()

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.processor.tokenizer.padding_side = "right"
        self._supported_languages = supported_languages or TRAINED_LANGUAGES
        self.transcript_lag = transcript_lag

    def supports_language(self, lang_code: str) -> bool:
        return lang_code in self._supported_languages

    def _score_one(self, item: dict) -> float:
        audio = item["audio"]
        messages = item.get("messages") or []
        prior_text = " ".join(m["content"] for m in messages if m.get("role") == "user")
        return self.model.predict_p_eot(
            self.processor,
            audio["array"],
            audio["sampling_rate"],
            prior_text=prior_text,
        )

    def predict_batch(self, batch: list[dict]) -> list[float]:
        # One forward pass per item for now -- correctness first. Qwen2-Audio's
        # feature extractor pads every clip to a fixed 30s mel grid regardless of
        # actual length, so naive padding-to-longest batching wastes little extra
        # compute relative to per-item calls; batching is a throughput follow-up,
        # not a correctness concern.
        return [self._score_one(item) for item in batch]

    def _query_timestamps(self, row: dict, inference_interval: float) -> list[float]:
        timestamps: set[float] = set()
        for span in row["silence_spans"]:
            start = float(span["start"])
            end = float(span["end"])
            if end - start < DEFAULT_MIN_SILENCE - _EPS:
                continue
            timestamps.update(_time_grid(start, end, inference_interval))
        return sorted(timestamps)

    def predict_turn(
        self,
        row: dict,
        *,
        inference_interval: float = DEFAULT_INFERENCE_INTERVAL,
    ) -> dict:
        array, sample_rate = decode_audio(row["audio"])
        events = []
        for timestamp in self._query_timestamps(row, inference_interval):
            end_sample = int(math.floor(timestamp * sample_rate + _EPS))
            messages = build_messages(row, timestamp, transcript_lag=self.transcript_lag)
            prior_text = " ".join(m["content"] for m in messages if m.get("role") == "user")
            p_eot = self.model.predict_p_eot(
                self.processor,
                array[:end_sample],
                sample_rate,
                prior_text=prior_text,
            )
            events.append({"timestamp": timestamp, "p_eot": p_eot})

        return {
            "id": row["id"],
            "audio_sec": float(len(array) / sample_rate),
            "events": events,
            "prediction_rows": build_event_prediction_rows(
                row,
                events,
                inference_interval=inference_interval,
            ),
        }
