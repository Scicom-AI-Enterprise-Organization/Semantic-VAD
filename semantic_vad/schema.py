"""Core data structures and the tunable turn-building configuration.

Coordinate convention
----------------------
Inside a :class:`Turn`, word and silence-span times are absolute in the *source*
recording. The dataset builder re-zeroes them to the emitted audio clip (subtracts
``window_start``) when it produces an :class:`EOTRow`, so a row's times are relative
to its own clip -- exactly like `livekit/eot-bench-data`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Word:
    """A force-aligned word with start/end times in seconds."""

    word: str
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict[str, Any]:
        return {"word": self.word, "start": round(self.start, 3), "end": round(self.end, 3)}


@dataclass
class SilenceSpan:
    """A silence interval that is a decision point for the EOT model.

    A span is a ``hold`` (mid-turn pause) unless it is the last span in a turn, in
    which case it is the ``eot`` (true end of turn). The label is positional and is
    assigned by the eot-bench harness at evaluation time, not stored here.
    """

    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict[str, Any]:
        return {"start": round(self.start, 3), "end": round(self.end, 3)}


@dataclass
class Turn:
    """One user turn (pre-audio-slicing), in source-absolute coordinates.

    ``silence_spans`` is ordered by time; the final element is the EOT span.
    ``window_start``/``window_end`` define the audio clip to cut from the source
    recording (``window_end`` may exceed the recording length, meaning the clip is
    zero-padded so a real trailing EOT silence exists).
    """

    words: list[Word]
    silence_spans: list[SilenceSpan]
    window_start: float
    window_end: float

    @property
    def n_holds(self) -> int:
        return max(0, len(self.silence_spans) - 1)

    @property
    def transcript(self) -> str:
        return " ".join(w.word for w in self.words).strip()


@dataclass
class EOTRow:
    """A finished dataset row, schema-compatible with `livekit/eot-bench-data`.

    ``audio`` is a mono float32 numpy array at ``sampling_rate`` Hz. When written to
    parquet it is cast to a HuggingFace ``Audio`` feature (WAV bytes).
    """

    id: str
    audio: Any  # np.ndarray, kept as Any to avoid a hard numpy import here
    sampling_rate: int
    language: str
    duration: float
    silence_spans: list[dict[str, float]]
    words: list[dict[str, Any]]
    messages: list[dict[str, str]] = field(default_factory=list)

    def to_record(self) -> dict[str, Any]:
        """Row dict ready for ``datasets.Dataset.from_list`` (audio as array+sr)."""
        return {
            "id": self.id,
            "audio": {"array": self.audio, "sampling_rate": self.sampling_rate},
            "language": self.language,
            "duration": round(self.duration, 3),
            "silence_spans": self.silence_spans,
            "words": self.words,
            "messages": self.messages,
        }


@dataclass
class TurnConfig:
    """Parameters that control how alignments become turns and silence spans.

    Attributes
    ----------
    min_silence:
        Minimum gap (seconds) to count as a silence span / decision point. Matches the
        eot-bench 100 ms floor. Gaps below this are ignored (normal between-word timing).
    mode:
        ``"single"`` treats the whole input utterance as one turn (best when each source
        row is already one complete utterance). ``"segment"`` splits a long recording into
        pseudo-turns at gaps >= ``turn_gap``. ``"sentence"`` splits at sentence-final
        punctuation (``. ? !``, with an abbreviation guard) and also at gaps >= ``turn_gap``
        -- best for punctuated transcripts (e.g. Malaysian-STT), since continuous speech has
        no clean gap threshold.
    turn_gap:
        In ``"segment"`` mode a gap this long ends a turn; in ``"sentence"`` mode it is the
        fallback boundary for un-punctuated stretches.
    eot_trailing:
        Trailing silence (seconds) to include after the last word as the EOT span. If the
        source audio has less trailing silence than this, the clip is zero-padded.
    max_trailing:
        Cap on trailing silence included in the EOT span, so long dead air is trimmed.
    lead_in:
        Audio context (seconds) kept before the first word of a turn window.
    eot_guard:
        Safety margin (seconds) kept away from a neighboring word when a turn borders other
        speech in a continuous recording (``segment``/``sentence`` mode). Trailing/lead-in are
        capped at the real silence gap minus this guard, so a clip never captures the next or
        previous word's onset (avoids "partial word" sounds).
    min_words:
        Drop turns with fewer words than this.
    min_hold_spans:
        Keep only turns with at least this many mid-turn ``hold`` spans (0 = keep all).
        Raise to 1+ to focus on the "hard" rows that break naive silence-VAD.
    """

    min_silence: float = 0.1
    mode: str = "single"
    turn_gap: float = 0.7
    eot_trailing: float = 0.5
    max_trailing: float = 1.0
    lead_in: float = 0.3
    eot_guard: float = 0.05
    min_words: int = 1
    min_hold_spans: int = 0

    def __post_init__(self) -> None:
        if self.mode not in ("single", "segment", "sentence"):
            raise ValueError(f"mode must be 'single', 'segment' or 'sentence', got {self.mode!r}")
        if self.eot_trailing > self.max_trailing:
            raise ValueError("eot_trailing must be <= max_trailing")
        if self.mode in ("segment", "sentence") and self.turn_gap <= self.min_silence:
            raise ValueError("turn_gap must be > min_silence")
