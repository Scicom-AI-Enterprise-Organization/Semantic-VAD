"""Semantic-VAD: build eot-bench-compatible end-of-turn datasets from forced alignments.

The package turns word-level forced-alignment corpora into rows shaped like
`livekit/eot-bench-data`: each row is a single user *turn* with an audio clip, the
words inside it, and a list of `silence_spans`. Following the eot-bench convention,
the **last** silence span is the true end-of-turn (`eot`) and every earlier span is a
mid-turn pause (`hold`); labels are derived from ordering, not stored.

See `README.md` for the full methodology.
"""

from .schema import Word, SilenceSpan, Turn, EOTRow, TurnConfig
from .parsers import parse_whisper_timestamps, normalize_words
from .turns import build_turns, compute_gaps

__all__ = [
    "Word",
    "SilenceSpan",
    "Turn",
    "EOTRow",
    "TurnConfig",
    "parse_whisper_timestamps",
    "normalize_words",
    "build_turns",
    "compute_gaps",
]
