"""Turn segmentation and silence-span labeling -- the heart of the pipeline.

Given a normalized ``list[Word]`` and the source recording duration, produce one or
more :class:`Turn` objects whose ``silence_spans`` follow the eot-bench rule: every span
except the last is a mid-turn ``hold``; the last is the true ``eot``.

Two strategies (see :class:`~semantic_vad.schema.TurnConfig`):

* ``mode="single"`` -- the whole utterance is one turn. Every internal pause >=
  ``min_silence`` is a genuine ``hold`` (the speaker had not finished), and the end of the
  utterance is a genuine ``eot``. The label comes from utterance structure, not gap size,
  so it is *not* trivially recoverable from silence duration -- the point of semantic VAD.

* ``mode="segment"`` -- a long recording is split into pseudo-turns wherever a gap is
  >= ``turn_gap``. Use this only when a row is a continuous monologue. Note the caveat: the
  turn boundary is defined *by* silence duration, so labels correlate with gap length.
"""

from __future__ import annotations

from .schema import SilenceSpan, Turn, TurnConfig, Word


def compute_gaps(words: list[Word]) -> list[float]:
    """Inter-word gaps in seconds (``words[i+1].start - words[i].end``), clamped >= 0.

    Length is ``len(words) - 1``; ``gaps[i]`` is the gap *after* ``words[i]``.
    """
    return [max(0.0, words[i + 1].start - words[i].end) for i in range(len(words) - 1)]


def _holds_within(words: list[Word], lo: int, hi: int, min_silence: float,
                  max_gap: float | None) -> list[SilenceSpan]:
    """Hold spans for gaps strictly inside word range ``[lo, hi]`` (inclusive indices).

    A gap qualifies when ``min_silence <= gap`` and, if ``max_gap`` is set, ``gap < max_gap``
    (so a turn-boundary-sized gap is never emitted as a mid-turn hold).
    """
    holds: list[SilenceSpan] = []
    for i in range(lo, hi):
        gap = words[i + 1].start - words[i].end
        if gap < min_silence:
            continue
        if max_gap is not None and gap >= max_gap:
            continue
        holds.append(SilenceSpan(start=words[i].end, end=words[i + 1].start))
    return holds


def _finish_turn(words: list[Word], lo: int, hi: int, cfg: TurnConfig,
                 next_word_start: float | None, audio_duration: float,
                 max_gap: float | None) -> Turn:
    """Assemble a :class:`Turn` from word range ``[lo, hi]`` (inclusive)."""
    turn_words = words[lo : hi + 1]
    holds = _holds_within(words, lo, hi, cfg.min_silence, max_gap)

    last_end = words[hi].end
    # How much real silence follows this turn in the source recording?
    if next_word_start is not None:
        available = next_word_start - last_end
    else:
        available = audio_duration - last_end
    # Clamp into [eot_trailing, max_trailing]; short trailing -> pad, long -> trim.
    used_trailing = min(max(available, cfg.eot_trailing), cfg.max_trailing)
    eot = SilenceSpan(start=last_end, end=last_end + used_trailing)

    window_start = max(0.0, words[lo].start - cfg.lead_in)
    window_end = eot.end
    return Turn(
        words=turn_words,
        silence_spans=holds + [eot],
        window_start=window_start,
        window_end=window_end,
    )


def build_turns(words: list[Word], audio_duration: float, cfg: TurnConfig | None = None) -> list[Turn]:
    """Segment ``words`` into turns and label their silence spans.

    Parameters
    ----------
    words:
        Normalized, time-sorted words for one source recording.
    audio_duration:
        Length of the source recording in seconds (used for trailing silence).
    cfg:
        A :class:`TurnConfig`; defaults to ``TurnConfig()`` (single-turn, 100 ms floor).

    Returns turns that satisfy the ``min_words`` / ``min_hold_spans`` filters.
    """
    cfg = cfg or TurnConfig()
    words = [w for w in words if w.end > w.start or w.duration == 0.0]
    if not words:
        return []

    if cfg.mode == "single":
        boundaries = [(0, len(words) - 1)]
        max_gap = None
    else:  # segment: cut at gaps >= turn_gap
        boundaries = []
        lo = 0
        for i in range(len(words) - 1):
            gap = words[i + 1].start - words[i].end
            if gap >= cfg.turn_gap:
                boundaries.append((lo, i))
                lo = i + 1
        boundaries.append((lo, len(words) - 1))
        max_gap = cfg.turn_gap

    turns: list[Turn] = []
    for bi, (lo, hi) in enumerate(boundaries):
        # The word that starts the *next* turn defines this turn's trailing silence.
        next_start = words[hi + 1].start if hi + 1 < len(words) else None
        turn = _finish_turn(words, lo, hi, cfg, next_start, audio_duration, max_gap)
        if len(turn.words) < cfg.min_words:
            continue
        if turn.n_holds < cfg.min_hold_spans:
            continue
        turns.append(turn)
    return turns
