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

# Sentence-final punctuation (Latin + a few CJK/Arabic variants) for "sentence" mode.
_SENT_END = tuple(".?!…؟。！？")
# Words that end in "." but are NOT sentence boundaries (abbreviations / titles).
_ABBREV = {
    "dr", "prof", "assoc", "mr", "mrs", "ms", "no", "bil", "hj", "hjh", "en", "pn", "tn",
    "tuan", "puan", "dato", "datuk", "datin", "tan", "sri", "ir", "ustaz", "ustazah",
    "yb", "yab", "ybhg", "st", "jln", "kg", "vs", "etc", "e.g", "i.e", "a.m", "p.m",
}


def _is_sentence_end(token: str) -> bool:
    tok = token.strip()
    if not tok or tok[-1] not in _SENT_END:
        return False
    core = tok.rstrip("".join(_SENT_END)).strip()
    if not core:
        return False
    # Guard abbreviations and single-letter initials (e.g. "Dr.", "A.", "R.").
    letters = core.replace(".", "")
    if len(letters) <= 1:
        return False
    if core.lower().strip(".") in _ABBREV:
        return False
    return True


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
                 next_word_start: float | None, prev_word_end: float | None,
                 audio_duration: float, max_gap: float | None) -> Turn:
    """Assemble a :class:`Turn` from word range ``[lo, hi]`` (inclusive)."""
    turn_words = words[lo : hi + 1]
    holds = _holds_within(words, lo, hi, cfg.min_silence, max_gap)

    last_end = words[hi].end
    if next_word_start is not None:
        # Mid-recording turn: trailing is the real silence gap before the next turn's word.
        # Cap at that gap (minus a guard) so the clip never runs into the next word's onset.
        available = next_word_start - last_end
        used_trailing = max(0.0, min(available, cfg.max_trailing) - cfg.eot_guard)
    else:
        # Last turn / isolated utterance: pad up to eot_trailing (synthetic trailing silence).
        available = audio_duration - last_end
        used_trailing = min(max(available, cfg.eot_trailing), cfg.max_trailing)
    eot = SilenceSpan(start=last_end, end=last_end + used_trailing)

    if prev_word_end is not None:
        # Symmetric guard at the start: don't reach back into the previous turn's word.
        gap_before = words[lo].start - prev_word_end
        lead = max(0.0, min(cfg.lead_in, gap_before) - cfg.eot_guard)
        window_start = words[lo].start - lead
    else:
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
    elif cfg.mode == "segment":  # cut at gaps >= turn_gap
        boundaries = []
        lo = 0
        for i in range(len(words) - 1):
            gap = words[i + 1].start - words[i].end
            if gap >= cfg.turn_gap:
                boundaries.append((lo, i))
                lo = i + 1
        boundaries.append((lo, len(words) - 1))
        max_gap = cfg.turn_gap
    else:  # sentence: cut at sentence-final punctuation OR a >= turn_gap silence
        boundaries = []
        lo = 0
        for i in range(len(words)):
            gap = words[i + 1].start - words[i].end if i + 1 < len(words) else None
            if _is_sentence_end(words[i].word) or (gap is not None and gap >= cfg.turn_gap):
                boundaries.append((lo, i))
                lo = i + 1
        if lo < len(words):
            boundaries.append((lo, len(words) - 1))
        max_gap = cfg.turn_gap

    turns: list[Turn] = []
    for bi, (lo, hi) in enumerate(boundaries):
        # Neighboring words bound this turn's trailing (next) and lead-in (previous).
        next_start = words[hi + 1].start if hi + 1 < len(words) else None
        prev_end = words[lo - 1].end if lo > 0 else None
        turn = _finish_turn(words, lo, hi, cfg, next_start, prev_end, audio_duration, max_gap)
        if len(turn.words) < cfg.min_words:
            continue
        if turn.n_holds < cfg.min_hold_spans:
            continue
        turns.append(turn)
    return turns
