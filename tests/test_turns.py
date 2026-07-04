import numpy as np

from semantic_vad.audio import slice_window, turn_to_row
from semantic_vad.parsers import normalize_words, parse_whisper_timestamps
from semantic_vad.schema import TurnConfig
from semantic_vad.turns import build_turns, compute_gaps
from tests.fixtures import (
    MALAYSIAN_WORD_TEXT,
    MULTILINGUAL_DURATION,
    MULTILINGUAL_WORDS,
)


def test_compute_gaps():
    words = normalize_words(MULTILINGUAL_WORDS)
    gaps = compute_gaps(words)
    assert len(gaps) == len(words) - 1
    assert all(g >= 0 for g in gaps)
    # "here"->"and" is a 0.36s pause.
    assert round(gaps[4], 2) == 0.36


def test_single_mode_labeling():
    words = normalize_words(MULTILINGUAL_WORDS)
    cfg = TurnConfig(mode="single", min_silence=0.1, eot_trailing=0.5)
    turns = build_turns(words, MULTILINGUAL_DURATION, cfg)
    assert len(turns) == 1
    t = turns[0]
    # 3 internal pauses >= 100ms + 1 EOT span.
    assert len(t.silence_spans) == 4
    assert t.n_holds == 3
    # eot-bench rule: the last span starts after the last word (the EOT).
    eot = t.silence_spans[-1]
    assert eot.start == words[-1].end
    assert round(eot.duration, 3) == 0.5
    # Holds are strictly before the EOT and ordered.
    starts = [s.start for s in t.silence_spans]
    assert starts == sorted(starts)
    assert all(s.start < eot.start for s in t.silence_spans[:-1])


def test_segment_mode_splits_on_turn_gap():
    words = parse_whisper_timestamps(MALAYSIAN_WORD_TEXT)
    cfg = TurnConfig(mode="segment", min_silence=0.1, turn_gap=0.3, eot_trailing=0.5)
    turns = build_turns(words, audio_duration=2.5, cfg=cfg)
    # The 0.36s gap after "kerja" is the only >= turn_gap -> exactly two turns.
    assert len(turns) == 2
    assert turns[0].words[-1].word == "kerja"
    assert turns[1].words[0].word == "Tak"
    # Turn 2 has two mid-turn holds (0.28s, 0.22s) then its EOT.
    assert turns[1].n_holds == 2
    # No hold is ever as long as turn_gap (those become boundaries, not holds).
    for t in turns:
        for hold in t.silence_spans[:-1]:
            assert hold.duration < cfg.turn_gap


def test_min_hold_spans_filter():
    words = normalize_words(MULTILINGUAL_WORDS)
    cfg = TurnConfig(mode="single", min_hold_spans=5)  # only 3 holds exist
    assert build_turns(words, MULTILINGUAL_DURATION, cfg) == []


def test_turn_to_row_rezeroes_and_pads():
    words = normalize_words(MULTILINGUAL_WORDS)
    cfg = TurnConfig(mode="single", lead_in=0.3, eot_trailing=0.5)
    turn = build_turns(words, MULTILINGUAL_DURATION, cfg)[0]

    sr = 16000
    # Source recording only 7.2s long -> EOT window (to 7.68s) must zero-pad.
    array = np.random.default_rng(0).standard_normal(int(7.2 * sr)).astype(np.float32)
    row = turn_to_row(turn, array, sr, row_id="en__x__turn_000", language="en")

    # Clip length matches window and duration is consistent.
    assert abs(row.duration - (turn.window_end - turn.window_start)) < 1e-3
    assert len(row.audio) == round(row.duration * sr)
    # First word window_start is 0 (0.11 - 0.3 clamped), so times are unchanged here.
    assert row.words[0]["start"] == round(words[0].start - turn.window_start, 3)
    # All spans fall within the clip.
    assert all(0 <= s["start"] < s["end"] <= row.duration + 1e-6 for s in row.silence_spans)
    # The tail is zero-padded silence (the EOT region beyond the source audio).
    assert np.allclose(row.audio[-int(0.3 * sr):], 0.0)


def test_slice_window_pads_past_end():
    sr = 16000
    a = np.ones(sr, dtype=np.float32)  # 1 second of ones
    out = slice_window(a, sr, 0.5, 2.0)  # ask for 1.5s starting at 0.5s
    assert len(out) == int(1.5 * sr)
    assert np.allclose(out[: int(0.5 * sr)], 1.0)   # real audio
    assert np.allclose(out[int(0.5 * sr):], 0.0)    # padded silence
