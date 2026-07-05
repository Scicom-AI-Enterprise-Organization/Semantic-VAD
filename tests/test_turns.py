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


def test_sentence_mode_splits_on_punctuation_not_abbrev():
    from semantic_vad.schema import Word
    # "Dr." must NOT split; "terperinci." must; a >=turn_gap gap must.
    words = [
        Word("Prof.", 0.0, 0.4), Word("Dr.", 0.5, 0.8), Word("Ramli", 0.9, 1.3),
        Word("hadir.", 1.4, 1.9),                        # sentence end
        Word("Sesi", 2.0, 2.3), Word("pagi", 2.4, 2.7),  # next sentence
        Word("ini.", 2.8, 3.1),                          # sentence end
    ]
    cfg = TurnConfig(mode="sentence", turn_gap=1.5, min_silence=0.1)
    turns = build_turns(words, audio_duration=4.0, cfg=cfg)
    assert len(turns) == 2
    assert turns[0].transcript == "Prof. Dr. Ramli hadir."   # abbreviations didn't split
    assert turns[1].transcript == "Sesi pagi ini."
    # every turn ends with an EOT span
    assert all(t.silence_spans[-1].start == t.words[-1].end for t in turns)


def test_sentence_mode_gap_fallback():
    from semantic_vad.schema import Word
    # no punctuation, but a 2s silence -> fallback split
    words = [Word("satu", 0.0, 0.4), Word("dua", 0.6, 1.0),
             Word("tiga", 3.2, 3.6), Word("empat", 3.7, 4.1)]
    cfg = TurnConfig(mode="sentence", turn_gap=1.5, min_silence=0.1)
    turns = build_turns(words, audio_duration=5.0, cfg=cfg)
    assert len(turns) == 2  # split at the 2.2s gap
    assert turns[0].transcript == "satu dua"


def test_trailing_never_reaches_next_word():
    from semantic_vad.schema import Word
    # "Hi." ends a sentence at 0.5; next word "Bye" starts at 0.8 (gap 0.3s). The old logic
    # forced eot_trailing=0.5 -> clip ran to 1.0s, into "Bye" (partial word). Now it must stop
    # before 0.8s.
    words = [Word("Hi.", 0.0, 0.5), Word("Bye", 0.8, 1.1)]
    cfg = TurnConfig(mode="sentence", turn_gap=1.5, eot_trailing=0.5, max_trailing=1.0,
                     eot_guard=0.05)
    turns = build_turns(words, audio_duration=2.0, cfg=cfg)
    eot = turns[0].silence_spans[-1]
    assert eot.start == 0.5
    assert eot.end <= 0.8                       # never into the next word
    assert abs(eot.duration - 0.25) < 1e-6      # min(gap 0.3, max 1.0) - guard 0.05


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
    row = turn_to_row(turn, array, sr, row_id="en__x__turn_000", language="en",
                      trim_trailing=False)

    # Clip length matches window and duration is consistent.
    assert abs(row.duration - (turn.window_end - turn.window_start)) < 1e-3
    assert len(row.audio) == round(row.duration * sr)
    # First word window_start is 0 (0.11 - 0.3 clamped), so times are unchanged here.
    assert row.words[0]["start"] == round(words[0].start - turn.window_start, 3)
    # All spans fall within the clip.
    assert all(0 <= s["start"] < s["end"] <= row.duration + 1e-6 for s in row.silence_spans)
    # The tail is zero-padded silence (the EOT region beyond the source audio).
    assert np.allclose(row.audio[-int(0.3 * sr):], 0.0)


def test_turn_to_row_vad_trims_tail_to_silence():
    from semantic_vad.schema import SilenceSpan, Turn, Word
    sr = 16000
    # 1s of tone (speech), then 1s of silence. Turn's last word ends at 1.0s; window runs to
    # 2.0s. VAD trim should end the clip shortly after 1.0s, in the silence (low tail RMS).
    rng = np.random.default_rng(0)
    speech = (0.3 * rng.standard_normal(sr)).astype(np.float32)
    array = np.concatenate([speech, np.zeros(sr, dtype=np.float32)])
    turn = Turn(words=[Word("hello", 0.0, 1.0)],
                silence_spans=[SilenceSpan(1.0, 2.0)],
                window_start=0.0, window_end=2.0)
    row = turn_to_row(turn, array, sr, row_id="x", language="en",
                      trim_trailing=True, trailing_pad=0.15)
    # clip ends ~1.15s (speech end 1.0 + 0.15 pad), well before the full 2.0s window
    assert 1.05 < row.duration < 1.35
    tail = np.asarray(row.audio)[-int(0.05 * sr):]
    assert float(np.sqrt(np.mean(tail ** 2))) < 0.02   # ends in silence, not a word
    assert row.silence_spans[-1]["end"] == round(row.duration, 3)


def test_slice_window_pads_past_end():
    sr = 16000
    a = np.ones(sr, dtype=np.float32)  # 1 second of ones
    out = slice_window(a, sr, 0.5, 2.0)  # ask for 1.5s starting at 0.5s
    assert len(out) == int(1.5 * sr)
    assert np.allclose(out[: int(0.5 * sr)], 1.0)   # real audio
    assert np.allclose(out[int(0.5 * sr):], 0.0)    # padded silence
