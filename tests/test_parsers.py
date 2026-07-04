from semantic_vad.parsers import normalize_words, parse_whisper_timestamps
from tests.fixtures import (
    MALAYSIAN_SEGMENT_TEXT,
    MALAYSIAN_WORD_TEXT,
    MULTILINGUAL_WORDS,
)


def test_parse_whisper_word_level():
    words = parse_whisper_timestamps(MALAYSIAN_WORD_TEXT)
    assert [w.word for w in words[:3]] == ["Collab", "dia", "tak"]
    assert words[0].start == 0.04 and words[0].end == 0.22
    # A real within-turn pause: "kerja" ends 0.70, "Tak" starts 1.06 -> 0.36s gap.
    kerja = next(w for w in words if w.word == "kerja")
    tak = next(w for w in words if w.word == "Tak")
    assert round(tak.start - kerja.end, 2) == 0.36
    # Every word is well-formed and time-ordered.
    assert all(w.end >= w.start for w in words)
    assert all(words[i].start <= words[i + 1].start for i in range(len(words) - 1))


def test_parse_whisper_segment_level():
    segs = parse_whisper_timestamps(MALAYSIAN_SEGMENT_TEXT)
    assert len(segs) == 1
    assert segs[0].start == 0.04 and segs[0].end == 3.42
    assert segs[0].word.startswith("Collab dia tak")


def test_parse_whisper_empty_and_garbage():
    assert parse_whisper_timestamps("") == []
    assert parse_whisper_timestamps("no timestamps here") == []
    assert parse_whisper_timestamps("<|0.1|>") == []  # dangling, no closing ts


def test_normalize_words_shape_and_sort():
    words = normalize_words(MULTILINGUAL_WORDS)
    assert len(words) == len(MULTILINGUAL_WORDS)
    assert words[0].word == "lots"
    assert all(words[i].start <= words[i + 1].start for i in range(len(words) - 1))


def test_normalize_words_tolerates_aliases_and_bad_rows():
    raw = [
        {"text": "hi", "start": 0.0, "end": 0.5},   # 'text' alias
        {"word": "", "start": 1.0, "end": 1.2},       # empty -> dropped
        {"word": "x", "start": None, "end": 1.0},     # missing time -> dropped
        {"word": "ok", "start": "0.6", "end": "0.9"}, # string floats -> coerced
    ]
    words = normalize_words(raw)
    assert [w.word for w in words] == ["hi", "ok"]
