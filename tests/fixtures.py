"""Real sample data captured from the source datasets (no network needed for tests)."""

# malaysia-ai/Malaysian-STT, config=malaysian, mode=whole, level=word (prefix of row 1).
MALAYSIAN_WORD_TEXT = (
    "<|0.04|> Collab<|0.22|><|0.26|> dia<|0.30|><|0.34|> tak<|0.38|>"
    "<|0.42|> boleh<|0.50|><|0.54|> kerja<|0.70|><|1.06|> Tak<|1.16|>"
    "<|1.20|> boleh<|1.36|><|1.64|> Kena<|1.80|><|2.02|> ambil<|2.14|>"
)

# malaysia-ai/Malaysian-STT, segment level (one phrase between two timestamps).
MALAYSIAN_SEGMENT_TEXT = (
    "<|0.04|> Collab dia tak boleh kerja Tak boleh Kena ambil yang business class jugalah<|3.42|>"
)

# AAdonis/multilingual_audio_alignments, config=english, row 0 "words" column.
MULTILINGUAL_WORDS = [
    {"word": "lots", "start": 0.11, "end": 0.7},
    {"word": "of", "start": 0.7, "end": 0.91},
    {"word": "good", "start": 0.91, "end": 1.09},
    {"word": "ideas", "start": 1.09, "end": 1.66},
    {"word": "here", "start": 1.66, "end": 2.02},
    {"word": "and", "start": 2.38, "end": 2.55},
    {"word": "they", "start": 2.55, "end": 2.67},
    {"word": "don't", "start": 2.67, "end": 2.96},
    {"word": "stray", "start": 2.96, "end": 3.57},
    {"word": "into", "start": 3.57, "end": 3.9},
    {"word": "the", "start": 3.9, "end": 3.99},
    {"word": "weird", "start": 3.99, "end": 4.45},
    {"word": "territory", "start": 4.45, "end": 5.04},
    {"word": "i", "start": 5.73, "end": 5.83},
    {"word": "was", "start": 5.83, "end": 6.03},
    {"word": "drawing", "start": 6.03, "end": 6.41},
    {"word": "a", "start": 6.41, "end": 6.54},
    {"word": "blank", "start": 6.65, "end": 7.18},
]
# Audio for that row is ~9.31s long (last word "good" ends at 9.31 in the full row).
MULTILINGUAL_DURATION = 7.5
