"""Offline tests for the torch-free training helpers (prompt/label build + varlen packing).

These import only :mod:`semantic_vad.training.prompt` / ``.packing`` — no torch,
transformers, or network — so they run anywhere the other unit tests do. The heavy pieces
(modeling/data/train) require the ``train`` extra + a GPU and are exercised on the pod.
"""

from semantic_vad.training.packing import build_cu_seqlens
from semantic_vad.training.prompt import (
    IGNORE_INDEX,
    SpecialIds,
    build_example,
    num_audio_tokens,
)

# Special ids chosen above ord() of any ASCII char so the fake encoder can't collide.
IDS = SpecialIds(im_start=1000, im_end=1001, audio_bos=1002, audio_eos=1003, audio=1004)
EOT_ID, HOLD_ID = 1005, 1006


def _encode(text):
    """Deterministic char-level fake tokenizer (no specials, all ids < 128)."""
    return [ord(c) for c in text]


def test_num_audio_tokens_matches_whisper_downsample():
    assert num_audio_tokens(3000) == 1500  # full 30 s window -> 1500 frames
    assert num_audio_tokens(1) == 1
    assert num_audio_tokens(0) == 0
    assert num_audio_tokens(-5) == 0


def test_build_example_shapes_align():
    ex = build_example(_encode, IDS, marker_id=EOT_ID, n_audio=7, transcript="hi there")
    assert len(ex["input_ids"]) == len(ex["labels"]) == len(ex["position_ids"])
    assert ex["position_ids"] == list(range(len(ex["input_ids"])))
    # Exactly n_audio placeholder tokens got inserted.
    assert ex["input_ids"].count(IDS.audio) == 7


def test_marker_is_first_supervised_token():
    ex = build_example(_encode, IDS, marker_id=EOT_ID, n_audio=3, transcript="okay")
    supervised = [i for i, y in enumerate(ex["labels"]) if y != IGNORE_INDEX]
    first = supervised[0]
    assert ex["labels"][first] == EOT_ID
    assert ex["input_ids"][first] == EOT_ID
    # The last supervised token is <|im_end|> (teach the model to stop).
    assert ex["labels"][supervised[-1]] == IDS.im_end


def test_only_assistant_span_is_supervised():
    ex = build_example(_encode, IDS, marker_id=HOLD_ID, n_audio=5, transcript="one two")
    # No audio placeholder, control token, or prompt token is ever a loss target.
    for tok, lab in zip(ex["input_ids"], ex["labels"]):
        if lab != IGNORE_INDEX:
            assert tok not in (IDS.audio, IDS.audio_bos, IDS.audio_eos, IDS.im_start)
    # Supervised span = marker + " one two" + im_end = 1 + 8 + 1 = 10 tokens.
    n_supervised = sum(1 for y in ex["labels"] if y != IGNORE_INDEX)
    assert n_supervised == 1 + len(_encode(" one two")) + 1


def test_supervise_transcript_false_leaves_only_marker_and_end():
    ex = build_example(
        _encode, IDS, marker_id=EOT_ID, n_audio=2, transcript="ignored",
        supervise_transcript=False,
    )
    supervised = [ex["input_ids"][i] for i, y in enumerate(ex["labels"]) if y != IGNORE_INDEX]
    assert supervised == [EOT_ID, IDS.im_end]


def test_build_cu_seqlens():
    cu, max_len = build_cu_seqlens([3, 5, 2])
    assert cu == [0, 3, 8, 10]  # ends at total token count
    assert max_len == 5
    assert build_cu_seqlens([]) == ([0], 0)
