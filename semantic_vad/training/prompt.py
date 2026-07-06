"""Prompt / label construction for the Whisper-encoder → Qwen3 EOT model.

This module is **pure Python** (stdlib only) so it can be unit-tested offline and so
``import semantic_vad`` never pulls in torch/transformers. The heavy training code lives
in :mod:`semantic_vad.training.modeling`, :mod:`~.data` and :mod:`~.train`.

Sequence layout (Qwen chat, one training example)::

    <|im_start|>system\n{system}<|im_end|>\n
    <|im_start|>user\n<|audio_bos|>{<|AUDIO|> * n_audio}<|audio_eos|>{instruction}<|im_end|>\n
    <|im_start|>assistant\n{marker}[ {transcript}]<|im_end|>\n

Only the assistant span (``marker`` → ``<|im_end|>``) is supervised. The **marker is the
first assistant token** so a server can read the end-of-turn decision in a single probe
step (INTEGRATION.md): ``P(eot) = softmax([logp(<|eot|>), logp(<|hold|>)])[0]``. The
transcript that follows is *adapter-alignment scaffolding* (README) — CE loss over it
aligns the projection into Qwen3's embedding space; it is never emitted at serving time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

#: A tokenizer's ``encode(text) -> list[int]`` with special tokens **disabled**.
Encode = Callable[[str], "list[int]"]

#: Label id for positions that do not contribute to the loss.
IGNORE_INDEX = -100

#: Placeholder / marker token strings the tokenizer must know (added in train.py). The
#: ``<|audio_*|>`` / ``<|AUDIO|>`` names match Qwen2-Audio so the vLLM serving path in
#: INTEGRATION.md lines up 1:1.
AUDIO_TOKEN = "<|AUDIO|>"
AUDIO_BOS_TOKEN = "<|audio_bos|>"
AUDIO_EOS_TOKEN = "<|audio_eos|>"
EOT_TOKEN = "<|eot|>"
HOLD_TOKEN = "<|hold|>"
SPECIAL_TOKENS = [AUDIO_TOKEN, AUDIO_BOS_TOKEN, AUDIO_EOS_TOKEN, EOT_TOKEN, HOLD_TOKEN]

DEFAULT_SYSTEM = (
    "You are an end-of-turn detector. Given the user's speech audio, decide whether they "
    "have finished their turn (<|eot|>) or are only pausing mid-turn (<|hold|>)."
)


@dataclass
class SpecialIds:
    """Token ids of the chat/audio control tokens (resolved from the tokenizer)."""

    im_start: int
    im_end: int
    audio_bos: int
    audio_eos: int
    audio: int


def num_audio_tokens(mel_valid_frames: int) -> int:
    """Number of encoder frames (= ``<|AUDIO|>`` placeholders) for ``mel_valid_frames`` mel frames.

    Whisper's two conv layers downsample the mel spectrogram by ``stride=2`` once, so
    ``n_out = (n_mel - 1) // 2 + 1`` — identical to Qwen2-Audio's
    ``(feature_attention_mask.sum(-1) - 1) // 2 + 1``. The count computed here (from the
    feature-extractor mask) must equal the model's per-clip valid-frame count, or the
    placeholder↔embedding merge misaligns.
    """
    if mel_valid_frames <= 0:
        return 0
    return (mel_valid_frames - 1) // 2 + 1


def build_example(
    encode: Encode,
    ids: SpecialIds,
    *,
    marker_id: int,
    n_audio: int,
    transcript: str = "",
    system: str = DEFAULT_SYSTEM,
    user_instruction: str = "",
    supervise_transcript: bool = True,
) -> dict[str, list[int]]:
    """Build one tokenized example (``input_ids``, ``labels``, ``position_ids``).

    ``labels`` is :data:`IGNORE_INDEX` everywhere except the supervised assistant span
    (``marker`` + optional ``transcript`` + ``<|im_end|>``). ``position_ids`` restart at 0
    (this example is packed against others via cumulative sequence lengths, so each packed
    sub-sequence gets its own 0-based positions).
    """
    input_ids: list[int] = []
    labels: list[int] = []

    def add(tokens: list[int], supervised: bool) -> None:
        input_ids.extend(tokens)
        labels.extend(tokens if supervised else [IGNORE_INDEX] * len(tokens))

    # --- system turn (prompt) ---
    add([ids.im_start], False)
    add(encode("system\n"), False)
    add(encode(system), False)
    add([ids.im_end], False)
    add(encode("\n"), False)

    # --- user turn: audio placeholders (+ optional instruction) (prompt) ---
    add([ids.im_start], False)
    add(encode("user\n"), False)
    add([ids.audio_bos], False)
    add([ids.audio] * n_audio, False)
    add([ids.audio_eos], False)
    if user_instruction:
        add(encode(user_instruction), False)
    add([ids.im_end], False)
    add(encode("\n"), False)

    # --- assistant turn: header is prompt, body is supervised ---
    add([ids.im_start], False)
    add(encode("assistant\n"), False)
    add([marker_id], True)  # marker FIRST — the single-probe EOT read
    if supervise_transcript and transcript:
        add(encode(" " + transcript), True)
    add([ids.im_end], True)
    add(encode("\n"), False)

    return {
        "input_ids": input_ids,
        "labels": labels,
        "position_ids": list(range(len(input_ids))),
    }
