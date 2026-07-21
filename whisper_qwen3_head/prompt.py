"""Prompt construction for the Whisper→Qwen3 + p(eot) head model (Option C).

Pure Python (stdlib only, no torch/transformers) so ``import whisper_qwen3_head.prompt``
stays offline, mirroring :mod:`semantic_vad.training.prompt`.

Unlike :mod:`semantic_vad.training.prompt` (Option B), there is no chat scaffolding and no
``<|eot|>``/``<|hold|>`` marker token: this model reads ``p(eot)`` off a dedicated
classification head on the pooled last hidden state (Option A's head, see
``semvad/modeling.py::EoTHead``), not a next-token logprob. The prompt is exactly Option A's
``AUDIO_PROMPT_PREFIX`` (audio placeholders + a fixed instruction), just built as token ids
directly instead of a formatted string, since the placeholder count (``n_audio``) varies with
clip length.

Sequence layout (input-only, nothing is generated/supervised as text)::

    <|audio_bos|>{<|AUDIO|> * n_audio}<|audio_eos|>{instruction}

The pooled hidden state at the last token of this sequence feeds the head.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

# Re-exported: identical formula to Qwen2-Audio's audio-token count (Whisper's stride-2 conv
# halves the mel frame count), already implemented and unit-tested for Option B.
from semantic_vad.training.prompt import num_audio_tokens

#: A tokenizer's ``encode(text) -> list[int]`` with special tokens **disabled**.
Encode = Callable[[str], "list[int]"]

AUDIO_TOKEN = "<|AUDIO|>"
AUDIO_BOS_TOKEN = "<|audio_bos|>"
AUDIO_EOS_TOKEN = "<|audio_eos|>"
SPECIAL_TOKENS = [AUDIO_TOKEN, AUDIO_BOS_TOKEN, AUDIO_EOS_TOKEN]

EOT_INSTRUCTION = "Has the speaker finished their turn? Answer yes or no."

__all__ = [
    "num_audio_tokens",
    "Encode",
    "AUDIO_TOKEN",
    "AUDIO_BOS_TOKEN",
    "AUDIO_EOS_TOKEN",
    "SPECIAL_TOKENS",
    "EOT_INSTRUCTION",
    "SpecialIds",
    "build_prompt_ids",
]


@dataclass
class SpecialIds:
    """Token ids of the audio control tokens (resolved from the tokenizer after
    ``tokenizer.add_tokens(SPECIAL_TOKENS)``)."""

    audio_bos: int
    audio_eos: int
    audio: int


def build_prompt_ids(
    encode: Encode,
    ids: SpecialIds,
    *,
    n_audio: int,
    instruction: str = EOT_INSTRUCTION,
) -> list[int]:
    """Build one input-only token sequence: audio placeholders + instruction.

    Built as explicit ids (not string concatenation + a single ``encode()`` call) because
    ``n_audio`` varies per clip and special tokens shouldn't be re-tokenized through BPE.
    """
    input_ids: list[int] = [ids.audio_bos]
    input_ids.extend([ids.audio] * n_audio)
    input_ids.append(ids.audio_eos)
    input_ids.extend(encode(instruction))
    return input_ids
