"""Sequence-packing math for flash-attention *varlen* training.

Multiple variable-length examples are concatenated into one long sequence (batch size 1)
and attention is confined to each sub-sequence via **cumulative sequence lengths**, exactly
like the reference ``qwen3_adamw.py``. This is the flash-attn varlen path
(``cu_seq_lens_q``/``cu_seq_lens_k`` + ``max_length_q``/``max_length_k``), **not** an SDPA
block-diagonal 4D mask (``block_diagonal_concat_inverted``) — no O(L²) mask is materialized.

Pure Python so it's offline-testable; the torch tensor wrapping is a thin layer in
:mod:`semantic_vad.training.data`.
"""

from __future__ import annotations


def build_cu_seqlens(lengths: list[int]) -> tuple[list[int], int]:
    """Return ``(cu_seqlens, max_len)`` for a list of packed sub-sequence lengths.

    ``cu_seqlens`` is the exclusive-prefix cumulative sum with a leading 0 (so it has
    ``len(lengths) + 1`` entries and ends at the total token count) — the boundary array
    flash-attention's varlen kernel expects for both queries and keys. ``max_len`` is the
    longest sub-sequence (the kernel's ``max_length_q``/``max_length_k``).
    """
    cu = [0]
    for n in lengths:
        cu.append(cu[-1] + n)
    return cu, (max(lengths) if lengths else 0)
