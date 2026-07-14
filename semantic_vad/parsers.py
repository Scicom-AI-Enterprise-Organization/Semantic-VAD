"""Parsers that turn source-specific transcripts into a normalized ``list[Word]``."""

from __future__ import annotations

import re
from typing import Any, Iterable

from .schema import Word

# Matches a whisper timestamp token like ``<|0.04|>`` or ``<|12|>``.
_TS = re.compile(r"<\|(\d+(?:\.\d+)?)\|>")


def parse_whisper_timestamps(text: str) -> list[Word]:
    """Parse a whisper-style timestamped string into words.

    Handles the Malaysian-STT ``word`` level, e.g.::

        "<|0.04|> Collab<|0.22|><|0.26|> dia<|0.30|>"

    Each ``<|start|> text <|end|>`` triple becomes one :class:`Word`. Chunks with no
    text between two timestamps (or a dangling final timestamp) are skipped. This also
    works on ``segment`` level, where each chunk is a whole phrase rather than a word.
    """
    if not text:
        return []

    # re.split with one capturing group yields: [text, ts, text, ts, text, ...]
    parts = _TS.split(text)
    words: list[Word] = []
    # timestamps live at odd indices; the text following ts_k sits at index (i+1).
    i = 1
    while i + 1 < len(parts):
        try:
            start = float(parts[i])
        except ValueError:
            i += 2
            continue
        chunk = (parts[i + 1] or "").strip()
        # The end timestamp is the next timestamp token, at index i+2.
        if i + 2 >= len(parts):
            break
        try:
            end = float(parts[i + 2])
        except ValueError:
            i += 2
            continue
        if chunk:
            words.append(Word(word=chunk, start=start, end=max(end, start)))
        i += 2
    return words


def normalize_words(raw: Iterable[Any]) -> list[Word]:
    """Normalize a ``words`` list of dicts into :class:`Word` objects.

    Accepts the AAdonis/multilingual_audio_alignments and eot-bench shape
    ``{"word": str, "start": float, "end": float}``, and tolerates ``text``/``token``
    key aliases. Non-finite or out-of-order times are clamped. Result is sorted by start.
    """
    out: list[Word] = []
    for item in raw:
        if item is None:
            continue
        token = item.get("word", item.get("text", item.get("token")))
        start = item.get("start")
        end = item.get("end")
        if token is None or start is None or end is None:
            continue
        try:
            start = float(start)
            end = float(end)
        except (TypeError, ValueError):
            continue
        token = str(token).strip()
        if not token:
            continue
        out.append(Word(word=token, start=start, end=max(end, start)))
    out.sort(key=lambda w: (w.start, w.end))
    return out
