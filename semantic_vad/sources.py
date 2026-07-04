"""Adapters that stream the two forced-alignment corpora into a common shape.

Each adapter yields :class:`SourceRecording` objects: a language, a normalized
``list[Word]``, and the decoded mono audio for that recording. The builder is agnostic
to which corpus a recording came from.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np

from .parsers import normalize_words, parse_whisper_timestamps
from .schema import Word

# AAdonis config name -> ISO code used by eot-bench.
MULTILINGUAL_LANG = {
    "english": "en", "french": "fr", "german": "de", "italian": "it",
    "japanese": "ja", "korean": "ko", "mandarin": "zh", "polish": "pl",
    "portuguese": "pt", "russian": "ru", "spanish": "es", "thai": "th",
    "turkish": "tr",
}

MULTILINGUAL_REPO = "AAdonis/multilingual_audio_alignments"
MALAYSIAN_REPO = "malaysia-ai/Malaysian-STT"


@dataclass
class SourceRecording:
    """One forced-aligned recording ready to be segmented into turns."""

    source_id: str
    language: str
    words: list[Word]
    audio: np.ndarray
    sampling_rate: int
    suggested_mode: str  # "single" or "segment"
    transcript: str = ""


def _decode_audio_field(audio) -> tuple[np.ndarray, int]:
    """Decode a HuggingFace ``Audio(decode=False)`` value to (mono-ish array, sr).

    Handles both the raw ``{"bytes": ..., "path": ...}`` shape (what ``decode=False``
    yields, and how eot-bench loads audio) and an already-decoded ``{"array", "sampling_rate"}``
    dict, so the adapter is stable across datasets versions (no torchcodec dependency).
    """
    import io

    import soundfile as sf

    if isinstance(audio, dict) and audio.get("array") is not None:
        return np.asarray(audio["array"]), int(audio["sampling_rate"])
    data = audio.get("bytes") if isinstance(audio, dict) else None
    if data:
        arr, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
        return np.asarray(arr), int(sr)
    path = audio.get("path") if isinstance(audio, dict) else None
    if path:
        arr, sr = sf.read(path, dtype="float32", always_2d=False)
        return np.asarray(arr), int(sr)
    raise ValueError("could not decode audio field")


def iter_multilingual(
    config: str,
    split: str = "train",
    *,
    limit: int | None = None,
    streaming: bool = True,
) -> Iterator[SourceRecording]:
    """Stream AAdonis/multilingual_audio_alignments (embedded audio, clean word list).

    Each row is a single sentence, so every recording is one clean turn (``single`` mode).
    """
    from datasets import Audio, load_dataset

    lang = MULTILINGUAL_LANG.get(config, config)
    ds = load_dataset(MULTILINGUAL_REPO, config, split=split, streaming=streaming)
    ds = ds.cast_column("audio", Audio(decode=False))  # raw bytes -> decode with soundfile
    for idx, row in enumerate(ds):
        if limit is not None and idx >= limit:
            break
        words = normalize_words(row.get("words") or [])
        if not words:
            continue
        try:
            array, sr = _decode_audio_field(row["audio"])
        except Exception as exc:  # noqa: BLE001 - skip undecodable rows, keep streaming
            print(f"[warn] skip {config}_{idx}: {exc}")
            continue
        yield SourceRecording(
            source_id=f"{config}_{idx}",
            language=lang,
            words=words,
            audio=array,
            sampling_rate=sr,
            suggested_mode="single",
            transcript=str(row.get("transcript", "")),
        )


def iter_malaysian(
    config: str = "malaysian",
    split: str = "train",
    *,
    source_mode: str = "streaming",
    limit: int | None = None,
    streaming: bool = True,
    zip_names: list[str] | None = None,
    token: str | None = None,
    max_scan: int | None = None,
    backend: str = "download",
    n_zips: int = 4,
    in_ram: bool = True,
    shard_index: int | None = None,
    shard_count: int | None = None,
) -> Iterator[SourceRecording]:
    """Stream malaysia-ai/Malaysian-STT (whisper-format text; audio inside zip archives).

    ``source_mode="streaming"`` yields one recording per natural segment (each already a
    turn -> ``single`` mode). ``source_mode="whole"`` yields the full recording -> ``segment``
    mode (split into pseudo-turns by gap). Only ``level == "word"`` rows are consumed.

    ``backend="download"`` (default) downloads whole zip archives (Xet-accelerated) and reads
    members locally -- far faster than the ``"remote"`` per-member HTTP range requests. Only
    segments present in the resident archives are emitted (the rest are skipped as the stream
    passes them). ``n_zips`` bounds how many archives to pull; ``shard_index``/``shard_count``
    partition the row stream across parallel worker processes (worker i takes rows where
    ``ridx % shard_count == shard_index``). ``max_scan`` caps rows read.
    """
    from datasets import load_dataset

    from .malaysian_audio import DownloadZipResolver, ZipAudioResolver, discover_zip_names

    prefix = f"{config}-{'segment' if source_mode == 'streaming' else 'whole'}"
    if zip_names is None:
        zip_names = discover_zip_names(prefix, token=token)[:n_zips]

    if backend == "download":
        resolver = DownloadZipResolver(zip_names, token=token, in_ram=in_ram)
    else:
        resolver = ZipAudioResolver(zip_names, token=token)
        for name in zip_names:
            resolver.index_zip(name)

    suggested = "single" if source_mode == "streaming" else "segment"
    ds = load_dataset(MALAYSIAN_REPO, config, split=split, streaming=streaming)
    emitted = 0
    for ridx, row in enumerate(ds):
        if max_scan is not None and ridx >= max_scan:
            break
        if shard_count and (ridx % shard_count) != shard_index:
            continue
        if row.get("mode") != source_mode or row.get("level") != "word":
            continue
        texts = row.get("texts") or []
        files = row.get("audio_filenames") or []
        for k, (text, fname) in enumerate(zip(texts, files)):
            if limit is not None and emitted >= limit:
                resolver.close()
                return
            words = parse_whisper_timestamps(text)
            if not words:
                continue
            try:
                array, sr = resolver.read_audio(fname)
            except KeyError:
                continue  # audio not in the indexed archive(s)
            except Exception as exc:  # noqa: BLE001 - skip unreadable audio, keep going
                print(f"[warn] skip {fname}: {exc}")
                continue
            emitted += 1
            yield SourceRecording(
                source_id=f"{config}_{ridx}_{k}",
                language="ms",
                words=words,
                audio=array,
                sampling_rate=sr,
                suggested_mode=suggested,
                transcript=" ".join(w.word for w in words),
            )
    resolver.close()


#: Registry so the CLI can pick an adapter by name.
SOURCES = {
    "multilingual": iter_multilingual,
    "malaysian": iter_malaysian,
}
