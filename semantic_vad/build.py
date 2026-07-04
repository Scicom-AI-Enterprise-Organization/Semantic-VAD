"""CLI: stream a forced-alignment corpus into an eot-bench-compatible parquet dataset.

Examples
--------
Build 200 English rows from the multilingual corpus::

    python -m semantic_vad.build --source multilingual --config english \\
        --limit 200 --out data/en.parquet

Build Malaysian rows from the natural streaming segments::

    python -m semantic_vad.build --source malaysian --config malaysian \\
        --malaysian-mode streaming --limit 200 --out data/ms.parquet
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Iterator

from .audio import AUDIO_EXT, encode_audio, resample_linear, turn_to_row
from .schema import EOTRow, TurnConfig
from .sources import SOURCES
from .turns import build_turns


def build_rows(
    source: str,
    config: str,
    cfg: TurnConfig,
    *,
    mode: str = "auto",
    split: str = "train",
    limit: int | None = None,
    target_sr: int = 16000,
    streaming: bool = True,
    malaysian_mode: str = "streaming",
    malaysian_zips: list[str] | None = None,
    malaysian_max_scan: int | None = None,
    malaysian_backend: str = "download",
    malaysian_n_zips: int = 4,
    malaysian_shard: tuple[int, int] | None = None,
    hf_token: str | None = None,
) -> Iterator[EOTRow]:
    """Yield finished :class:`EOTRow` objects from a source corpus.

    ``mode`` is ``"auto"`` (use each recording's suggested mode), or a forced
    ``"single"``/``"segment"`` that overrides ``cfg.mode`` for every recording.
    """
    if source not in SOURCES:
        raise ValueError(f"unknown source {source!r}; choose from {list(SOURCES)}")

    kwargs = dict(split=split, limit=limit, streaming=streaming)
    if source == "malaysian":
        kwargs["source_mode"] = malaysian_mode
        kwargs["zip_names"] = malaysian_zips
        kwargs["max_scan"] = malaysian_max_scan
        kwargs["token"] = hf_token
        kwargs["backend"] = malaysian_backend
        kwargs["n_zips"] = malaysian_n_zips
        if malaysian_shard is not None:
            kwargs["shard_index"], kwargs["shard_count"] = malaysian_shard
    recordings = SOURCES[source](config, **kwargs)

    def with_mode(m: str) -> TurnConfig:
        return TurnConfig(**{**cfg.__dict__, "mode": m})

    for rec in recordings:
        run_cfg = with_mode(rec.suggested_mode if mode == "auto" else mode)

        array = rec.audio
        sr = rec.sampling_rate
        if sr != target_sr:
            array = resample_linear(array, sr, target_sr)
            sr = target_sr

        duration = len(array) / sr
        turns = build_turns(rec.words, duration, run_cfg)
        for ti, turn in enumerate(turns):
            row_id = f"{rec.language}__{rec.source_id}__turn_{ti:03d}"
            yield turn_to_row(turn, array, sr, row_id=row_id, language=rec.language)


_VAL_STR = {"dtype": "string", "_type": "Value"}
_VAL_F64 = {"dtype": "float64", "_type": "Value"}


def _hf_features(sampling_rate: int) -> dict:
    """HuggingFace `Features` dict (the `_type` encoding datasets stores in parquet metadata).

    Matches `livekit/eot-bench-data`: `audio` is an `Audio` feature; `silence_spans`,
    `words`, `messages` are lists of structs (encoded as a single-element `[ {...} ]`).
    """
    return {
        "id": _VAL_STR,
        "audio": {"sampling_rate": sampling_rate, "_type": "Audio"},
        "language": _VAL_STR,
        "duration": _VAL_F64,
        "silence_spans": [{"start": _VAL_F64, "end": _VAL_F64}],
        "words": [{"word": _VAL_STR, "start": _VAL_F64, "end": _VAL_F64}],
        "messages": [{"role": _VAL_STR, "content": _VAL_STR}],
    }


def _arrow_schema(sampling_rate: int):
    import json

    import pyarrow as pa

    span_t = pa.struct([("start", pa.float64()), ("end", pa.float64())])
    word_t = pa.struct([("word", pa.string()), ("start", pa.float64()), ("end", pa.float64())])
    msg_t = pa.struct([("role", pa.string()), ("content", pa.string())])
    audio_t = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
    schema = pa.schema(
        [
            ("id", pa.string()),
            ("audio", audio_t),
            ("language", pa.string()),
            ("duration", pa.float64()),
            ("silence_spans", pa.list_(span_t)),
            ("words", pa.list_(word_t)),
            ("messages", pa.list_(msg_t)),
        ]
    )
    # Embed HF feature metadata so load_dataset recognizes `audio` as an Audio feature.
    meta = {"huggingface": json.dumps({"info": {"features": _hf_features(sampling_rate)}})}
    return schema.with_metadata(meta)


def write_parquet(
    rows: Iterator[EOTRow],
    out_path: str,
    sampling_rate: int = 16000,
    batch_size: int = 256,
    audio_format: str = "wav",
) -> int:
    """Write an eot-bench-compatible parquet directly with pyarrow. Returns row count.

    Audio is stored as pre-encoded WAV bytes (``{"bytes", "path"}``). We bypass
    ``datasets``' Audio encoder because in datasets>=5 it imports torch/torchcodec
    unconditionally; eot-bench itself just reads these bytes with soundfile. Rows are
    flushed in batches so memory stays bounded on large builds.
    """
    import os

    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = _arrow_schema(sampling_rate)
    ext = AUDIO_EXT.get(audio_format, audio_format)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    def empty() -> dict[str, list]:
        return {k: [] for k in
                ["id", "audio", "language", "duration", "silence_spans", "words", "messages"]}

    writer = None
    buf = empty()
    n = 0

    def flush():
        nonlocal writer
        if not buf["id"]:
            return
        table = pa.Table.from_pydict(buf, schema=schema)
        if writer is None:
            writer = pq.ParquetWriter(out_path, schema)
        writer.write_table(table)

    try:
        for r in rows:
            buf["id"].append(r.id)
            buf["audio"].append({"bytes": encode_audio(r.audio, r.sampling_rate, audio_format),
                                 "path": f"{r.id}.{ext}"})
            buf["language"].append(r.language)
            buf["duration"].append(float(round(r.duration, 3)))
            buf["silence_spans"].append(r.silence_spans)
            buf["words"].append(r.words)
            buf["messages"].append(r.messages)
            n += 1
            if len(buf["id"]) >= batch_size:
                flush()
                buf = empty()
        flush()
    finally:
        if writer is not None:
            writer.close()

    if n == 0:
        raise SystemExit("no rows produced -- check the source/config/filters")
    return n


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Build an eot-bench-compatible EOT dataset.")
    p.add_argument("--source", required=True, choices=list(SOURCES))
    p.add_argument("--config", required=True, help="language/subset config of the source")
    p.add_argument("--split", default="train")
    p.add_argument("--out", required=True, help="output .parquet path")
    p.add_argument("--limit", type=int, default=None, help="max source recordings to read")
    p.add_argument("--target-sr", type=int, default=16000)
    p.add_argument("--audio-format", default="wav", choices=["wav", "flac", "mp3", "ogg"])
    p.add_argument("--no-streaming", action="store_true", help="download the full split first")
    p.add_argument("--malaysian-mode", default="streaming", choices=["streaming", "whole"])
    p.add_argument("--malaysian-zips", default=None,
                   help="comma-separated zip archive names to read Malaysian audio from "
                        "(default: discover all matching archives)")
    p.add_argument("--malaysian-max-scan", type=int, default=None,
                   help="max dataset rows to scan for Malaysian (bounds a small run)")

    # TurnConfig knobs
    p.add_argument("--mode", default="auto", choices=["auto", "single", "segment"],
                   help="'auto' uses each corpus's suggested mode")
    p.add_argument("--min-silence", type=float, default=0.1)
    p.add_argument("--turn-gap", type=float, default=0.7)
    p.add_argument("--eot-trailing", type=float, default=0.5)
    p.add_argument("--max-trailing", type=float, default=1.0)
    p.add_argument("--lead-in", type=float, default=0.3)
    p.add_argument("--min-words", type=int, default=1)
    p.add_argument("--min-hold-spans", type=int, default=0)
    args = p.parse_args(argv)

    # TurnConfig requires a concrete mode; the "auto" choice is handled by build_rows.
    cfg = TurnConfig(
        min_silence=args.min_silence,
        mode="single" if args.mode == "auto" else args.mode,
        turn_gap=args.turn_gap,
        eot_trailing=args.eot_trailing,
        max_trailing=args.max_trailing,
        lead_in=args.lead_in,
        min_words=args.min_words,
        min_hold_spans=args.min_hold_spans,
    )

    zips = args.malaysian_zips.split(",") if args.malaysian_zips else None
    rows = build_rows(
        args.source,
        args.config,
        cfg,
        mode=args.mode,
        split=args.split,
        limit=args.limit,
        target_sr=args.target_sr,
        streaming=not args.no_streaming,
        malaysian_mode=args.malaysian_mode,
        malaysian_zips=zips,
        malaysian_max_scan=args.malaysian_max_scan,
        hf_token=os.environ.get("HF_TOKEN"),
    )
    n = write_parquet(rows, args.out, sampling_rate=args.target_sr, audio_format=args.audio_format)
    print(f"wrote {n} rows -> {args.out}", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    # Hard-exit: streaming readers / libsndfile spawn native threads that can crash during
    # interpreter finalization (harmless, but returns non-zero). The parquet is already on
    # disk, so skip finalizers with a clean exit code.
    os._exit(0)


if __name__ == "__main__":
    main()
