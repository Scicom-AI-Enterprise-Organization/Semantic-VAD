"""Benchmark p(eot) classification accuracy/F1/AUC on `Scicom-intl/semantic-vad-eot`
for either the Whisper->Qwen3 + `p(eot)` head checkpoint (Option C,
`whisper_qwen3_head/modeling.py::WhisperQwen3EoTClassifier`) or LiveKit's `turn-detector-v1`
cloud model. Same methodology as `scripts/benchmark_eot.py` (Option A / `semvad`) so a
`--backend local` run here and a `scripts/benchmark_eot.py --backend local` run score
directly comparable numbers -- both are scored via `semvad.metrics.
compute_classification_metrics` over the same causal per-silence-span cut points training
itself uses (`semvad.data.iter_causal_examples`, offsets=`TRUNCATION_OFFSETS` by default).

`test`/`validation` splits run ~50k turns per language; pass --limit (with a
fixed --seed) to draw a small, reproducible sample instead of scoring
everything -- the same (--dataset-name, --split, --limit, --seed) always
yields the same rows, streamed via the same shuffle+take convention as
scripts/duration_bias_check.py, so different backends/checkpoints are compared fairly.

Usage:
  # a from-scratch (untrained) model, just to sanity-check the harness
  python scripts/benchmark_eot_qwen3_head.py --backend local \\
      --dataset-name en --split test --limit 500 --seed 42 --output qwen3_head_en.json

  # a trained checkpoint (whisper_qwen3_head.train's --output_dir)
  python scripts/benchmark_eot_qwen3_head.py --backend local \\
      --checkpoint runs/eot-whisper-qwen3-head \\
      --dataset-name en --split test --limit 500 --seed 42 --output qwen3_head_en.json

  # LiveKit's cloud turn-detector-v1, same sample for a fair comparison
  LIVEKIT_API_KEY=... LIVEKIT_API_SECRET=... \\
  python scripts/benchmark_eot_qwen3_head.py --backend livekit \\
      --dataset-name en --split test --limit 500 --seed 42 --output livekit_en.json
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import os
import random
import time
from collections import defaultdict
from typing import Any, Optional

import numpy as np

from semvad.data import TRUNCATION_OFFSETS, iter_causal_examples
from semvad.metrics import compute_classification_metrics

LIVEKIT_SAMPLE_RATE = 16000
DEFAULT_LIVEKIT_MODEL = "turn-detector-v1"
_TRANSIENT_ERROR_CODES = (408, 429, 500, 502, 503, 504)


# --------------------------------------------------------------------------
# dataset sampling
# --------------------------------------------------------------------------


def load_rows(dataset_path: str, dataset_name: Optional[str], split: str, limit: int, seed: int) -> list:
    """Stream `limit` rows via shuffle+take -- same convention as
    `scripts/duration_bias_check.py` -- instead of downloading a whole
    ~50k-turn split just to sample a few hundred rows. The same
    (dataset_name, split, limit, seed) always yields the same rows, so
    different `--backend` runs are directly comparable."""
    from datasets import load_dataset

    ds = load_dataset(dataset_path, name=dataset_name, split=split, streaming=True)
    if limit and limit > 0:
        ds = ds.shuffle(buffer_size=limit, seed=seed).take(limit)
    else:
        print("[benchmark] WARNING: no --limit given, scoring the ENTIRE split -- this may be slow/expensive")
    return list(ds)


# --------------------------------------------------------------------------
# local (WhisperQwen3EoTClassifier) backend
# --------------------------------------------------------------------------


def _build_fresh_model(args, tokenizer, dtype):
    from transformers import WhisperFeatureExtractor

    from whisper_qwen3_head.modeling import WhisperQwen3EoTClassifier
    from whisper_qwen3_head.prompt import AUDIO_BOS_TOKEN, AUDIO_EOS_TOKEN, AUDIO_TOKEN, SPECIAL_TOKENS, SpecialIds

    print("[benchmark] WARNING: no --checkpoint given, scoring with an untrained backbone + head")
    tokenizer.add_tokens(SPECIAL_TOKENS)
    special_ids = SpecialIds(
        audio_bos=tokenizer.convert_tokens_to_ids(AUDIO_BOS_TOKEN),
        audio_eos=tokenizer.convert_tokens_to_ids(AUDIO_EOS_TOKEN),
        audio=tokenizer.convert_tokens_to_ids(AUDIO_TOKEN),
    )
    model = WhisperQwen3EoTClassifier.from_pretrained(
        args.qwen3_name, args.whisper_name, dtype=dtype, attn_implementation=args.attn_implementation
    )
    model.backbone.resize_token_embeddings(len(tokenizer), mean_resizing=False, pad_to_multiple_of=8)
    model.set_special_ids(special_ids)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(args.whisper_name)
    return model, feature_extractor, special_ids


def _load_checkpoint_model(args, tokenizer, dtype):
    from transformers import WhisperFeatureExtractor

    from whisper_qwen3_head.modeling import WhisperQwen3EoTClassifier
    from whisper_qwen3_head.prompt import AUDIO_BOS_TOKEN, AUDIO_EOS_TOKEN, AUDIO_TOKEN, SpecialIds

    model = WhisperQwen3EoTClassifier.from_checkpoint(
        args.checkpoint, dtype=dtype, attn_implementation=args.attn_implementation
    )
    special_ids = SpecialIds(
        audio_bos=tokenizer.convert_tokens_to_ids(AUDIO_BOS_TOKEN),
        audio_eos=tokenizer.convert_tokens_to_ids(AUDIO_EOS_TOKEN),
        audio=tokenizer.convert_tokens_to_ids(AUDIO_TOKEN),
    )
    model.set_special_ids(special_ids)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(args.checkpoint)
    return model, feature_extractor, special_ids


def run_local_backend(rows: list, args, print_every: int) -> list:
    import torch
    from transformers import AutoTokenizer

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]

    tokenizer_source = args.checkpoint or args.qwen3_name
    print(f"[benchmark] loading tokenizer from {tokenizer_source} ...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)

    if args.checkpoint:
        print(f"[benchmark] loading checkpoint {args.checkpoint} ({args.dtype}) on {args.device} ...")
        model, feature_extractor, special_ids = _load_checkpoint_model(args, tokenizer, dtype)
    else:
        print(f"[benchmark] building fresh {args.qwen3_name} + {args.whisper_name} ({args.dtype}) on {args.device} ...")
        model, feature_extractor, special_ids = _build_fresh_model(args, tokenizer, dtype)
    model.to(args.device)
    model.eval()

    predictions = []
    for row in rows:
        exs = list(iter_causal_examples(row, offsets=tuple(args.offsets)))
        for ex in exs:
            p_eot = model.predict_p_eot(tokenizer, feature_extractor, special_ids, ex["audio"], ex["sampling_rate"])
            predictions.append({
                "id": row["id"],
                "language": ex["language"],
                "label": ex["label"],
                "p_eot": p_eot,
                "count_span": ex["count_span"],
                "word_count": ex["word_count"],
                "full_audio_seconds": ex["full_audio_seconds"],
                "audio_seconds": ex["audio_seconds"]
            })
        if exs and len(predictions) % print_every < len(exs):
            print(f"[benchmark] ... {len(predictions)} spans scored")
    return predictions


# --------------------------------------------------------------------------
# LiveKit turn-detector-v1 cloud backend
#
# Speaks the same agent-gateway EOT websocket protocol as
# `eot_harness.livekit_turn_detector_adapter.LiveKitTurnDetectorAdapter`
# (session_create -> input_audio chunks -> inference_start -> eot_prediction),
# reimplemented standalone here (only depends on `aiohttp` + `livekit-agents`,
# not the `eot-harness` package) so we can score our own causal cut points
# instead of that adapter's silence-span time grid -- see `_score_row_livekit_once`.
# --------------------------------------------------------------------------


def _import_aiohttp():
    try:
        import aiohttp
    except ImportError as exc:
        raise RuntimeError("--backend livekit requires the `aiohttp` package (pip install aiohttp).") from exc
    return aiohttp


def _import_livekit_auth():
    try:
        from livekit.agents.inference import _utils
    except ImportError as exc:
        raise RuntimeError(
            "--backend livekit requires `livekit-agents` for cloud auth helpers (pip install livekit-agents)."
        ) from exc
    return _utils


def _import_livekit_proto():
    try:
        from livekit.protocol.agent_pb import agent_inference
    except ImportError as exc:
        raise RuntimeError(
            "--backend livekit requires `livekit-agents` (livekit.protocol) for the EOT websocket protocol."
        ) from exc
    return agent_inference


def _now():
    from google.protobuf.timestamp_pb2 import Timestamp

    ts = Timestamp()
    ts.GetCurrentTime()
    return ts


class _EotServerError(RuntimeError):
    """An EOT websocket error, tagged with whether retrying the row may help."""

    def __init__(self, message: str, *, code: Optional[int] = None, transient: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.transient = transient


def _is_transient_error(exc: BaseException, aiohttp) -> bool:
    if isinstance(exc, _EotServerError):
        return exc.transient
    if isinstance(exc, (asyncio.TimeoutError, ConnectionError)):
        return True
    return isinstance(exc, aiohttp.ClientError)


def _retry_after_seconds(exc: BaseException) -> Optional[float]:
    """Honor a server-provided `Retry-After` (seconds) on 429s, e.g. from
    `aiohttp.WSServerHandshakeError` raised by a rate-limited `ws_connect`."""
    headers = getattr(exc, "headers", None)
    value = headers.get("Retry-After") if headers else None
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _to_pcm16_bytes(audio: np.ndarray, sr: int, target_sr: int = LIVEKIT_SAMPLE_RATE) -> bytes:
    audio = np.asarray(audio, dtype=np.float32)
    if sr != target_sr:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()


@dataclasses.dataclass
class LiveKitContext:
    aiohttp: Any
    proto: Any
    ws_url: str
    headers: dict
    chunk_size: int
    timeout: float
    max_retries: int
    retry_backoff: float


async def _send(ws, msg) -> None:
    await ws.send_bytes(msg.SerializeToString())


async def _send_audio(ws, proto, pcm_bytes: bytes, start_byte: int, end_byte: int, chunk_size: int) -> None:
    for offset in range(start_byte, end_byte, chunk_size):
        segment = pcm_bytes[offset : min(offset + chunk_size, end_byte)]
        if not segment:
            continue
        await _send(
            ws,
            proto.ClientMessage(
                input_audio=proto.InputAudio(audio=segment, num_samples=len(segment) // 2, created_at=_now())
            ),
        )


async def _await_prediction(ws, aiohttp, proto, request_id: str, *, row_id: Any) -> float:
    while True:
        msg = await ws.receive()
        if msg.type in (
            aiohttp.WSMsgType.CLOSED,
            aiohttp.WSMsgType.CLOSE,
            aiohttp.WSMsgType.CLOSING,
            aiohttp.WSMsgType.ERROR,
        ):
            raise _EotServerError(f"eot websocket closed unexpectedly for row {row_id!r}: {msg.type}", transient=True)
        if msg.type != aiohttp.WSMsgType.BINARY:
            continue
        server_msg = proto.ServerMessage()
        server_msg.ParseFromString(msg.data)
        which = server_msg.WhichOneof("message")
        if which == "error":
            code = server_msg.error.code
            raise _EotServerError(
                f"eot server error for row {row_id!r}: {server_msg.error.message} (code={code})",
                code=code,
                transient=code in _TRANSIENT_ERROR_CODES,
            )
        if which == "eot_prediction" and server_msg.request_id == request_id:
            return server_msg.eot_prediction.probability


async def _score_row_livekit_once(row: dict, exs: list, ctx: LiveKitContext) -> list:
    """One websocket session per row: streams the turn's audio once, issuing one
    `inference_start` per causal cut point -- the same points `iter_causal_examples`
    built for the local backend, so both backends are scored against identical
    (label, cut-time) pairs."""
    native_sr = row["audio"]["sampling_rate"]
    pcm_bytes = _to_pcm16_bytes(row["audio"]["array"], native_sr)

    async with ctx.aiohttp.ClientSession() as session:
        async with session.ws_connect(ctx.ws_url, headers=ctx.headers, max_msg_size=0) as ws:
            await _send(
                ws,
                ctx.proto.ClientMessage(
                    session_create=ctx.proto.SessionCreate(
                        settings=ctx.proto.SessionSettings(
                            sample_rate=LIVEKIT_SAMPLE_RATE, encoding=ctx.proto.AUDIO_ENCODING_PCM_S16LE
                        )
                    )
                ),
            )
            probs = []
            sent_bytes = 0
            for idx, ex in enumerate(exs):
                cut_time = len(ex["audio"]) / native_sr  # same cut point the local backend scored
                target_bytes = min(len(pcm_bytes), int(round(cut_time * LIVEKIT_SAMPLE_RATE)) * 2)
                if target_bytes > sent_bytes:
                    await _send_audio(ws, ctx.proto, pcm_bytes, sent_bytes, target_bytes, ctx.chunk_size)
                    sent_bytes = target_bytes
                request_id = f"{row['id']}-{idx}"
                await _send(
                    ws, ctx.proto.ClientMessage(inference_start=ctx.proto.InferenceStart(request_id=request_id))
                )
                prob = await asyncio.wait_for(
                    _await_prediction(ws, ctx.aiohttp, ctx.proto, request_id, row_id=row["id"]), timeout=ctx.timeout
                )
                probs.append(prob)
            await _send(ws, ctx.proto.ClientMessage(session_close=ctx.proto.SessionClose()))
    return probs


async def _score_row_livekit(row: dict, exs: list, ctx: LiveKitContext) -> list:
    for attempt in range(ctx.max_retries + 1):
        try:
            return await _score_row_livekit_once(row, exs, ctx)
        except Exception as exc:
            if attempt >= ctx.max_retries or not _is_transient_error(exc, ctx.aiohttp):
                raise
            delay = _retry_after_seconds(exc)
            if delay is None:
                # exponential backoff + jitter, not linear -- a sustained 429 needs
                # more than a couple seconds of cooldown to clear.
                delay = ctx.retry_backoff * (2**attempt) + random.uniform(0, ctx.retry_backoff)
            print(
                f"[benchmark] transient error for row {row['id']!r} "
                f"(attempt {attempt + 1}/{ctx.max_retries}): {exc}; retrying in {delay:.1f}s"
            )
            await asyncio.sleep(delay)
    raise AssertionError("unreachable")


async def _run_livekit_backend_async(rows: list, args, print_every: int) -> list:
    aiohttp = _import_aiohttp()
    proto = _import_livekit_proto()
    auth = _import_livekit_auth()

    api_key = args.livekit_api_key or os.environ.get("LIVEKIT_API_KEY")
    api_secret = args.livekit_api_secret or os.environ.get("LIVEKIT_API_SECRET")
    if not api_key or not api_secret:
        raise RuntimeError(
            "--backend livekit requires LIVEKIT_API_KEY and LIVEKIT_API_SECRET "
            "(env or --livekit-api-key/--livekit-api-secret)."
        )

    ws_url = args.livekit_base_url or auth.get_default_inference_url()
    if ws_url.startswith(("http://", "https://")):
        ws_url = ws_url.replace("http", "ws", 1)
    ws_url = f"{ws_url}/eot"
    headers = {
        **auth.get_inference_headers(),
        "Authorization": f"Bearer {auth.create_access_token(api_key, api_secret)}",
    }

    chunk_size = int(LIVEKIT_SAMPLE_RATE * args.livekit_chunk_ms / 1000) * 2
    ctx = LiveKitContext(
        aiohttp=aiohttp,
        proto=proto,
        ws_url=ws_url,
        headers=headers,
        chunk_size=chunk_size,
        timeout=args.livekit_timeout,
        max_retries=args.livekit_max_retries,
        retry_backoff=args.livekit_retry_backoff,
    )

    predictions: list = []
    failed_row_ids: list = []
    lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(args.livekit_concurrency)
    # a row that exhausted all retries means the transient-error path (429/5xx/timeout)
    # never cleared -- almost certainly an account-level quota, not a burst limit. Once
    # that happens, stop attempting new rows outright: hammering the gateway with more
    # rows would just repeat the same multi-minute retry-and-fail for every remaining row.
    quota_exhausted = {"flag": False}

    async def handle_row(row: dict) -> None:
        if quota_exhausted["flag"]:
            failed_row_ids.append(row["id"])
            return
        exs = list(iter_causal_examples(row, offsets=tuple(args.offsets)))
        if not exs:
            return
        async with semaphore:
            if quota_exhausted["flag"]:
                failed_row_ids.append(row["id"])
                return
            try:
                probs = await _score_row_livekit(row, exs, ctx)
            except Exception as exc:
                async with lock:
                    failed_row_ids.append(row["id"])
                    if not quota_exhausted["flag"]:
                        quota_exhausted["flag"] = True
                        print(
                            f"[benchmark] giving up on row {row['id']!r} after exhausting retries ({exc}); "
                            "treating this as quota exhaustion and skipping all remaining rows "
                            "(check the LiveKit Cloud dashboard for your turn-detector-v1 quota)"
                        )
                return
        async with lock:
            for ex, p_eot in zip(exs, probs):
                predictions.append(
                    {"id": row["id"], "language": ex["language"], "label": ex["label"], "p_eot": p_eot}
                )
            if len(predictions) % print_every < len(exs):
                print(f"[benchmark] ... {len(predictions)} spans scored")

    await asyncio.gather(*(handle_row(row) for row in rows))
    if failed_row_ids:
        print(
            f"[benchmark] WARNING: {len(failed_row_ids)}/{len(rows)} row(s) skipped after retry exhaustion "
            f"-- metrics below only cover the {len(predictions)} spans that succeeded"
        )
    return predictions


def run_livekit_backend(rows: list, args, print_every: int) -> list:
    return asyncio.run(_run_livekit_backend_async(rows, args, print_every))


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------


def report(predictions: list) -> dict:
    labels = [p["label"] for p in predictions]
    probs = [p["p_eot"] for p in predictions]
    overall = compute_classification_metrics(probs, labels)
    overall["n"] = len(predictions)

    grouped = defaultdict(list)
    for p in predictions:
        grouped[p["language"]].append(p)
    by_language = {}
    for language, preds in sorted(grouped.items()):
        m = compute_classification_metrics([p["p_eot"] for p in preds], [p["label"] for p in preds])
        m["n"] = len(preds)
        by_language[language] = m

    header = f"{'language':>10}  {'n':>6}  {'accuracy':>9}  {'f1':>7}  {'auc':>7}"
    print()
    print(header)
    print("-" * len(header))
    for language, m in sorted(by_language.items()):
        print(
            f"{language:>10}  {m['n']:>6}  {m['accuracy']:>9.3f}  "
            f"{m.get('f1', float('nan')):>7.3f}  {m.get('auc', float('nan')):>7.3f}"
        )
    print("-" * len(header))
    print(
        f"{'overall':>10}  {overall['n']:>6}  {overall['accuracy']:>9.3f}  "
        f"{overall.get('f1', float('nan')):>7.3f}  {overall.get('auc', float('nan')):>7.3f}"
    )
    return {"overall": overall, "by_language": by_language}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", required=True, choices=["local", "livekit"])
    parser.add_argument("--dataset-path", default="Scicom-intl/semantic-vad-eot")
    parser.add_argument("--dataset-name", default="en", help="HF dataset config, e.g. en/de/es/... or 'all'")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=512, help="rows (turns) to sample; <= 0 scores the entire split")
    parser.add_argument("--seed", type=int, default=42, help="must match across --backend runs for a fair comparison")
    parser.add_argument(
        "--offsets",
        type=float,
        nargs="+",
        default=list(TRUNCATION_OFFSETS),
        help="seconds into each silence span to cut at, same as training (semvad.data.TRUNCATION_OFFSETS)",
    )
    parser.add_argument("--output", default=None, help="optional path to dump per-span predictions + metrics as JSON")
    parser.add_argument("--print-every", type=int, default=100)

    local = parser.add_argument_group("local backend")
    local.add_argument("--qwen3-name", default="Qwen/Qwen3-0.6B", help="ignored if --checkpoint is given")
    local.add_argument("--whisper-name", default="openai/whisper-base", help="ignored if --checkpoint is given")
    local.add_argument(
        "--checkpoint", default=None,
        help="dir/repo written by whisper_qwen3_head.train (WhisperQwen3EoTClassifier.save_pretrained)",
    )
    local.add_argument("--device", default=None)
    local.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    local.add_argument("--attn_implementation", default="sdpa")

    livekit = parser.add_argument_group("livekit backend")
    livekit.add_argument("--livekit-model", default=DEFAULT_LIVEKIT_MODEL, help="label only, not sent over the wire")
    livekit.add_argument(
        "--livekit-base-url", default=None, help="defaults to $LIVEKIT_INFERENCE_URL / the production gateway"
    )
    livekit.add_argument("--livekit-api-key", default=None, help="defaults to $LIVEKIT_API_KEY")
    livekit.add_argument("--livekit-api-secret", default=None, help="defaults to $LIVEKIT_API_SECRET")
    livekit.add_argument("--livekit-chunk-ms", type=int, default=100)
    livekit.add_argument(
        "--livekit-concurrency",
        type=int,
        default=4,
        help="rows scored concurrently over separate ws sessions; lower this if you see 429s "
        "(rate limit) on the eot websocket handshake",
    )
    livekit.add_argument("--livekit-timeout", type=float, default=30.0, help="seconds to wait for one inference reply")
    livekit.add_argument(
        "--livekit-max-retries",
        type=int,
        default=6,
        help="whole-row reconnect retries on transient errors (429/5xx/timeouts), exponential backoff",
    )
    livekit.add_argument(
        "--livekit-retry-backoff", type=float, default=1.0, help="base seconds for exponential backoff (2**attempt)"
    )
    args = parser.parse_args()

    if args.backend == "local" and args.device is None:
        import torch

        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    print(
        f"[benchmark] streaming {args.dataset_path}/{args.dataset_name} [{args.split}], "
        f"limit={args.limit}, seed={args.seed} ..."
    )
    rows = load_rows(args.dataset_path, args.dataset_name, args.split, args.limit, args.seed)
    print(f"[benchmark] sampled {len(rows)} turns")

    start = time.perf_counter()
    if args.backend == "local":
        predictions = run_local_backend(rows, args, args.print_every)
    else:
        predictions = run_livekit_backend(rows, args, args.print_every)
    elapsed = time.perf_counter() - start
    print(f"[benchmark] scored {len(predictions)} spans in {elapsed:.1f}s")

    result = report(predictions)
    if args.output:
        with open(args.output, "w") as f:
            json.dump({"args": vars(args), "predictions": predictions, "metrics": result}, f, indent=2)
        print(f"[benchmark] wrote {args.output}")


if __name__ == "__main__":
    main()
