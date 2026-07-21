"""Inference latency benchmark across variable audio input lengths, for the Whisper-encoder
-> Qwen3 backbone + `p(eot)` classification head model (Option C,
`whisper_qwen3_head/modeling.py::WhisperQwen3EoTClassifier`) -- combines Option B's backbone
(`semantic_vad/training/modeling.py::WhisperQwen3`) with Option A's head (`semvad/modeling.py`
`::EoTHead`), full fine-tuned. Same methodology as `scripts/benchmark_latency.py` and
`scripts/benchmark_latency_qwen3.py` so all three JSON outputs are directly comparable.

Measures wall-clock latency of a single forward pass through
`WhisperQwen3EoTClassifier.predict_p_eot`: build the audio-placeholder + instruction prompt,
run the backbone once, pool the last hidden state, and read `p(eot) = sigmoid(head(pooled))`
-- no marker-token softmax (unlike Option B) and no autoregressive decoding.

Usage:
  python scripts/benchmark_latency_qwen3_head.py --device cuda
  python scripts/benchmark_latency_qwen3_head.py --device cuda --checkpoint runs/eot-whisper-qwen3-head \\
      --durations 0.5 1 2 4 8 16 24 30 --repeats 20 --output latency_qwen3_head.json

On the GPU training pod, pass `--attn_implementation flash_attention_2` to match the real
training kernel; the default `sdpa` runs anywhere (including a laptop CPU) for a quick check.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time

import numpy as np
import torch


@dataclasses.dataclass
class LatencyStats:
    duration_s: float
    n: int
    mean_ms: float
    median_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float
    std_ms: float

    def to_row(self) -> str:
        return (
            f"{self.duration_s:>8.2f}  {self.n:>4d}  {self.mean_ms:>9.1f}  "
            f"{self.median_ms:>9.1f}  {self.p95_ms:>9.1f}  {self.min_ms:>9.1f}  "
            f"{self.max_ms:>9.1f}  {self.std_ms:>8.1f}"
        )


def summarize(duration_s: float, samples_ms: list) -> LatencyStats:
    arr = np.array(samples_ms)
    return LatencyStats(
        duration_s=duration_s,
        n=len(arr),
        mean_ms=float(arr.mean()),
        median_ms=float(np.median(arr)),
        p95_ms=float(np.percentile(arr, 95)),
        min_ms=float(arr.min()),
        max_ms=float(arr.max()),
        std_ms=float(arr.std()),
    )


def make_audio(duration_s: float, sampling_rate: int, seed: int) -> np.ndarray:
    """Synthetic clip -- forward-pass latency depends on shape, not content."""
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(int(duration_s * sampling_rate)) * 0.05).astype(np.float32)


def bench_one(model, tokenizer, feature_extractor, special_ids, audio, sampling_rate, device, warmup, repeats) -> list:
    for _ in range(warmup):
        model.predict_p_eot(tokenizer, feature_extractor, special_ids, audio, sampling_rate)
    if device.startswith("cuda"):
        torch.cuda.synchronize()

    samples_ms = []
    for _ in range(repeats):
        start = time.perf_counter()
        model.predict_p_eot(tokenizer, feature_extractor, special_ids, audio, sampling_rate)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        samples_ms.append((time.perf_counter() - start) * 1000)
    return samples_ms


def build_fresh_model(qwen3_name, whisper_name, dtype, attn_implementation, tokenizer):
    from whisper_qwen3_head.modeling import WhisperQwen3EoTClassifier
    from whisper_qwen3_head.prompt import AUDIO_BOS_TOKEN, AUDIO_EOS_TOKEN, AUDIO_TOKEN, SPECIAL_TOKENS, SpecialIds

    tokenizer.add_tokens(SPECIAL_TOKENS)
    special_ids = SpecialIds(
        audio_bos=tokenizer.convert_tokens_to_ids(AUDIO_BOS_TOKEN),
        audio_eos=tokenizer.convert_tokens_to_ids(AUDIO_EOS_TOKEN),
        audio=tokenizer.convert_tokens_to_ids(AUDIO_TOKEN),
    )

    model = WhisperQwen3EoTClassifier.from_pretrained(
        qwen3_name, whisper_name, dtype=dtype, attn_implementation=attn_implementation
    )
    model.backbone.resize_token_embeddings(len(tokenizer), mean_resizing=False, pad_to_multiple_of=8)
    model.set_special_ids(special_ids)

    from transformers import WhisperFeatureExtractor

    feature_extractor = WhisperFeatureExtractor.from_pretrained(whisper_name)
    return model, feature_extractor, special_ids


def load_checkpoint_model(checkpoint, dtype, attn_implementation, tokenizer):
    from transformers import WhisperFeatureExtractor

    from whisper_qwen3_head.modeling import WhisperQwen3EoTClassifier
    from whisper_qwen3_head.prompt import AUDIO_BOS_TOKEN, AUDIO_EOS_TOKEN, AUDIO_TOKEN, SpecialIds

    model = WhisperQwen3EoTClassifier.from_checkpoint(checkpoint, dtype=dtype, attn_implementation=attn_implementation)
    special_ids = SpecialIds(
        audio_bos=tokenizer.convert_tokens_to_ids(AUDIO_BOS_TOKEN),
        audio_eos=tokenizer.convert_tokens_to_ids(AUDIO_EOS_TOKEN),
        audio=tokenizer.convert_tokens_to_ids(AUDIO_TOKEN),
    )
    model.set_special_ids(special_ids)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(checkpoint)
    return model, feature_extractor, special_ids


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--qwen3-name", default="Qwen/Qwen3-0.6B", help="ignored if --checkpoint is given")
    parser.add_argument("--whisper-name", default="openai/whisper-base", help="ignored if --checkpoint is given")
    parser.add_argument(
        "--checkpoint", default=None,
        help="dir written by whisper_qwen3_head.train (WhisperQwen3EoTClassifier.save_pretrained)",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument(
        "--durations", type=float, nargs="+", default=[0.5, 1, 2, 4, 8, 16, 24, 30],
        help="audio clip durations in seconds to benchmark",
    )
    parser.add_argument("--sampling-rate", type=int, default=16000)
    parser.add_argument("--warmup", type=int, default=3, help="untimed calls per duration, to settle caches/cudnn")
    parser.add_argument("--repeats", type=int, default=10, help="timed calls per duration")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default=None, help="optional path to dump results as JSON")
    parser.add_argument("--attn_implementation", default="sdpa")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]

    tokenizer_source = args.checkpoint or args.qwen3_name
    print(f"[bench] loading tokenizer from {tokenizer_source} ...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)

    if args.checkpoint:
        print(f"[bench] loading checkpoint {args.checkpoint} ({args.dtype}) on {args.device} ...")
        model, feature_extractor, special_ids = load_checkpoint_model(
            args.checkpoint, dtype, args.attn_implementation, tokenizer
        )
    else:
        print(
            f"[bench] building fresh {args.qwen3_name} + {args.whisper_name} "
            f"({args.dtype}) on {args.device} ..."
        )
        model, feature_extractor, special_ids = build_fresh_model(
            args.qwen3_name, args.whisper_name, dtype, args.attn_implementation, tokenizer
        )
    model.to(args.device)
    model.eval()

    header = (
        f"{'dur(s)':>8}  {'n':>4}  {'mean(ms)':>9}  {'p50(ms)':>9}  "
        f"{'p95(ms)':>9}  {'min(ms)':>9}  {'max(ms)':>9}  {'std(ms)':>8}"
    )
    print(header)
    print("-" * len(header))

    results = []
    for duration_s in args.durations:
        audio = make_audio(duration_s, args.sampling_rate, seed=args.seed)
        samples_ms = bench_one(
            model, tokenizer, feature_extractor, special_ids, audio, args.sampling_rate,
            args.device, args.warmup, args.repeats,
        )
        stats = summarize(duration_s, samples_ms)
        print(stats.to_row())
        results.append(dataclasses.asdict(stats))

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[bench] wrote {args.output}")


if __name__ == "__main__":
    main()
