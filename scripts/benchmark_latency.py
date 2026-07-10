"""Inference latency benchmark across variable audio input lengths.

Measures wall-clock latency of `Qwen2AudioEoTClassifier.predict_p_eot` -- the
single causal forward pass used by both the eot-harness adapter
(semvad/eot_adapter.py) and the Gradio demo (app/gradio_app.py) -- for a range
of audio clip durations. This loads the full 7B backbone; run it on a GPU box
(e.g. the RunPod pod used for training), not a laptop.

Note: Qwen2-Audio's feature extractor pads every clip to a fixed 30s mel grid
regardless of actual length (see the comment in semvad/eot_adapter.py), so the
audio-tower cost should stay roughly flat across durations -- what actually
scales with clip length is the LLM trunk's attention over the audio-token
sequence plus any `--prior-text`. This benchmark is what confirms or refutes
that in practice.

Usage:
  python scripts/benchmark_latency.py --device cuda
  python scripts/benchmark_latency.py --device cuda --checkpoint runs/eot-v1 \\
      --durations 0.5 1 2 4 8 16 24 30 --repeats 20 --output latency.json
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


def bench_one(model, processor, audio, sampling_rate, prior_text, device, warmup, repeats) -> list:
    for _ in range(warmup):
        model.predict_p_eot(processor, audio, sampling_rate, prior_text=prior_text)
    if device.startswith("cuda"):
        torch.cuda.synchronize()

    samples_ms = []
    for _ in range(repeats):
        start = time.perf_counter()
        model.predict_p_eot(processor, audio, sampling_rate, prior_text=prior_text)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        samples_ms.append((time.perf_counter() - start) * 1000)
    return samples_ms


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-name", default="Qwen/Qwen2-Audio-7B-Instruct")
    parser.add_argument("--checkpoint", default=None, help="dir written by Qwen2AudioEoTClassifier.save_adapter")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument(
        "--durations", type=float, nargs="+", default=[0.5, 1, 2, 4, 8, 16, 24, 30],
        help="audio clip durations in seconds to benchmark",
    )
    parser.add_argument("--sampling-rate", type=int, default=16000)
    parser.add_argument("--prior-text", default="", help="simulated transcript-so-far appended to the prompt")
    parser.add_argument("--warmup", type=int, default=3, help="untimed calls per duration, to settle caches/cudnn")
    parser.add_argument("--repeats", type=int, default=10, help="timed calls per duration")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default=None, help="optional path to dump results as JSON")
    parser.add_argument("--attn_implementation", default="sdpa")
    args = parser.parse_args()

    from transformers import AutoProcessor

    from semvad.modeling import Qwen2AudioEoTClassifier

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]

    print(f"[bench] loading {args.model_name} ({args.dtype}) on {args.device} ...")
    model = Qwen2AudioEoTClassifier.from_pretrained(args.model_name, dtype=dtype, attn_implementation=args.attn_implementation)
    if args.checkpoint:
        model.load_adapter(args.checkpoint)
    model.to(args.device)
    model.eval()

    processor = AutoProcessor.from_pretrained(args.model_name)
    processor.tokenizer.padding_side = "right"

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
            model, processor, audio, args.sampling_rate, args.prior_text,
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
