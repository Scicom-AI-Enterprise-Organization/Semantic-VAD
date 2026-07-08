"""Cheap sanity check: does the model's false-positive rate on `hold` spans rise
with silence duration?

Motivation: `semvad/data.py`'s `TRUNCATION_OFFSETS` caps every training example's
observed trailing silence at the span's real duration (`min(start + offset, end)`),
and training hold spans are short (see `test.ipynb` inspection of
`Scicom-intl/semantic-vad-eot`: p90 ~0.75s, max ~1.3s). `livekit/eot-bench-data`
hold spans run much longer (max ~3.3s), and its eot spans are a fixed 1.5s window.
If the model leans on "long silence -> turn's over" as a shortcut instead of
semantic content, it should misfire on long `hold` spans specifically -- this
script checks that directly before spending effort on synthetic data augmentation.

For each silence span, cuts the causal audio at the span's real end (the single
hardest, most-information point within that span -- no per-100ms probing, this is
meant to be cheap) and scores p(eot). Buckets by span duration and reports, per
bucket: `hold` false-positive rate and `eot` recall.

Run on a GPU box (e.g. the RunPod pod used for training) -- this loads the full
7B backbone, same as scripts/benchmark_latency.py.

Usage:
  python scripts/duration_bias_check.py --checkpoint runs/eot-v4/checkpoint-2500
  python scripts/duration_bias_check.py --checkpoint runs/eot-v4/checkpoint-2500 \\
      --limit 500 --output duration_bias.csv
"""

from __future__ import annotations

import argparse
import csv

import numpy as np
import torch

DEFAULT_BUCKET_EDGES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]


def trim_recent(audio: np.ndarray, sr: int, max_seconds: float) -> np.ndarray:
    """Same "keep the most recent window" rule as `EoTCollator._load_audio`."""
    max_len = int(max_seconds * sr)
    return audio[-max_len:] if len(audio) > max_len else audio


def bucket_of(duration: float, edges: list) -> str:
    for lo, hi in zip(edges, edges[1:]):
        if lo <= duration < hi:
            return f"[{lo:g}, {hi:g})"
    return f">= {edges[-1]:g}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-name", default="Qwen/Qwen2-Audio-7B-Instruct")
    parser.add_argument("--checkpoint", default=None, help="dir written by Qwen2AudioEoTClassifier.save_adapter")
    parser.add_argument("--dataset-path", default="livekit/eot-bench-data")
    parser.add_argument("--dataset-name", default="en")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--limit", type=int, default=300, help="number of rows (turns) to sample")
    parser.add_argument("--seed", type=int, default=42, help="matches the seed used in test.ipynb's inspection")
    parser.add_argument("--max-audio-seconds", type=float, default=16.0)
    parser.add_argument(
        "--bucket-edges", type=float, nargs="+", default=DEFAULT_BUCKET_EDGES,
        help="lower edges of duration buckets in seconds; the last bucket is open-ended",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--output", default=None, help="optional path to dump raw per-span records as CSV")
    args = parser.parse_args()

    from datasets import load_dataset
    from transformers import AutoProcessor

    from semvad.modeling import Qwen2AudioEoTClassifier

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]

    print(f"[bias-check] loading {args.model_name} ({args.dtype}) on {args.device} ...")
    model = Qwen2AudioEoTClassifier.from_pretrained(args.model_name, dtype=dtype)
    if args.checkpoint:
        model.load_adapter(args.checkpoint)
        print(f"[bias-check] loaded adapter from {args.checkpoint}")
    else:
        print("[bias-check] WARNING: no --checkpoint given, scoring with the untrained head")
    model.to(args.device)
    model.eval()

    processor = AutoProcessor.from_pretrained(args.model_name)
    processor.tokenizer.padding_side = "right"

    print(f"[bias-check] streaming {args.dataset_path}/{args.dataset_name} [{args.split}], limit={args.limit} ...")
    ds = (
        load_dataset(args.dataset_path, name=args.dataset_name, split=args.split, streaming=True)
        .shuffle(buffer_size=args.limit, seed=args.seed)
        .take(args.limit)
    )

    records = []
    for row in ds:
        spans = sorted(row["silence_spans"], key=lambda s: s["start"])
        if not spans:
            continue
        audio = np.asarray(row["audio"]["array"], dtype=np.float32)
        sr = row["audio"]["sampling_rate"]
        n = len(spans)
        for idx, span in enumerate(spans):
            label = "eot" if idx == n - 1 else "hold"
            duration = span["end"] - span["start"]
            cut_sample = int(span["end"] * sr)
            if cut_sample <= 0:
                continue
            clip = trim_recent(audio[:cut_sample], sr, args.max_audio_seconds)
            p_eot = model.predict_p_eot(processor, clip, sr)
            records.append({"id": row.get("id"), "label": label, "duration": duration, "p_eot": p_eot})
        if len(records) % 200 < n:
            print(f"[bias-check] ... {len(records)} spans scored")

    if args.output:
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "label", "duration", "p_eot"])
            writer.writeheader()
            writer.writerows(records)
        print(f"[bias-check] wrote {len(records)} raw records to {args.output}")

    edges = sorted(args.bucket_edges)
    # sort buckets by their lower edge, not lexicographically
    buckets = sorted({bucket_of(r["duration"], edges) for r in records}, key=lambda b: float(b.strip(">=[ ").split(",")[0]))

    header = f"{'duration bucket':>16}  {'hold n':>7}  {'hold FP rate':>13}  {'eot n':>6}  {'eot recall':>10}"
    print()
    print(header)
    print("-" * len(header))
    for b in buckets:
        hold = [r["p_eot"] for r in records if r["label"] == "hold" and bucket_of(r["duration"], edges) == b]
        eot = [r["p_eot"] for r in records if r["label"] == "eot" and bucket_of(r["duration"], edges) == b]
        hold_fp = np.mean([p >= 0.5 for p in hold]) if hold else float("nan")
        eot_tp = np.mean([p >= 0.5 for p in eot]) if eot else float("nan")
        print(f"{b:>16}  {len(hold):>7}  {hold_fp:>13.1%}  {len(eot):>6}  {eot_tp:>10.1%}")

    hold_durations = np.array([r["duration"] for r in records if r["label"] == "hold"])
    hold_p_eot = np.array([r["p_eot"] for r in records if r["label"] == "hold"])
    if len(hold_durations) > 1:
        corr = np.corrcoef(hold_durations, hold_p_eot)[0, 1]
        print(f"\n[bias-check] corr(hold span duration, p_eot) = {corr:.3f} (positive => longer hold silence pushes p_eot up)")


if __name__ == "__main__":
    main()
