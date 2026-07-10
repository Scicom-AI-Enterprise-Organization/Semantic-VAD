"""Render clean + telephony-degraded `.wav` files so you can listen to what
`TelephonyDegrader` (semvad/degrade.py) actually does, without needing a GPU or
the training loop. Runs entirely locally (CPU + ffmpeg).

Usage:
  python scripts/preview_telephony_degrade.py
  python scripts/preview_telephony_degrade.py --input some_call.wav --n 6
  python scripts/preview_telephony_degrade.py --output-dir /tmp/telephony_preview

Then, on macOS:
  afplay /tmp/telephony_preview/clean.wav
  afplay /tmp/telephony_preview/degraded_0_mulaw.wav
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import soundfile as sf

sys.path.insert(0, __file__.rsplit("/scripts/", 1)[0])

from semvad.degrade import TelephonyDegrader, _probe_gsm_encode  # noqa: E402


def _load_input(path: str | None) -> tuple[np.ndarray, int]:
    if path:
        audio, sr = sf.read(path, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return audio, sr
    from datasets import load_dataset

    print("[preview] no --input given, pulling one real turn from Scicom-intl/semantic-vad-eot ...")
    row = next(
        iter(
            load_dataset("Scicom-intl/semantic-vad-eot", name="en", split="train", streaming=True)
            .shuffle(buffer_size=50, seed=0)
            .take(1)
        )
    )
    print(f"[preview] using turn id={row.get('id')!r}, transcript: {row.get('messages', [{}])[-1].get('content', '')[:120]!r}")
    return np.asarray(row["audio"]["array"], dtype=np.float32), row["audio"]["sampling_rate"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default=None, help="path to a wav file; defaults to a real dataset turn")
    parser.add_argument("--output-dir", default="telephony_preview")
    parser.add_argument("--n", type=int, default=4, help="number of degraded variants to render")
    parser.add_argument("--noise-prob", type=float, default=0.85)
    parser.add_argument("--packet-loss-prob", type=float, default=0.0)
    args = parser.parse_args()

    print(f"[preview] gsm encode available on this machine: {_probe_gsm_encode()}")

    audio, sr = _load_input(args.input)
    os.makedirs(args.output_dir, exist_ok=True)
    clean_path = os.path.join(args.output_dir, "clean.wav")
    sf.write(clean_path, audio, sr)
    print(f"[preview] wrote {clean_path}  ({len(audio) / sr:.2f}s @ {sr}Hz)")

    degrader = TelephonyDegrader(apply_prob=1.0, noise_prob=args.noise_prob, packet_loss_prob=args.packet_loss_prob)
    for i in range(args.n):
        trace: dict = {}
        out = degrader.degrade(audio, sr, trace=trace)
        codec = trace.get("codec", "none")
        path = os.path.join(args.output_dir, f"degraded_{i}_{codec}.wav")
        sf.write(path, out, sr)
        print(f"[preview] wrote {path}  trace={trace}")


if __name__ == "__main__":
    main()
