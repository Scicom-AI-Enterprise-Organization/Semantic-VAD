"""Find the "sweet spot" gap that separates mid-turn pauses from turn boundaries.

Inter-word gaps in conversational speech are multi-modal: a dense cluster of short
between-word gaps, and a sparser cluster of longer between-turn / between-sentence gaps.
The threshold that best splits ``hold`` from ``eot`` sits in the valley between them.

This module estimates that valley with a numpy-only Gaussian KDE over ``log10(gap)`` and
also reports percentiles, so you can pick ``TurnConfig.turn_gap`` with evidence rather
than a guess. No plotting dependencies -- it prints a text histogram.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np


@dataclass
class GapAnalysis:
    n_gaps: int
    floor: float
    percentiles: dict[int, float]
    valley: float | None
    modes: list[float]
    histogram: str

    def recommended_turn_gap(self) -> float:
        """Best single threshold: the KDE valley, else the 75th percentile."""
        if self.valley is not None:
            return round(self.valley, 3)
        return round(self.percentiles.get(75, self.floor * 5), 3)

    def report(self) -> str:
        lines = [
            f"gaps analyzed : {self.n_gaps:,} (>= {self.floor:.3f}s floor)",
            "percentiles   : " + ", ".join(f"p{p}={v:.3f}s" for p, v in sorted(self.percentiles.items())),
            "kde modes     : " + (", ".join(f"{m:.3f}s" for m in self.modes) or "n/a"),
            "kde valley    : " + (f"{self.valley:.3f}s" if self.valley is not None else "not found (unimodal)"),
            f"RECOMMENDED turn_gap = {self.recommended_turn_gap():.3f}s",
            "",
            self.histogram,
        ]
        return "\n".join(lines)


def _kde_on_grid(log_gaps: np.ndarray, grid: np.ndarray, bandwidth: float) -> np.ndarray:
    # Gaussian KDE evaluated on ``grid`` (both in log10 space).
    diffs = (grid[:, None] - log_gaps[None, :]) / bandwidth
    kernel = np.exp(-0.5 * diffs * diffs)
    return kernel.sum(axis=1) / (len(log_gaps) * bandwidth * np.sqrt(2 * np.pi))


def _local_extrema(y: np.ndarray) -> tuple[list[int], list[int]]:
    maxima, minima = [], []
    for i in range(1, len(y) - 1):
        if y[i] > y[i - 1] and y[i] >= y[i + 1]:
            maxima.append(i)
        elif y[i] < y[i - 1] and y[i] <= y[i + 1]:
            minima.append(i)
    return maxima, minima


def _text_histogram(gaps: np.ndarray, bins: int = 30, width: int = 50) -> str:
    lo, hi = np.log10(gaps.min()), np.log10(gaps.max())
    if hi <= lo:
        hi = lo + 1.0
    edges = np.logspace(lo, hi, bins + 1)
    counts, _ = np.histogram(gaps, bins=edges)
    peak = counts.max() or 1
    rows = ["log-spaced gap histogram (seconds):"]
    for k in range(bins):
        bar = "#" * int(round(width * counts[k] / peak))
        rows.append(f"  {edges[k]:6.3f}-{edges[k+1]:6.3f} | {bar} {counts[k]}")
    return "\n".join(rows)


def analyze_gaps(gaps, floor: float = 0.1, bandwidth: float = 0.12) -> GapAnalysis:
    """Analyze a collection of gaps and estimate the hold/eot threshold.

    Parameters
    ----------
    gaps:
        Iterable of inter-word gaps in seconds (e.g. flattened ``compute_gaps`` output).
    floor:
        Ignore gaps below this (normal between-word timing, not decision points).
    bandwidth:
        Gaussian KDE bandwidth in log10 space; larger = smoother.
    """
    arr = np.asarray([g for g in gaps if g is not None and g >= floor], dtype=np.float64)
    if arr.size == 0:
        return GapAnalysis(0, floor, {}, None, [], "no gaps >= floor")

    pcts = {p: float(np.percentile(arr, p)) for p in (50, 75, 90, 95, 99)}
    hist = _text_histogram(arr)

    valley = None
    modes: list[float] = []
    if arr.size >= 20 and arr.max() > arr.min():
        log_gaps = np.log10(arr)
        grid = np.linspace(log_gaps.min(), log_gaps.max(), 200)
        dens = _kde_on_grid(log_gaps, grid, bandwidth)
        maxima, minima = _local_extrema(dens)
        modes = [round(float(10 ** grid[i]), 3) for i in maxima]
        if len(maxima) >= 2:
            # Two tallest modes; the deepest minimum between them is the valley.
            top2 = sorted(maxima, key=lambda i: dens[i], reverse=True)[:2]
            a, b = sorted(top2)
            between = [m for m in minima if a < m < b]
            if between:
                vi = min(between, key=lambda i: dens[i])
                valley = float(10 ** grid[vi])
    return GapAnalysis(int(arr.size), floor, pcts, valley, modes, hist)


def main(argv: list[str] | None = None) -> None:
    """CLI: stream a source corpus, collect inter-word gaps, print the sweet-spot report."""
    import argparse

    from .sources import SOURCES
    from .turns import compute_gaps

    p = argparse.ArgumentParser(description="Analyze inter-word gaps to find the EOT threshold.")
    p.add_argument("--source", required=True, choices=list(SOURCES))
    p.add_argument("--config", required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--limit", type=int, default=300, help="source recordings to sample")
    p.add_argument("--floor", type=float, default=0.1)
    p.add_argument("--malaysian-mode", default="streaming", choices=["streaming", "whole"])
    args = p.parse_args(argv)

    kwargs = dict(split=args.split, limit=args.limit, streaming=True)
    if args.source == "malaysian":
        kwargs["source_mode"] = args.malaysian_mode

    all_gaps: list[float] = []
    for rec in SOURCES[args.source](args.config, **kwargs):
        all_gaps.extend(compute_gaps(rec.words))
    import sys

    print(analyze_gaps(all_gaps, floor=args.floor).report(), flush=True)
    sys.stdout.flush()
    # Streaming readers spawn native threads that can crash at finalization; exit cleanly.
    os._exit(0)


if __name__ == "__main__":
    main()
