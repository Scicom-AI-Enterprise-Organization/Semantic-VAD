"""Shared classification metrics for p(eot) predictions vs. ground-truth labels.

Used by `semvad.train.compute_metrics` (training-time eval) and
`scripts/benchmark_eot.py` (offline benchmarking of arbitrary adapters, e.g. the
LiveKit Turn Detector v1 cloud model) so both report numbers on the same
definition.
"""

from __future__ import annotations

import numpy as np


def compute_classification_metrics(probs, labels) -> dict[str, float]:
    """accuracy/f1/auc for `p(eot)` predictions against binary `labels`.

    `hold` outnumbers `eot` in this dataset (every turn contributes exactly one
    `eot` span but n_spans - 1 `hold` spans, see
    `semvad.data.iter_causal_examples`), so accuracy alone can look good while
    the model misses the minority `eot` class -- `f1_score` defaults to
    `pos_label=1`, i.e. the `eot` class.
    """
    probs = np.asarray(probs, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    preds = (probs >= 0.5).astype(int)
    metrics = {"accuracy": float((preds == labels.astype(int)).mean())}
    try:
        from sklearn.metrics import f1_score, roc_auc_score

        metrics["f1"] = float(f1_score(labels.astype(int), preds, zero_division=0))
        if len(set(labels.astype(int).tolist())) > 1:
            metrics["auc"] = float(roc_auc_score(labels, probs))
    except ImportError:
        pass
    return metrics
