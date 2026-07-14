import numpy as np

from semantic_vad.analyze import analyze_gaps


def test_analyze_finds_valley_in_bimodal_gaps():
    rng = np.random.default_rng(42)
    # Two clusters: within-turn pauses ~0.15s, between-turn gaps ~1.2s.
    holds = 10 ** rng.normal(np.log10(0.15), 0.12, size=2000)
    boundaries = 10 ** rng.normal(np.log10(1.2), 0.12, size=600)
    gaps = np.concatenate([holds, boundaries])

    res = analyze_gaps(gaps, floor=0.1)
    assert res.n_gaps > 2000
    assert res.valley is not None
    # The valley should sit between the two modes.
    assert 0.2 < res.valley < 1.0
    rec = res.recommended_turn_gap()
    assert 0.2 < rec < 1.0
    assert "RECOMMENDED turn_gap" in res.report()


def test_analyze_handles_empty_and_unimodal():
    assert analyze_gaps([], floor=0.1).n_gaps == 0
    uni = analyze_gaps(list(0.15 + 0.01 * np.random.default_rng(1).standard_normal(500)), floor=0.1)
    assert uni.n_gaps > 0
    # Unimodal -> no valley, falls back to a percentile.
    assert uni.recommended_turn_gap() > 0
