import numpy as np
import pandas as pd

from fraudguard.ews.drift import MONITORED_FEATURES, bin_proportions, build_reference_profile, psi, quantile_edges


def test_psi_zero_for_identical():
    assert psi([0.25] * 4, [0.25] * 4) == 0.0


def test_psi_detects_shift():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=20_000)
    edges = quantile_edges(ref)
    same = psi(bin_proportions(ref, edges), bin_proportions(rng.normal(size=5000), edges))
    shifted = psi(bin_proportions(ref, edges), bin_proportions(rng.normal(1.0, 1, 5000), edges))
    assert same < 0.02 and shifted > 0.25


def test_bin_proportions_sum_to_one_and_ignore_nan():
    p = bin_proportions([1, 2, np.nan, 3, 4], [1.5, 3.5])
    assert np.isclose(sum(p), 1.0) and len(p) == 3


def test_reference_profile_structure():
    rng = np.random.default_rng(1)
    feats = pd.DataFrame({f: rng.normal(size=500) for f in MONITORED_FEATURES})
    prof = build_reference_profile(feats, rng.random(500) * 0.01, threshold=0.005)
    assert set(prof["features"]) == set(MONITORED_FEATURES)
    assert 0 < prof["score"]["suspicious_rate"] < 1
