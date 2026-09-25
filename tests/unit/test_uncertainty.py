import numpy as np
import pytest

from fraudguard.config import CostModel
from fraudguard.modeling.uncertainty import block_bootstrap_indices, bootstrap_metrics, paired_bootstrap, point_metrics


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    n = 20_000
    y = rng.random(n) < 0.01
    p = np.clip(np.where(y, rng.beta(2, 4, n), rng.beta(1, 15, n)), 1e-6, 1 - 1e-6)
    return y, p, rng.lognormal(3, 1, n)


def test_block_indices_shape_and_contiguity():
    rng = np.random.default_rng(1)
    idx = block_bootstrap_indices(1000, 100, rng)
    assert len(idx) == 1000 and idx.min() >= 0 and idx.max() < 1000
    assert np.all(np.diff(idx[:100]) == 1)  # primeiro bloco é contíguo


def test_block_indices_rejects_empty():
    with pytest.raises(ValueError):
        block_bootstrap_indices(0, 10, np.random.default_rng(0))


def test_interval_contains_point_and_is_ordered(data):
    y, p, amt = data
    ci = bootstrap_metrics(y, p, amt, 0.3, CostModel(), n_boot=200, block_size=200)
    for iv in ci.values():
        assert iv.low <= iv.point <= iv.high


def test_more_data_narrows_interval(data):
    y, p, amt = data
    small = bootstrap_metrics(y[:4000], p[:4000], amt[:4000], 0.3, CostModel(), n_boot=200, block_size=100)
    large = bootstrap_metrics(y, p, amt, 0.3, CostModel(), n_boot=200, block_size=100)
    assert (large["auprc"].high - large["auprc"].low) < (small["auprc"].high - small["auprc"].low)


def test_point_metrics_match_definitions(data):
    y, p, amt = data
    m = point_metrics(y, p, amt, 0.3, CostModel())
    pred = p >= 0.3
    assert m["recall"] == pytest.approx((pred & y).sum() / y.sum())


def test_paired_identical_models_have_zero_delta(data):
    y, p, amt = data
    cmp = paired_bootstrap(y, p, p, amt, 0.3, 0.3, CostModel(), n_boot=100)
    assert cmp.delta_auprc.low == cmp.delta_auprc.high == 0.0
    assert cmp.prob_auprc_better == 0.0


def test_paired_detects_better_model(data):
    y, p, amt = data
    noisy = np.clip(p + np.random.default_rng(3).normal(0, 0.15, len(p)), 0, 1)
    cmp = paired_bootstrap(y, noisy, p, amt, 0.3, 0.3, CostModel(), n_boot=200)
    assert cmp.delta_auprc.low > 0 and cmp.prob_auprc_better > 0.95
