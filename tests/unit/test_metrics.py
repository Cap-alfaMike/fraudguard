import numpy as np
import pytest

from fraudguard.config import CostModel
from fraudguard.modeling.metrics import (
    business_cost,
    classification_report_at,
    optimize_threshold,
    ranking_metrics,
    recall_at_precision,
)

COST = CostModel(
    review_cost=3, chargeback_fee=15, customer_lifetime_value=400, churn_prob_after_fraud=0.05, churn_prob_after_false_decline=0.01
)


def test_cost_model_components():
    assert COST.false_positive_cost == pytest.approx(7.0)
    assert COST.false_negative_cost(100.0) == pytest.approx(135.0)


def test_business_cost_hand_computed():
    y = np.array([1, 1, 0, 0])
    pred = np.array([1, 0, 1, 0])
    amt = np.array([100.0, 50.0, 10.0, 10.0])
    c = business_cost(y, pred, amt, COST)
    # FN: 50+35=85 ; FP: 7 ; TP review: 3
    assert c.total_cost == pytest.approx(95.0)
    assert c.baseline_cost == pytest.approx(135 + 85)
    assert c.savings == pytest.approx(220 - 95)
    assert c.fraud_loss_prevented == 100 and c.fraud_loss_missed == 50 and c.n_flagged == 2


def test_optimize_threshold_matches_brute_force():
    rng = np.random.default_rng(0)
    y = rng.random(3000) < 0.02
    p = np.clip(np.where(y, rng.beta(5, 2, 3000), rng.beta(1, 20, 3000)), 1e-6, 1 - 1e-6)
    amt = rng.lognormal(3, 1, 3000)
    t, cost, (grid, costs) = optimize_threshold(y, p, amt, COST)
    brute = [business_cost(y, p >= g, amt, COST).total_cost for g in grid[::25]]
    np.testing.assert_allclose(costs[::25], brute, rtol=1e-9)
    assert cost == pytest.approx(min(costs)) and 0 < t < 1


def test_classification_report_at():
    r = classification_report_at([1, 0, 1, 0], [0.9, 0.8, 0.1, 0.05], 0.5)
    assert r["confusion_matrix"] == {"tp": 1, "fp": 1, "fn": 1, "tn": 1}
    assert r["precision"] == 0.5 and r["recall"] == 0.5


def test_ranking_metrics_perfect():
    m = ranking_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert m["auprc"] == 1.0 and m["roc_auc"] == 1.0
    assert recall_at_precision([0, 1], [0.9, 0.1], 0.99) >= 0.0
