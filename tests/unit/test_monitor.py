import numpy as np
import pytest

from fraudguard.config import Settings
from fraudguard.ews.drift import MONITORED_FEATURES, bin_proportions, quantile_edges
from fraudguard.ews.monitor import EarlyWarningSystem


def _reference(rng):
    ref = {"features": {}, "score": {}}
    for f in MONITORED_FEATURES:
        v = rng.normal(size=5000)
        e = quantile_edges(v)
        ref["features"][f] = {"edges": e, "proportions": bin_proportions(v, e)}
    scores = rng.beta(1, 300, 5000)
    se = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1, 0.3, 0.5, 0.8]
    ref["score"] = {"edges": se, "proportions": bin_proportions(scores, se), "suspicious_rate": 0.002}
    return ref


@pytest.fixture
def ews():
    s = Settings(ews_min_samples=100, ews_window_size=1000, ews_alert_cooldown_s=300, llm_enabled=False)
    return EarlyWarningSystem(s, _reference(np.random.default_rng(0)), threshold=0.1, eval_every=50)


def feats(rng, shift=0.0):
    return {f: float(rng.normal() + shift) for f in MONITORED_FEATURES}


def test_normal_traffic_is_green(ews):
    rng = np.random.default_rng(1)
    for _ in range(500):
        ews.record(float(rng.beta(1, 300)), False, 30.0, 1.0, feats(rng))
    st = ews.status()
    assert st["health"] == "VERDE" and st["window_size"] == 500
    assert st["score_psi"] < 0.1


def test_card_testing_burst(ews):
    rng = np.random.default_rng(2)
    types = []
    for _ in range(12):
        types += [a.type for a in ews.record(0.9, True, 1.0, 1.0, feats(rng))]
    assert "CARD_TESTING_BURST" in types
    assert ews.status()["health"] == "VERMELHO"


def test_rate_spike_and_prediction_drift(ews):
    rng = np.random.default_rng(3)
    fired = []
    for i in range(300):
        p = 0.6 if i % 10 == 0 else float(rng.beta(1, 300))
        fired += [a.type for a in ews.record(p, p >= 0.1, 50.0, 1.0, feats(rng))]
    assert "SUSPICIOUS_RATE_SPIKE" in fired


def test_data_drift_and_latency(ews):
    rng = np.random.default_rng(4)
    fired = []
    for _ in range(300):
        fired += [a.type for a in ews.record(float(rng.beta(1, 300)), False, 30.0, 120.0, feats(rng, 2.0))]
    assert "DATA_DRIFT" in fired and "LATENCY_SLO_BREACH" in fired


def test_cooldown_deduplicates(ews):
    for _ in range(40):
        ews.record(0.95, True, 5000.0, 1.0, {})
    assert sum(a["type"] == "HIGH_VALUE_FRAUD_ATTEMPT" for a in ews.alerts()) == 1


def test_listener_failure_does_not_break_flow(ews):
    ews.subscribe(lambda a: (_ for _ in ()).throw(RuntimeError("boom")))
    ews.record(0.95, True, 5000.0, 1.0, {})  # não deve levantar
    assert ews.total_events == 1


def test_financial_counters_and_reset(ews):
    ews.record(0.5, True, 100.0, 1.0, {})
    ews.record(0.01, False, 999.0, 1.0, {})
    st = ews.status()
    assert st["value_blocked_eur"] == 100.0 and st["expected_loss_avoided_eur"] == 50.0
    ews.reset()
    assert ews.status()["total_events"] == 0 and ews.alerts() == []


def test_severity_filter(ews):
    ews.record(0.95, True, 5000.0, 1.0, {})
    assert ews.alerts(min_severity="CRITICAL")
