import numpy as np
import pandas as pd
import pytest

from fraudguard.config import CostModel
from fraudguard.ews.batch_monitor import (
    BatchMonitorConfig,
    cohort_report,
    expected_calibration_error,
    simulate_label_arrival,
)


def test_ece_perfect_and_bad():
    rng = np.random.default_rng(0)
    p = rng.random(50_000)
    y = rng.random(50_000) < p
    assert expected_calibration_error(y, p) < 0.01
    assert expected_calibration_error(y, np.clip(p * 0.2, 0, 1)) > 0.2


@pytest.fixture
def decisions():
    rng = np.random.default_rng(1)
    n = 6000
    y = rng.random(n) < 0.02
    p = np.clip(np.where(y, rng.beta(2, 4, n), rng.beta(1, 15, n)), 0, 1)
    df = pd.DataFrame(
        {"ts_h": np.sort(rng.uniform(0, 6, n)), "probability": p, "amount": rng.lognormal(3, 1, n), "label": y.astype(int)}
    )
    df["flagged"] = df["probability"] >= 0.3
    df["cohort"] = (df["ts_h"] // 3).astype(int)
    df["label_arrival_h"] = simulate_label_arrival(df, rng)
    return df


def test_flagged_labels_arrive_before_chargebacks(decisions):
    delay = decisions["label_arrival_h"] - decisions["ts_h"]
    fraud = decisions["label"] == 1
    assert delay[fraud & decisions["flagged"]].median() < delay[fraud & ~decisions["flagged"]].median()
    assert np.isinf(delay[~fraud & ~decisions["flagged"]]).all()


def test_immature_cohorts_get_no_metrics(decisions):
    rep = cohort_report(decisions, as_of_h=10, cost=CostModel(), reference_auprc=0.8)
    assert (rep["status"] == "IMATURA").all() and "auprc" not in rep


def test_mature_cohorts_report_metrics_and_full_coverage(decisions):
    rep = cohort_report(decisions, as_of_h=10_000, cost=CostModel(), reference_auprc=0.5, cfg=BatchMonitorConfig())
    assert (rep["label_coverage"] == 1.0).all()
    assert rep["status"].isin(["OK", "DESCALIBRADO"]).all()


def test_degradation_flagged(decisions):
    rep = cohort_report(decisions, as_of_h=10_000, cost=CostModel(), reference_auprc=0.9999)
    assert (rep["status"] == "DEGRADADO").all()
