import numpy as np
import pytest

from fraudguard.analysis.eda import run_eda
from fraudguard.config import RAW_FEATURES
from fraudguard.ews.simulator import scenario_frame


def test_eda_report_sections_and_decisions(small_df):
    report = run_eda(small_df, "test", make_plots=False)
    for section in ("Desbalanceamento", "Missing Analysis", "Duplicatas", "Padrão temporal", "KS", "Outliers"):
        assert section in report
    assert report.count("**Decisão:**") >= 8


def test_eda_reports_missing(small_df):
    df = small_df.copy()
    df.loc[df.index[:30], "V5"] = np.nan
    assert "| V5 |" in run_eda(df, "test", make_plots=False)


@pytest.mark.parametrize("scenario", ["normal", "card_testing", "drift", "high_value"])
def test_scenarios(scenario):
    df = scenario_frame(scenario, 200, seed=1)
    assert len(df) == 200 and list(df.columns) == RAW_FEATURES


def test_card_testing_has_micro_amounts():
    df = scenario_frame("card_testing", 200, seed=1)
    assert (df["Amount"] < 2).sum() >= 60


def test_unknown_scenario():
    with pytest.raises(ValueError):
        scenario_frame("meteoro", 10)
