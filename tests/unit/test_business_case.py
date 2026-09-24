import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("bc", Path(__file__).parents[2] / "scripts" / "business_case.py")
bc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bc)


def test_projection_reproduces_test_savings_at_same_volume(trained):
    _, meta = trained
    n = meta["data"]["split"]["test"]["rows"]
    hours = meta["test_business"]["period_hours"]
    per_minute = n / (365 * 24 * 60)  # volume anual = tamanho do teste
    res = bc.project(meta, per_minute)
    full = next(s for s in res["cenarios"] if s["prevalencia_relativa"] == 1.0)
    assert full["economia_eur_ano"] == pytest.approx(meta["test_business"]["savings"], abs=1)
    assert hours > 0


def test_lower_prevalence_lowers_savings(trained):
    _, meta = trained
    s = bc.project(meta, 1000)["cenarios"]
    assert s[0]["economia_eur_ano"] < s[1]["economia_eur_ano"] < s[2]["economia_eur_ano"]
