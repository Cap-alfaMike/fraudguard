import json

import pytest

from fraudguard.modeling.registry import GatePolicy, ModelRegistry, promotion_gate
from fraudguard.modeling.uncertainty import Interval, PairedComparison


def _artifact(tmp_path, version, content=b"model"):
    d = tmp_path / f"src-{version}"
    d.mkdir()
    (d / "model.joblib").write_bytes(content)
    (d / "meta.json").write_text(json.dumps({"version": version, "model_name": "m", "test_metrics": {"auprc": 0.7}}))
    return d / "model.joblib", d / "meta.json"


def test_register_and_stages(tmp_path):
    reg = ModelRegistry(tmp_path / "reg")
    v1 = reg.register(*_artifact(tmp_path, "v1"))
    v2 = reg.register(*_artifact(tmp_path, "v2", b"other"))
    reg.set_stage(v1, "champion")
    reg.set_stage(v2, "champion", "melhor")
    assert reg.get("champion") == "v2" and reg.get("archived") == "v1"
    assert reg.verify("v2")
    assert [h["event"] for h in reg.history()][-2:] == ["archived", "stage:champion"]


def test_register_is_idempotent_but_immutable(tmp_path):
    reg = ModelRegistry(tmp_path / "reg")
    m, meta = _artifact(tmp_path, "v1")
    assert reg.register(m, meta) == reg.register(m, meta)
    m.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="imutabilidade"):
        reg.register(m, meta)


def test_verify_detects_tampering(tmp_path):
    reg = ModelRegistry(tmp_path / "reg")
    v = reg.register(*_artifact(tmp_path, "v1"))
    reg.path(v).write_bytes(b"changed")
    assert not reg.verify(v)


def test_invalid_stage_and_unknown_version(tmp_path):
    reg = ModelRegistry(tmp_path / "reg")
    v = reg.register(*_artifact(tmp_path, "v1"))
    with pytest.raises(ValueError):
        reg.set_stage(v, "production")
    with pytest.raises(KeyError):
        reg.set_stage("nope", "shadow")


def _cmp(d_auprc_low, p_cost, fpr_high):
    iv = lambda lo, hi: Interval(0.0, lo, hi, 0.95)
    return PairedComparison(iv(d_auprc_low, 0.02), iv(-50, 10), iv(-0.0001, fpr_high), 0.9, p_cost)


def test_gate_passes_good_challenger():
    assert promotion_gate(_cmp(-0.005, 0.95, 0.0001)).passed


@pytest.mark.parametrize("args", [(-0.05, 0.95, 0.0001), (0.0, 0.5, 0.0001), (0.0, 0.95, 0.01)])
def test_gate_rejects_each_violation(args):
    res = promotion_gate(_cmp(*args), GatePolicy())
    assert not res.passed and len(res.reasons) == 3
