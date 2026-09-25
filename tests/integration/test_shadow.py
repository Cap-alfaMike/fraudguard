"""Modo sombra: o challenger pontua em paralelo e NUNCA altera a resposta."""

from fastapi.testclient import TestClient

from fraudguard.api.main import create_app


def test_shadow_disabled_by_default(client):
    assert client.get("/model/shadow").json()["enabled"] is False


def test_shadow_scores_without_changing_response(trained, tx):
    settings, _ = trained
    with TestClient(create_app(settings)) as base:
        expected = base.post("/predict", json=tx).json()
    s = settings.model_copy(update={"shadow_model_path": settings.model_path})
    with TestClient(create_app(s)) as c:
        got = c.post("/predict", json=tx).json()
        c.post("/predict/batch", json={"transactions": [tx, tx]})
        info = c.get("/model/shadow").json()
        metrics = c.get("/metrics").text
    assert got["fraud_probability"] == expected["fraud_probability"] and got["decision"] == expected["decision"]
    assert info["enabled"] and info["predictions"] == 3 and info["disagreement_rate"] == 0.0
    assert "fraudguard_shadow_predictions_total" in metrics


def test_broken_shadow_never_breaks_service(trained, tx, tmp_path):
    settings, _ = trained
    s = settings.model_copy(update={"shadow_model_path": tmp_path / "missing.joblib"})
    with TestClient(create_app(s)) as c:
        assert c.get("/health").status_code == 200
        assert c.post("/predict", json=tx).status_code == 200
        assert c.get("/model/shadow").json()["enabled"] is False


def test_shadow_runtime_failure_is_isolated(trained, tx, monkeypatch):
    settings, _ = trained
    s = settings.model_copy(update={"shadow_model_path": settings.model_path})
    with TestClient(create_app(s)) as c:
        shadow = c.app.state.fg.shadow
        monkeypatch.setattr(shadow, "predict_proba", lambda f: (_ for _ in ()).throw(RuntimeError("boom")))
        assert c.post("/predict", json=tx).status_code == 200
        assert c.get("/model/shadow").json()["errors"] == 1
