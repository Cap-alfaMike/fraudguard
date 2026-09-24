"""Contrato da API (FastAPI TestClient, app real com modelo real)."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from fraudguard.api.main import create_app
from fraudguard.config import Settings


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["model_loaded"] and body["model_version"]


def test_liveness(client):
    assert client.get("/health/live").json() == {"status": "alive"}


def test_predict_contract(client, tx):
    r = client.post("/predict", json=tx)
    assert r.status_code == 200
    b = r.json()
    assert set(b) >= {"transaction_id", "fraud_probability", "decision", "risk_level", "threshold", "model_version", "latency_ms"}
    assert b["transaction_id"] == "tx-001"
    assert 0 <= b["fraud_probability"] <= 1
    assert b["decision"] in {"APROVADO", "SUSPEITO"}
    assert b["decision"] == ("SUSPEITO" if b["fraud_probability"] >= b["threshold"] else "APROVADO")
    assert b["latency_ms"] >= 0 and b["explanation"] is None
    assert "x-request-id" in r.headers and "x-process-time-ms" in r.headers


def test_predict_with_explanation(client, tx):
    b = client.post("/predict?explain=true", json=tx).json()
    assert len(b["explanation"]) == 5


def test_request_id_propagated(client, tx):
    r = client.post("/predict", json=tx, headers={"X-Request-ID": "abc-123"})
    assert r.headers["x-request-id"] == "abc-123"


def test_obvious_fraud_is_flagged(client, tx):
    fraud = {
        **tx,
        "V14": -12.0,
        "V17": -12.0,
        "V12": -11.0,
        "V10": -10.0,
        "V3": -12.0,
        "V4": 8.0,
        "V11": 7.0,
        "V7": -10.0,
        "V16": -8.0,
        "V1": -8.0,
        "Amount": 1.0,
        "Time": 2 * 3600.0,
    }
    b = client.post("/predict", json=fraud).json()
    assert b["decision"] == "SUSPEITO" and b["risk_level"] in {"ALTO", "CRITICO"}


@pytest.mark.parametrize(
    "patch",
    [
        {"Amount": -5},
        {"Amount": "10"},
        {"V1": 1e6},
        {"Time": -1},
        {"unknown_field": 1},
        {"V2": None},
    ],
)
def test_predict_validation_422(client, tx, patch):
    r = client.post("/predict", json={**tx, **patch})
    assert r.status_code == 422 and "detail" in r.json()


def test_predict_missing_field_422(client, tx):
    tx.pop("V7")
    assert client.post("/predict", json=tx).status_code == 422


def test_predict_nan_rejected(client, tx):
    body = str({**tx, "V5": 0.0}).replace("'", '"').replace("0.0,", "NaN,", 1)
    r = client.post("/predict", content=body, headers={"content-type": "application/json"})
    assert r.status_code == 422


def test_predict_malformed_json(client):
    r = client.post("/predict", content="{not json", headers={"content-type": "application/json"})
    assert r.status_code == 422


def test_batch(client, tx):
    items = [{**tx, "transaction_id": f"tx-{i}", "Amount": float(i)} for i in range(25)]
    r = client.post("/predict/batch", json={"transactions": items})
    assert r.status_code == 200
    b = r.json()
    assert b["count"] == 25 and len(b["results"]) == 25
    assert [x["transaction_id"] for x in b["results"]] == [f"tx-{i}" for i in range(25)]


def test_batch_consistent_with_single(client, tx):
    single = client.post("/predict", json=tx).json()["fraud_probability"]
    batch = client.post("/predict/batch", json={"transactions": [tx, tx]}).json()
    assert all(np.isclose(r["fraud_probability"], single) for r in batch["results"])


def test_batch_too_large(trained, tx):
    settings, _ = trained
    s = settings.model_copy(update={"max_batch_size": 3})
    with TestClient(create_app(s)) as c:
        assert c.post("/predict/batch", json={"transactions": [tx] * 4}).status_code == 413


def test_model_info(client):
    b = client.get("/model/info").json()
    assert "test_metrics" in b and "cost_model" in b and "feature_names" not in b


def test_ews_flow(client, tx):
    client.post("/ews/reset")
    r = client.post("/ews/simulate", json={"scenario": "card_testing", "n": 400})
    assert r.status_code == 200 and r.json()["suspicious"] > 0
    st = client.get("/ews/status").json()
    assert st["total_events"] == 400 and st["health"] in {"VERDE", "AMARELO", "VERMELHO"}
    assert any(a["type"] == "CARD_TESTING_BURST" for a in client.get("/ews/alerts").json())
    rep = client.post("/ews/report", json={"audience": "executivo"}).json()
    assert rep["source"] == "template" and "## Situação" in rep["content"]
    assert client.get("/ews/report/latest").status_code == 200


def test_ews_report_invalid_audience(client):
    assert client.post("/ews/report", json={"audience": "x"}).status_code == 422


def test_simulate_disabled_outside_demo(trained):
    settings, _ = trained
    with TestClient(create_app(settings.model_copy(update={"demo_mode": False}))) as c:
        assert c.post("/ews/simulate", json={"scenario": "normal"}).status_code == 404


def test_metrics_endpoint(client, tx):
    client.post("/predict", json=tx)
    text = client.get("/metrics").text
    assert "fraudguard_predictions_total" in text and "fraudguard_inference_latency_seconds" in text


def test_dashboard_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]


def test_openapi_documents_contract(client):
    paths = client.get("/openapi.json").json()["paths"]
    assert {"/health", "/predict", "/predict/batch", "/ews/status", "/ews/report"} <= set(paths)


def test_unavailable_model_returns_503(tmp_path, tx):
    s = Settings(artifacts_dir=tmp_path, llm_enabled=False)
    with TestClient(create_app(s)) as c:
        assert c.get("/health").status_code == 503
        assert c.get("/health/live").status_code == 200
        assert c.post("/predict", json=tx).status_code == 503
