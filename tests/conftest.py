"""Fixtures compartilhadas.

O modelo de teste é treinado do zero em diretório temporário (dados
sintéticos reduzidos, modo rápido): os testes não dependem de artefatos
pré-existentes e validam o pipeline de treino de ponta a ponta.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from fraudguard.config import PCA_FEATURES, Settings
from fraudguard.data.synthetic import generate_synthetic_creditcard


@pytest.fixture(scope="session")
def small_df():
    return generate_synthetic_creditcard(n_samples=30_000, n_frauds=300, seed=123)


@pytest.fixture(scope="session")
def trained(tmp_path_factory, small_df):
    from fraudguard.modeling.train import train

    art = tmp_path_factory.mktemp("artifacts")
    settings = Settings(artifacts_dir=art, llm_enabled=False, log_json=True, ews_min_samples=50)
    metadata = train(settings=settings, df=small_df.copy(), fast=True, make_plots=False)
    return settings, metadata


@pytest.fixture
def client(trained):
    from fraudguard.api.main import create_app

    settings, _ = trained
    with TestClient(create_app(settings)) as c:
        yield c


@pytest.fixture
def tx():
    rng = np.random.default_rng(0)
    payload = {f: float(v) for f, v in zip(PCA_FEATURES, rng.normal(size=28), strict=True)}
    return {"transaction_id": "tx-001", "Time": 3600.0, "Amount": 42.5, **payload}
