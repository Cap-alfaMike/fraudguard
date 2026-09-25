"""Testes comportamentais do modelo (inspirados em CheckList, Ribeiro et al., 2020).

Métricas agregadas podem esconder comportamentos inaceitáveis. Estes testes
verificam propriedades que o modelo DEVE ter independentemente do dataset:
invariâncias, direcionalidade e funcionalidade mínima.
"""

import numpy as np
import pytest

from fraudguard.config import RAW_FEATURES, TARGET
from fraudguard.modeling.predictor import FraudPredictor


@pytest.fixture(scope="module")
def model(trained):
    settings, _ = trained
    return FraudPredictor.load(settings.model_path)


@pytest.fixture(scope="module")
def legit(small_df):
    return small_df[small_df[TARGET] == 0][RAW_FEATURES].sample(500, random_state=0).reset_index(drop=True)


def test_invariance_to_calendar_day(model, legit):
    """Mesma hora em outro dia => mesma probabilidade (Time bruto não é feature)."""
    shifted = legit.copy()
    shifted["Time"] += 86_400 * 7
    np.testing.assert_allclose(model.predict_proba(legit), model.predict_proba(shifted), atol=1e-12)


def test_invariance_to_row_order(model, legit):
    p = model.predict_proba(legit)
    rev = model.predict_proba(legit.iloc[::-1])[::-1]
    np.testing.assert_allclose(p, rev, atol=1e-12)


def test_robust_to_tiny_perturbations(model, legit):
    noisy = legit.copy()
    noisy[RAW_FEATURES[1:29]] += np.random.default_rng(0).normal(0, 1e-6, (len(legit), 28))
    assert np.max(np.abs(model.predict_proba(noisy) - model.predict_proba(legit))) < 1e-3


def test_directional_fraud_signature_raises_risk(model, legit):
    """Mover as componentes mais discriminativas na direção das fraudes aumenta o risco médio."""
    pushed = legit.copy()
    for col, delta in (("V14", -4.0), ("V17", -4.0), ("V12", -3.0), ("V10", -3.0)):
        pushed[col] += delta
    assert model.predict_proba(pushed).mean() > 5 * model.predict_proba(legit).mean()


def test_minimum_functionality_obvious_fraud_flagged(model, legit):
    fraud = legit.head(20).copy()
    for col, v in (("V14", -12), ("V17", -12), ("V12", -11), ("V10", -10), ("V3", -12), ("V4", 8), ("V11", 7)):
        fraud[col] = v
    assert np.mean(model.predict_proba(fraud) >= model.threshold) >= 0.95


def test_typical_legit_mostly_approved(model, legit):
    assert np.mean(model.predict_proba(legit) < model.threshold) >= 0.98
