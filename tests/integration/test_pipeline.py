"""Integração do pipeline de treino -> artefato -> inferência."""

import json

import joblib
import numpy as np
import pytest

from fraudguard.config import RAW_FEATURES, TARGET
from fraudguard.data.loader import temporal_split
from fraudguard.modeling.predictor import FraudPredictor, ModelNotLoadedError, risk_level


def test_artifacts_written(trained):
    settings, meta = trained
    assert settings.model_path.exists() and settings.metadata_path.exists() and settings.reference_path.exists()
    on_disk = json.loads(settings.metadata_path.read_text())
    assert on_disk["version"] == meta["version"]
    assert set(meta["candidates"]) == {"logreg_balanced", "lgbm_unweighted", "lgbm_sqrt_weight", "lgbm_full_weight"}
    assert 0 < meta["threshold"] < 1


def test_bundle_is_self_contained_pipeline(trained, small_df):
    """O artefato recebe o JSON bruto (30 colunas): nenhuma transformação fora dele."""
    settings, _ = trained
    bundle = joblib.load(settings.model_path)
    proba = bundle["model"].predict_proba(small_df[RAW_FEATURES].head(10))[:, 1]
    assert proba.shape == (10,) and np.all((proba >= 0) & (proba <= 1))


def test_model_selection_chose_best_valid_auprc(trained):
    _, meta = trained
    best = max(meta["candidates"], key=lambda k: meta["candidates"][k]["valid_auprc"])
    assert meta["model_name"] == best


def test_test_set_untouched_by_selection(trained, small_df):
    """Tamanho do teste no metadata = último bloco temporal após dedup."""
    _, meta = trained
    split = temporal_split(small_df.drop_duplicates().reset_index(drop=True))
    assert meta["data"]["split"]["test"]["rows"] == len(split.test)


def test_model_beats_random_ranking(trained):
    _, meta = trained
    base_rate = meta["data"]["split"]["test"]["frauds"] / meta["data"]["split"]["test"]["rows"]
    assert meta["test_metrics"]["auprc"] > 10 * base_rate
    assert meta["test_business"]["savings"] > 0


def test_fast_path_parity(trained, small_df):
    settings, _ = trained
    pred = FraudPredictor.load(settings.model_path)
    assert pred.fast_path and pred.fast_path_max_abs_diff <= 1e-9
    X = small_df[RAW_FEATURES].sample(500, random_state=0)
    ref = pred.model.predict_proba(X)[:, 1]
    np.testing.assert_allclose(pred.predict_proba(X), ref, atol=1e-9)


def test_fraud_scores_higher_than_legit(trained, small_df):
    settings, _ = trained
    pred = FraudPredictor.load(settings.model_path)
    test = small_df.iloc[-6000:]
    p = pred.predict_proba(test)
    assert p[test[TARGET] == 1].mean() > 20 * p[test[TARGET] == 0].mean()


def test_explanations(trained, small_df):
    settings, _ = trained
    pred = FraudPredictor.load(settings.model_path)
    out = pred.predict(small_df[RAW_FEATURES].head(3), explain=True)
    assert len(out) == 3 and len(out[0].factors) == 5
    assert {"feature", "description", "contribution", "direction"} <= out[0].factors[0].keys()


def test_deterministic_predictions(trained, small_df):
    settings, _ = trained
    a = FraudPredictor.load(settings.model_path).predict_proba(small_df.head(50))
    b = FraudPredictor.load(settings.model_path).predict_proba(small_df.head(50))
    np.testing.assert_array_equal(a, b)


def test_invalid_bundle_and_missing_file(tmp_path):
    with pytest.raises(ValueError, match="inválido"):
        FraudPredictor({"model": None})
    with pytest.raises(ModelNotLoadedError):
        FraudPredictor.load(tmp_path / "missing.joblib")


def test_predictor_rejects_wrong_shape(trained):
    settings, _ = trained
    with pytest.raises(ValueError):
        FraudPredictor.load(settings.model_path).predict_proba(np.zeros((1, 5)))


def test_nan_input_is_imputed_in_batch_scoring(trained, small_df):
    settings, _ = trained
    X = small_df[RAW_FEATURES].head(5).copy()
    X.iloc[0, 5] = np.nan
    p = FraudPredictor.load(settings.model_path).predict_proba(X)
    assert np.isfinite(p).all()


@pytest.mark.parametrize("p,level", [(0.0001, "BAIXO"), (0.04, "MEDIO"), (0.2, "ALTO"), (0.9, "CRITICO")])
def test_risk_levels(p, level):
    assert risk_level(p, threshold=0.1) == level


def test_sklearn_fallback_when_fast_path_unavailable(trained, small_df, monkeypatch):
    """Se o caminho compilado não puder ser construído, o serviço continua correto."""
    import fraudguard.modeling.predictor as mod

    settings, _ = trained
    fast = FraudPredictor.load(settings.model_path)
    monkeypatch.setattr(mod, "_CompiledPipeline", lambda b: (_ for _ in ()).throw(TypeError("x")))
    slow = FraudPredictor.load(settings.model_path)
    assert not slow.fast_path
    X = small_df[RAW_FEATURES].head(100)
    np.testing.assert_allclose(slow.predict_proba(X), fast.predict_proba(X), atol=1e-9)
    assert len(slow.predict(X.head(2), explain=True)[0].factors) == 5
