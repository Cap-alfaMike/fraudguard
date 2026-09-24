"""Serviço de inferência: carrega o bundle joblib e expõe predição + explicação.

* O bundle contém o pipeline COMPLETO (feature engineering -> imputação ->
  escalonamento -> LightGBM -> calibração). A API nunca reimplementa
  transformações: elimina training/serving skew por construção.
* Caminho rápido "compilado": no carregamento, os parâmetros aprendidos
  (medianas, centro/escala do RobustScaler, árvores, coeficientes Platt) são
  extraídos do pipeline e aplicados com NumPy puro. O sklearn com saída
  pandas custa ~7 ms/chamada; o caminho compilado, ~0,1 ms. A paridade é
  VERIFICADA no startup contra o pipeline de referência (tolerância 1e-9);
  se divergir, o serviço usa o pipeline sklearn e registra o evento.
* Explicações usam ``pred_contrib=True`` do LightGBM — TreeSHAP exato
  (Lundberg et al., 2020) sem dependência extra do pacote ``shap``.
  As contribuições estão em log-odds do modelo base (antes da calibração
  monotônica), portanto o ranking de fatores é preservado.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from fraudguard.config import RAW_FEATURES
from fraudguard.features.transformers import engineer_array

logger = logging.getLogger(__name__)

REQUIRED_KEYS = {"model", "base_model", "threshold", "version", "model_name", "feature_names"}

FEATURE_LABELS = {
    "log_amount": "valor da transação",
    "hour_sin": "horário do dia",
    "hour_cos": "horário do dia",
    "is_night": "madrugada (0h–6h)",
    "amount_is_zero": "valor zero (verificação de cartão)",
    "amount_is_micro": "micro-valor (≤ €2, padrão de teste de cartão)",
}


class ModelNotLoadedError(RuntimeError):
    pass


@dataclass(frozen=True)
class Prediction:
    probability: float
    is_suspicious: bool
    risk_level: str
    factors: list[dict] | None = None


def risk_level(p: float, threshold: float) -> str:
    if p >= max(0.5, threshold * 4):
        return "CRITICO"
    if p >= threshold:
        return "ALTO"
    if p >= threshold / 3:
        return "MEDIO"
    return "BAIXO"


class _CompiledPipeline:
    """Reprodução NumPy do pipeline ajustado (somente LightGBM + Platt)."""

    def __init__(self, bundle: dict):
        pre = bundle["base_model"].named_steps["preprocess"]
        self.clf = bundle["base_model"].named_steps["classifier"]
        if not hasattr(self.clf, "booster_"):
            raise TypeError("caminho compilado suporta apenas LightGBM")
        cols = pre.named_steps["columns"]
        cont = cols.named_transformers_["cont"]
        self.cont_median = cont.named_steps["impute"].statistics_.astype(float)
        self.center = cont.named_steps["scale"].center_.astype(float)
        self.scale = cont.named_steps["scale"].scale_.astype(float)
        self.flag_fill = cols.named_transformers_["flags"].statistics_.astype(float)
        self.n_cont = len(self.cont_median)
        cal = bundle["model"].calibrated_classifiers_
        if len(cal) != 1:
            raise TypeError("esperado um único calibrador (FrozenEstimator)")
        self.calibrator = cal[0].calibrators[0]
        # O sklearn prioriza decision_function (log-odds) sobre predict_proba ao
        # alimentar o calibrador; replicamos a mesma escolha.
        self.uses_raw_score = hasattr(cal[0].estimator, "decision_function")
        self.booster = self.clf.booster_

    def features(self, raw: np.ndarray) -> np.ndarray:
        x = engineer_array(raw)
        c = x[:, : self.n_cont]
        c = np.where(np.isnan(c), self.cont_median, c)
        x[:, : self.n_cont] = (c - self.center) / self.scale
        f = x[:, self.n_cont :]
        x[:, self.n_cont :] = np.where(np.isnan(f), self.flag_fill, f)
        return x

    def calibrator_input(self, feats: np.ndarray) -> np.ndarray:
        return self.booster.predict(feats, raw_score=self.uses_raw_score, num_threads=1)

    def proba(self, raw: np.ndarray) -> np.ndarray:
        return np.clip(self.calibrator.predict(self.calibrator_input(self.features(raw))), 0.0, 1.0)


def _as_array(frame) -> np.ndarray:
    if isinstance(frame, pd.DataFrame):
        return frame[RAW_FEATURES].to_numpy(dtype=float)
    arr = np.asarray(frame, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != len(RAW_FEATURES):
        raise ValueError(f"esperado array (n, {len(RAW_FEATURES)})")
    return arr


class FraudPredictor:
    def __init__(self, bundle: dict, verify_samples: int = 512):
        missing = REQUIRED_KEYS - bundle.keys()
        if missing:
            raise ValueError(f"Bundle de modelo inválido; chaves ausentes: {sorted(missing)}")
        self.model = bundle["model"]
        self.base_model = bundle["base_model"]
        self.threshold = float(bundle["threshold"])
        self.version = str(bundle["version"])
        self.model_name = str(bundle["model_name"])
        self.feature_names = list(bundle["feature_names"])
        self._compiled: _CompiledPipeline | None = None
        self.fast_path_max_abs_diff: float | None = None
        try:
            compiled = _CompiledPipeline(bundle)
            diff = self._parity(compiled, verify_samples)
            self.fast_path_max_abs_diff = diff
            if diff <= 1e-9:
                self._compiled = compiled
            else:
                logger.warning("caminho compilado divergiu; usando sklearn", extra={"max_abs_diff": diff})
        except Exception as exc:
            logger.warning("caminho compilado indisponível; usando sklearn", extra={"reason": str(exc)})

    @property
    def fast_path(self) -> bool:
        return self._compiled is not None

    def _parity(self, compiled: _CompiledPipeline, n: int) -> float:
        rng = np.random.default_rng(0)
        raw = np.column_stack(
            [
                rng.uniform(0, 172_800, n),
                rng.standard_t(4, size=(n, 28)) * 3,
                np.where(rng.random(n) < 0.1, 0.0, rng.lognormal(3, 1.5, n)),
            ]
        )
        raw[rng.random(raw.shape) < 0.02] = np.nan  # exercita imputação
        ref = self.model.predict_proba(pd.DataFrame(raw, columns=RAW_FEATURES))[:, 1]
        return float(np.max(np.abs(ref - compiled.proba(raw))))

    @classmethod
    def load(cls, path: Path) -> FraudPredictor:
        if not Path(path).exists():
            raise ModelNotLoadedError(f"Artefato não encontrado: {path}")
        predictor = cls(joblib.load(path))
        predictor.warmup()
        logger.info("modelo carregado", extra={"model_version": predictor.version, "fast_path": predictor.fast_path})
        return predictor

    def warmup(self) -> None:
        """Primeira chamada paga custos de inicialização; fazemos no startup, não no cliente."""
        dummy = np.zeros((1, len(RAW_FEATURES)))
        self.predict_proba(dummy)
        self.explain(dummy)

    def predict_proba(self, frame) -> np.ndarray:
        raw = _as_array(frame)
        if self._compiled is not None:
            return self._compiled.proba(raw)
        return self.model.predict_proba(pd.DataFrame(raw, columns=RAW_FEATURES))[:, 1]

    def _features(self, raw: np.ndarray) -> np.ndarray:
        if self._compiled is not None:
            return self._compiled.features(raw)
        pre = self.base_model.named_steps["preprocess"]
        return pre.transform(pd.DataFrame(raw, columns=RAW_FEATURES)).to_numpy()

    def explain(self, frame, top_k: int = 5) -> list[list[dict]]:
        raw = _as_array(frame)
        clf = self.base_model.named_steps["classifier"]
        feats = self._features(raw)
        if hasattr(clf, "booster_"):
            contrib = clf.booster_.predict(feats, pred_contrib=True, num_threads=1)[:, :-1]  # última = bias
        else:  # baseline linear: contribuição = coef * x
            contrib = feats * clf.coef_[0]
        out = []
        for row in contrib:
            idx = np.argsort(-np.abs(row))[:top_k]
            out.append(
                [
                    {
                        "feature": self.feature_names[i],
                        "description": FEATURE_LABELS.get(
                            self.feature_names[i], f"componente {self.feature_names[i]} (anonimizada via PCA)"
                        ),
                        "contribution": round(float(row[i]), 4),
                        "direction": "aumenta risco" if row[i] > 0 else "reduz risco",
                    }
                    for i in idx
                ]
            )
        return out

    def predict(self, frame, explain: bool = False) -> list[Prediction]:
        raw = _as_array(frame)
        proba = self.predict_proba(raw)
        factors = self.explain(raw) if explain else [None] * len(proba)
        return [
            Prediction(
                probability=float(p),
                is_suspicious=bool(p >= self.threshold),
                risk_level=risk_level(float(p), self.threshold),
                factors=f,
            )
            for p, f in zip(proba, factors, strict=True)
        ]
