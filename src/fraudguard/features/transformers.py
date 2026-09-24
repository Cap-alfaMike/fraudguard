"""Feature engineering e pré-processamento.

Princípio anti-leakage: tudo que APRENDE parâmetros dos dados (imputação,
escalonamento) vive dentro do ``sklearn.Pipeline`` e só é ajustado no treino.
As transformações do ``FraudFeatureEngineer`` são *stateless* (funções puras
por linha), portanto idênticas em treino e em produção — sem training/serving
skew.

Por que descartar ``Time`` bruto?
No dataset ULB, ``Time`` é "segundos desde a primeira transação do arquivo":
um contador monotônico. Usado cru, o modelo aprenderia o *período* da coleta,
e em produção todos os valores estariam fora da distribuição de treino.
Mantemos apenas sua projeção cíclica (hora do dia), que generaliza.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from fraudguard.config import PCA_FEATURES, RAW_FEATURES

SECONDS_PER_DAY = 86_400
MICRO_AMOUNT = 2.0

ENGINEERED_CONTINUOUS = [*PCA_FEATURES, "log_amount"]
ENGINEERED_FLAGS = ["hour_sin", "hour_cos", "is_night", "amount_is_zero", "amount_is_micro"]


def engineer_array(raw: np.ndarray) -> np.ndarray:
    """Feature engineering vetorizado em NumPy puro.

    Recebe ``raw`` com colunas na ordem de ``RAW_FEATURES`` e devolve as
    colunas ``ENGINEERED_CONTINUOUS + ENGINEERED_FLAGS``. É a ÚNICA
    implementação das regras: o transformer sklearn (treino) e o caminho
    rápido de inferência (produção) chamam esta mesma função, eliminando
    training/serving skew por construção.
    """
    raw = np.asarray(raw, dtype=float)
    time_s, pca, amount = raw[:, 0], raw[:, 1:29], raw[:, 29]
    hour = (time_s % SECONDS_PER_DAY) / 3600.0
    nan_h, nan_a = np.isnan(hour), np.isnan(amount)
    out = np.empty((raw.shape[0], 34), dtype=float)
    out[:, :28] = pca
    with np.errstate(invalid="ignore"):
        out[:, 28] = np.log1p(np.clip(amount, 0, None))
        out[:, 29] = np.sin(2 * np.pi * hour / 24)
        out[:, 30] = np.cos(2 * np.pi * hour / 24)
        # NaN em Time/Amount propaga como NaN (imputado depois), não como 0 silencioso
        out[:, 31] = np.where(nan_h, np.nan, (hour < 6).astype(float))
        out[:, 32] = np.where(nan_a, np.nan, (amount == 0).astype(float))
        out[:, 33] = np.where(nan_a, np.nan, ((amount > 0) & (amount <= MICRO_AMOUNT)).astype(float))
    return out


class FraudFeatureEngineer(BaseEstimator, TransformerMixin):
    """Transformações determinísticas por linha (sem estado aprendido)."""

    def fit(self, X: pd.DataFrame, y=None):
        missing = [c for c in RAW_FEATURES if c not in X.columns]
        if missing:
            raise ValueError(f"Colunas ausentes para feature engineering: {missing}")
        self.n_features_in_ = len(RAW_FEATURES)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        values = engineer_array(X[RAW_FEATURES].to_numpy(dtype=float))
        return pd.DataFrame(values, columns=self.get_feature_names_out(), index=X.index)

    def get_feature_names_out(self, input_features=None):
        return np.array([*ENGINEERED_CONTINUOUS, *ENGINEERED_FLAGS], dtype=object)


def build_preprocessor() -> Pipeline:
    """Feature engineering + imputação + escalonamento robusto.

    * ``SimpleImputer(median)``: o dataset não tem missing, mas produção terá
      (falha de upstream). Mediana é robusta a caudas pesadas. Defesa em
      profundidade: a API já rejeita nulos; o batch scoring não.
    * ``RobustScaler`` (mediana/IQR): ``Amount`` e as componentes têm caudas
      pesadas; ``StandardScaler`` seria dominado pelos outliers, justamente
      onde as fraudes vivem. Árvores são invariantes a escala, mas o baseline
      linear e o monitoramento de drift se beneficiam.
    """
    continuous = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", RobustScaler())])
    flags = SimpleImputer(strategy="most_frequent")
    columns = ColumnTransformer(
        [("cont", continuous, ENGINEERED_CONTINUOUS), ("flags", flags, ENGINEERED_FLAGS)],
        verbose_feature_names_out=False,
    )
    pre = Pipeline([("engineer", FraudFeatureEngineer()), ("columns", columns)])
    pre.set_output(transform="pandas")
    return pre
