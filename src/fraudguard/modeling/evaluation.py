"""Reconstrução determinística da partição de teste para avaliações offline.

Reaplica exatamente os passos do treino (carga -> deduplicação -> split
temporal) e confere o fingerprint dos dados contra o metadata do modelo,
garantindo que a avaliação usa o MESMO teste em que o modelo foi medido.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from fraudguard.config import TARGET, Settings, get_settings
from fraudguard.data.loader import deduplicate, load_dataset, temporal_split
from fraudguard.modeling.predictor import FraudPredictor
from fraudguard.modeling.train import SEED, _data_fingerprint


@dataclass
class EvalSet:
    test: pd.DataFrame
    valid: pd.DataFrame
    y: np.ndarray
    amount: np.ndarray
    proba: np.ndarray
    predictor: FraudPredictor
    metadata: dict


def load_eval_set(
    settings: Settings | None = None, data_path: Path | None = None, df: pd.DataFrame | None = None, model_path: Path | None = None
) -> EvalSet:
    settings = settings or get_settings()
    meta = json.loads(settings.metadata_path.read_text())
    if df is None:
        df, _ = load_dataset(data_path or settings.raw_data_path, seed=SEED)
    fp = _data_fingerprint(df)
    if fp != meta["data"]["fingerprint"]:
        raise ValueError(f"dados diferentes dos usados no treino (fingerprint {fp} != {meta['data']['fingerprint']})")
    df, _ = deduplicate(df)
    split = temporal_split(df)
    predictor = FraudPredictor.load(model_path or settings.model_path)
    return EvalSet(
        test=split.test,
        valid=split.valid,
        y=split.test[TARGET].to_numpy(),
        amount=split.test["Amount"].to_numpy(),
        proba=predictor.predict_proba(split.test),
        predictor=predictor,
        metadata=meta,
    )
