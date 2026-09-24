"""Carga, validação e particionamento dos dados.

Decisões (ver docs/sdr/SDR-002-split-temporal.md):
* Deduplicação ANTES do split: duplicatas exatas em treino e teste inflam
  as métricas (vazamento por memorização).
* Split TEMPORAL (passado -> futuro), não aleatório: em produção o modelo
  sempre prevê o futuro. Split aleatório vaza padrões de ataques que
  acontecem em rajadas no tempo (Le Borgne et al., 2022, cap. 5).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from fraudguard.config import RAW_FEATURES, TARGET
from fraudguard.data.synthetic import generate_synthetic_creditcard

logger = logging.getLogger(__name__)


class SchemaError(ValueError):
    """Dados não respeitam o contrato esperado."""


def validate_schema(df: pd.DataFrame, require_target: bool = True) -> None:
    expected = RAW_FEATURES + ([TARGET] if require_target else [])
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise SchemaError(f"Colunas ausentes: {missing}")
    non_numeric = [c for c in RAW_FEATURES if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise SchemaError(f"Colunas não numéricas: {non_numeric}")
    if (df["Amount"].dropna() < 0).any():
        raise SchemaError("Amount negativo encontrado")
    if require_target and not set(df[TARGET].dropna().unique()) <= {0, 1}:
        raise SchemaError("Target deve ser binário {0,1}")


def load_dataset(path: Path | None = None, synthetic_if_missing: bool = True, seed: int = 42):
    """Retorna (df, fonte). Usa o CSV real do Kaggle se disponível."""
    if path is not None and Path(path).exists():
        df = pd.read_csv(path)
        source = f"kaggle:{Path(path).name}"
    elif synthetic_if_missing:
        logger.warning("CSV real não encontrado; usando dados sintéticos equivalentes")
        df = generate_synthetic_creditcard(seed=seed)
        source = "synthetic:ulb-profile"
    else:
        raise FileNotFoundError(path)
    validate_schema(df)
    return df, source


def deduplicate(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    before = len(df)
    out = df.drop_duplicates().reset_index(drop=True)
    return out, before - len(out)


@dataclass(frozen=True)
class TemporalSplit:
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame


def temporal_split(df: pd.DataFrame, valid_frac: float = 0.2, test_frac: float = 0.2) -> TemporalSplit:
    """Split cronológico por ``Time``: treino | validação | teste."""
    if not 0 < valid_frac < 1 or not 0 < test_frac < 1 or valid_frac + test_frac >= 1:
        raise ValueError("Frações inválidas")
    ordered = df.sort_values("Time", kind="stable").reset_index(drop=True)
    n = len(ordered)
    i_valid = int(np.floor(n * (1 - valid_frac - test_frac)))
    i_test = int(np.floor(n * (1 - test_frac)))
    split = TemporalSplit(
        train=ordered.iloc[:i_valid].reset_index(drop=True),
        valid=ordered.iloc[i_valid:i_test].reset_index(drop=True),
        test=ordered.iloc[i_test:].reset_index(drop=True),
    )
    for name, part in (("train", split.train), ("valid", split.valid), ("test", split.test)):
        if part[TARGET].sum() == 0:
            raise ValueError(f"Partição '{name}' ficou sem fraudes; ajuste as frações")
    return split
