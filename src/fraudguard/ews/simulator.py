"""Gerador de cenários para demonstrar o EWS a stakeholders (modo demo).

Permite mostrar, ao vivo, o sistema reagindo a: tráfego normal, ataque de
teste de cartão, drift de dados e tentativas de alto valor. Desligado em
produção via ``FRAUDGUARD_DEMO_MODE=false``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fraudguard.config import TARGET
from fraudguard.data.synthetic import generate_synthetic_creditcard

_POOL: pd.DataFrame | None = None


def _pool() -> pd.DataFrame:
    global _POOL
    if _POOL is None:
        _POOL = generate_synthetic_creditcard(n_samples=40_000, n_frauds=400, duplicate_rate=0, seed=7)
    return _POOL


def scenario_frame(scenario: str, n: int, seed: int | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pool = _pool()
    legit, fraud = pool[pool[TARGET] == 0], pool[pool[TARGET] == 1]

    if scenario == "normal":
        # mistura realista: ~0,17% fraude
        n_f = rng.binomial(n, 0.0017)
        df = pd.concat([legit.sample(n - n_f, random_state=seed, replace=True), fraud.sample(n_f, random_state=seed, replace=True)])
    elif scenario == "card_testing":
        n_f = max(int(n * 0.3), 12)
        attack = fraud.sample(n_f, random_state=seed, replace=True).copy()
        attack["Amount"] = np.round(rng.uniform(0.5, 1.99, n_f), 2)
        df = pd.concat([legit.sample(n - n_f, random_state=seed, replace=True), attack])
    elif scenario == "drift":
        df = legit.sample(n, random_state=seed, replace=True).copy()
        df["V14"] = df["V14"] - 2.5
        df["V12"] = df["V12"] - 2.0
        df["Amount"] = df["Amount"] * 4
    elif scenario == "high_value":
        n_f = max(int(n * 0.05), 3)
        big = fraud.sample(n_f, random_state=seed, replace=True).copy()
        big["Amount"] = np.round(rng.uniform(1500, 5000, n_f), 2)
        df = pd.concat([legit.sample(n - n_f, random_state=seed, replace=True), big])
    else:
        raise ValueError(f"cenário desconhecido: {scenario}")
    return df.sample(frac=1, random_state=seed).drop(columns=[TARGET]).reset_index(drop=True)
