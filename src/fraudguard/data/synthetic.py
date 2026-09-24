"""Gerador de dados sintéticos com o mesmo schema e perfil estatístico do
dataset ULB (Dal Pozzolo et al., 2015).

Por que não uma simples ``make_classification``?
Porque ela produz classes quase linearmente separáveis e sem as propriedades
que tornam o problema real difícil. Este gerador reproduz deliberadamente:

* 284.807 transações em ~48h com 492 fraudes (0,172%);
* V1..V28 descorrelacionadas (saída de PCA) com variância decrescente e
  caudas pesadas (t-Student), como no dataset original;
* fraudes com deslocamento nas componentes sabidamente discriminativas do
  dataset real (V14, V17, V12, V10, V4, V11, V3...), porém como MISTURA:
  parte das fraudes é "camuflada" (deslocamento pequeno) — isso impede
  separabilidade trivial e produz AUPRC realista;
* sazonalidade diária: volume legítimo cai de madrugada, fraude não;
* ``Amount`` log-normal para legítimas e bimodal para fraudes (teste de
  cartão com micro-valores + saques de alto valor);
* duplicatas exatas (~0,38% no original: 1.081 linhas) para exercitar a
  deduplicação.

Premissa honesta: métricas obtidas em dados sintéticos NÃO substituem a
validação no CSV real. O pipeline usa o CSV real automaticamente se existir.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fraudguard.config import PCA_FEATURES, TARGET

# Deslocamento médio aproximado das fraudes por componente, inspirado nas
# médias condicionais publicamente conhecidas do dataset ULB.
_FRAUD_SHIFT = {
    "V1": -4.8,
    "V2": 3.6,
    "V3": -7.0,
    "V4": 4.5,
    "V5": -3.2,
    "V6": -1.4,
    "V7": -5.6,
    "V9": -2.6,
    "V10": -5.7,
    "V11": 3.8,
    "V12": -6.3,
    "V14": -7.0,
    "V16": -4.1,
    "V17": -6.7,
    "V18": -2.2,
    "V19": 0.7,
    "V21": 0.7,
    "V27": 0.2,
}


def generate_synthetic_creditcard(
    n_samples: int = 284_807,
    n_frauds: int = 492,
    duplicate_rate: float = 0.0038,
    missing_rate: float = 0.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Gera um DataFrame com colunas Time, V1..V28, Amount, Class."""
    if n_frauds >= n_samples:
        raise ValueError("n_frauds deve ser menor que n_samples")
    rng = np.random.default_rng(seed)
    n_legit = n_samples - n_frauds

    # Variância decrescente como autovalores de PCA (V1 ~ 3.8 ... V28 ~ 0.1)
    stds = np.sqrt(np.linspace(3.8, 0.11, len(PCA_FEATURES)))

    # t-Student(df=5) normalizada -> caudas pesadas
    def heavy(n: int) -> np.ndarray:
        t = rng.standard_t(df=5, size=(n, len(PCA_FEATURES)))
        return t / np.sqrt(5 / 3) * stds

    X_legit = heavy(n_legit)

    # Fraudes: mistura de perfis "evidente" (65%), "moderado" (20%), "camuflado" (15%)
    profile = rng.choice([1.0, 0.45, 0.12], size=n_frauds, p=[0.65, 0.20, 0.15])
    X_fraud = heavy(n_frauds) * 1.6  # fraudes são mais dispersas
    shift = np.zeros(len(PCA_FEATURES))
    for name, value in _FRAUD_SHIFT.items():
        shift[PCA_FEATURES.index(name)] = value
    X_fraud += profile[:, None] * shift[None, :] * 0.55

    # Tempo: 2 dias. Legítimas seguem ciclo diário; fraudes quase uniformes.
    def daily_times(n: int, amplitude: float) -> np.ndarray:
        out = np.empty(0)
        while out.size < n:
            cand = rng.uniform(0, 172_792, size=n * 2)
            hour = (cand % 86_400) / 3_600
            # pico ~14h, vale ~4h
            intensity = 1 + amplitude * np.sin(2 * np.pi * (hour - 8) / 24)
            keep = rng.uniform(0, 1 + amplitude, size=cand.size) < intensity
            out = np.concatenate([out, cand[keep]])
        return np.floor(out[:n])

    t_legit = daily_times(n_legit, amplitude=0.85)
    t_fraud = daily_times(n_frauds, amplitude=0.15)

    # Amount
    amt_legit = rng.lognormal(mean=3.0, sigma=1.35, size=n_legit)
    small = rng.random(n_frauds) < 0.5
    amt_fraud = np.where(
        small,
        rng.uniform(0.0, 5.0, size=n_frauds),  # teste de cartão
        rng.lognormal(mean=4.6, sigma=1.1, size=n_frauds),
    )
    # ~0.6% de transações com Amount = 0 (verificação de cartão), como no real
    amt_legit[rng.random(n_legit) < 0.006] = 0.0
    amt_legit = np.clip(amt_legit, 0, 25_691.16)  # máximo do dataset real
    amt_fraud = np.clip(amt_fraud, 0, 2_125.87)

    # Acoplamento leve Amount <-> V7/V20 (existe no dataset real)
    X_legit[:, PCA_FEATURES.index("V7")] += 0.25 * (np.log1p(amt_legit) - 3.0)
    X_legit[:, PCA_FEATURES.index("V20")] += 0.20 * (np.log1p(amt_legit) - 3.0)

    df = pd.DataFrame(np.vstack([X_legit, X_fraud]), columns=PCA_FEATURES)
    df.insert(0, "Time", np.concatenate([t_legit, t_fraud]))
    df["Amount"] = np.round(np.concatenate([amt_legit, amt_fraud]), 2)
    df[TARGET] = np.concatenate([np.zeros(n_legit, int), np.ones(n_frauds, int)])

    # Duplicatas exatas (substituem linhas legítimas para manter n_samples)
    n_dup = int(n_samples * duplicate_rate)
    if n_dup:
        src = rng.choice(n_legit, size=n_dup, replace=False)
        dst = rng.choice(np.setdiff1d(np.arange(n_legit), src), size=n_dup, replace=False)
        df.iloc[dst] = df.iloc[src].to_numpy()

    if missing_rate > 0:
        mask = rng.random((len(df), len(PCA_FEATURES))) < missing_rate
        df[PCA_FEATURES] = df[PCA_FEATURES].mask(mask)

    df = df.sort_values("Time", kind="stable").reset_index(drop=True)
    df[TARGET] = df[TARGET].astype(int)
    return df
