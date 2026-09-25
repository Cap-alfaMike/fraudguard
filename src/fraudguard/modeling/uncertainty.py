"""Quantificação de incerteza das métricas (docs/sdr/SDR-010-incerteza.md).

Com ~94 fraudes no teste, uma métrica pontual ("AUPRC = 0,774") sugere uma
precisão que os dados não sustentam. Este módulo produz intervalos de
confiança por *block bootstrap* temporal:

* Fraudes ocorrem em rajadas (o mesmo fraudador em minutos). O bootstrap
  i.i.d. trata eventos correlacionados como independentes e ESTREITA os
  intervalos artificialmente. Reamostrar blocos contíguos no tempo preserva
  essa dependência (Künsch, 1989; Politis & Romano, 1994).
* O bootstrap PAREADO compara dois modelos nas MESMAS reamostragens, o que
  cancela a variância comum aos dois e é a base do portão de promoção
  champion/challenger (registry.py).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import average_precision_score

from fraudguard.config import CostModel

METRICS = ("auprc", "recall", "precision", "false_positive_rate", "savings_eur", "savings_pct")


@dataclass(frozen=True)
class Interval:
    point: float
    low: float
    high: float
    level: float

    def to_dict(self) -> dict:
        return {"point": round(self.point, 6), "low": round(self.low, 6), "high": round(self.high, 6), "level": self.level}


def block_bootstrap_indices(n: int, block_size: int, rng: np.random.Generator) -> np.ndarray:
    """Índices de uma réplica: blocos contíguos sorteados com reposição até cobrir n."""
    if n <= 0:
        raise ValueError("n deve ser positivo")
    block_size = max(1, min(block_size, n))
    n_blocks = int(np.ceil(n / block_size))
    starts = rng.integers(0, n - block_size + 1, size=n_blocks)
    idx = (starts[:, None] + np.arange(block_size)[None, :]).ravel()
    return idx[:n]


def point_metrics(y, proba, amount, threshold: float, cost: CostModel) -> dict:
    y = np.asarray(y).astype(bool)
    proba = np.asarray(proba, dtype=float)
    amount = np.asarray(amount, dtype=float)
    pred = proba >= threshold
    tp, fp = int((pred & y).sum()), int((pred & ~y).sum())
    fn, tn = int((~pred & y).sum()), int((~pred & ~y).sum())
    fn_cost = cost.false_negative_cost(amount)
    total = fn_cost[~pred & y].sum() + fp * cost.false_positive_cost + tp * cost.true_positive_cost
    baseline = fn_cost[y].sum()
    return {
        "auprc": float(average_precision_score(y, proba)) if y.any() else np.nan,
        "recall": tp / (tp + fn) if tp + fn else np.nan,
        "precision": tp / (tp + fp) if tp + fp else np.nan,
        "false_positive_rate": fp / (fp + tn) if fp + tn else np.nan,
        "savings_eur": float(baseline - total),
        "savings_pct": float(100 * (baseline - total) / baseline) if baseline else np.nan,
    }


def _interval(point: float, samples: np.ndarray, level: float) -> Interval:
    s = samples[~np.isnan(samples)]
    alpha = (1 - level) / 2
    lo, hi = np.quantile(s, [alpha, 1 - alpha]) if s.size else (np.nan, np.nan)
    return Interval(float(point), float(lo), float(hi), level)


def bootstrap_metrics(
    y,
    proba,
    amount,
    threshold: float,
    cost: CostModel,
    n_boot: int = 1000,
    block_size: int = 500,
    level: float = 0.95,
    seed: int = 0,
) -> dict[str, Interval]:
    """IC percentílico por block bootstrap. Dados devem estar em ordem temporal."""
    y, proba, amount = np.asarray(y), np.asarray(proba, dtype=float), np.asarray(amount, dtype=float)
    rng = np.random.default_rng(seed)
    point = point_metrics(y, proba, amount, threshold, cost)
    samples = {m: np.empty(n_boot) for m in METRICS}
    for b in range(n_boot):
        idx = block_bootstrap_indices(len(y), block_size, rng)
        rep = point_metrics(y[idx], proba[idx], amount[idx], threshold, cost)
        for m in METRICS:
            samples[m][b] = rep[m]
    return {m: _interval(point[m], samples[m], level) for m in METRICS}


@dataclass(frozen=True)
class PairedComparison:
    delta_auprc: Interval  # challenger - champion (maior é melhor)
    delta_cost_eur: Interval  # challenger - champion (menor é melhor)
    delta_fpr: Interval  # challenger - champion (menor é melhor)
    prob_auprc_better: float
    prob_cost_lower: float

    def to_dict(self) -> dict:
        return {
            "delta_auprc": self.delta_auprc.to_dict(),
            "delta_cost_eur": self.delta_cost_eur.to_dict(),
            "delta_fpr": self.delta_fpr.to_dict(),
            "prob_auprc_better": round(self.prob_auprc_better, 4),
            "prob_cost_lower": round(self.prob_cost_lower, 4),
        }


def paired_bootstrap(
    y,
    p_champion,
    p_challenger,
    amount,
    thr_champion: float,
    thr_challenger: float,
    cost: CostModel,
    n_boot: int = 1000,
    block_size: int = 500,
    level: float = 0.95,
    seed: int = 0,
) -> PairedComparison:
    y = np.asarray(y)
    pa, pb = np.asarray(p_champion, dtype=float), np.asarray(p_challenger, dtype=float)
    amount = np.asarray(amount, dtype=float)
    rng = np.random.default_rng(seed)

    def deltas(idx):
        a = point_metrics(y[idx], pa[idx], amount[idx], thr_champion, cost)
        b = point_metrics(y[idx], pb[idx], amount[idx], thr_challenger, cost)
        # custo = baseline - savings; baseline é igual nos dois -> Δcusto = -(Δsavings)
        return (b["auprc"] - a["auprc"], -(b["savings_eur"] - a["savings_eur"]), b["false_positive_rate"] - a["false_positive_rate"])

    full = deltas(np.arange(len(y)))
    reps = np.array([deltas(block_bootstrap_indices(len(y), block_size, rng)) for _ in range(n_boot)])
    valid = ~np.isnan(reps[:, 0])
    return PairedComparison(
        delta_auprc=_interval(full[0], reps[:, 0], level),
        delta_cost_eur=_interval(full[1], reps[:, 1], level),
        delta_fpr=_interval(full[2], reps[:, 2], level),
        prob_auprc_better=float(np.mean(reps[valid, 0] > 0)),
        prob_cost_lower=float(np.mean(reps[:, 1] < 0)),
    )
