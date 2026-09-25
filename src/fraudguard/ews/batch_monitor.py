"""Camada 3 do monitoramento: desempenho real quando os rótulos chegam.

Problemas que este módulo trata explicitamente (docs/MONITORING.md):

1. **Atraso de rótulos.** A fraude só é confirmada no chargeback. Uma coorte
   recente parece "sem fraude" apenas porque os rótulos ainda não chegaram.
   Métricas só são calculadas para coortes MADURAS (idade ≥ janela de
   maturação) e a cobertura de rótulos é reportada junto.
2. **Viés de feedback.** Transações sinalizadas são revisadas e rotuladas
   rápido; aprovadas só ganham rótulo se o cliente contestar. O simulador
   reproduz essa assimetria para que o relatório mostre o efeito.
3. **Calibração em produção.** ECE (Expected Calibration Error) por coorte:
   a decisão por custo depende de probabilidades confiáveis.
4. **Coortes pela data da TRANSAÇÃO**, nunca pela data de chegada do rótulo.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from fraudguard.config import CostModel


def expected_calibration_error(y, proba, n_bins: int = 10) -> float:
    """ECE com bins por quantil (bins iguais seriam quase todos vazios com p ≈ 0)."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(proba, dtype=float)
    if p.size == 0:
        return float("nan")
    edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    bins = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, max(len(edges) - 2, 0))
    ece = 0.0
    for b in np.unique(bins):
        m = bins == b
        ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


def simulate_label_arrival(
    decisions: pd.DataFrame, rng: np.random.Generator, review_delay_h: float = 1.0, chargeback_delay_h: float = 72.0
) -> pd.Series:
    """Hora de chegada do rótulo. Sinalizadas: revisão rápida. Fraudes aprovadas:
    chargeback lento (exponencial). Legítimas aprovadas: nunca recebem rótulo
    explícito; assumidas legítimas após a maturação."""
    n = len(decisions)
    arrival = np.full(n, np.inf)
    flagged = decisions["flagged"].to_numpy()
    fraud = decisions["label"].to_numpy().astype(bool)
    arrival[flagged] = decisions["ts_h"].to_numpy()[flagged] + rng.exponential(review_delay_h, flagged.sum())
    missed = fraud & ~flagged
    arrival[missed] = decisions["ts_h"].to_numpy()[missed] + rng.exponential(chargeback_delay_h, missed.sum())
    return pd.Series(arrival, index=decisions.index)


@dataclass(frozen=True)
class BatchMonitorConfig:
    maturity_h: float = 96.0
    min_frauds: int = 5
    auprc_tolerance: float = 0.10  # alerta se cair >10% contra a referência
    ece_limit: float = 0.002


def cohort_report(
    decisions: pd.DataFrame, as_of_h: float, cost: CostModel, reference_auprc: float, cfg: BatchMonitorConfig | None = None
) -> pd.DataFrame:
    """Uma linha por coorte (``cohort``) com métricas visíveis na data ``as_of_h``.

    Colunas esperadas: cohort, ts_h, probability, flagged, amount, label, label_arrival_h.
    """
    cfg = cfg or BatchMonitorConfig()
    rows = []
    for cohort, g in decisions.groupby("cohort", sort=True):
        age = as_of_h - g["ts_h"].max()
        known = g["label_arrival_h"] <= as_of_h
        mature = age >= cfg.maturity_h
        # Visão "como o sistema vê": fraude só conta se o rótulo já chegou
        y_seen = (g["label"].astype(bool) & known).to_numpy()
        frauds_true, frauds_seen = int(g["label"].sum()), int(y_seen.sum())
        row = {
            "cohort": cohort,
            "transactions": len(g),
            "age_h": round(float(age), 1),
            "mature": bool(mature),
            "frauds_true": frauds_true,
            "frauds_labeled": frauds_seen,
            "label_coverage": round(frauds_seen / frauds_true, 3) if frauds_true else 1.0,
        }
        if mature and frauds_seen >= cfg.min_frauds:
            p = g["probability"].to_numpy()
            pred = g["flagged"].to_numpy()
            amt = g["amount"].to_numpy()
            fn_cost = cost.false_negative_cost(amt)
            realized = (
                fn_cost[y_seen & ~pred].sum()
                + (pred & ~y_seen).sum() * cost.false_positive_cost
                + (pred & y_seen).sum() * cost.true_positive_cost
            )
            auprc = float(average_precision_score(y_seen, p))
            row.update(
                {
                    "auprc": round(auprc, 4),
                    "recall": round(float((pred & y_seen).sum() / frauds_seen), 4),
                    "ece": round(expected_calibration_error(y_seen, p), 6),
                    "realized_cost_eur": round(float(realized), 2),
                    "status": "DEGRADADO" if auprc < reference_auprc * (1 - cfg.auprc_tolerance) else "OK",
                }
            )
            if row["status"] == "OK" and row["ece"] > cfg.ece_limit:
                row["status"] = "DESCALIBRADO"
        else:
            row["status"] = "IMATURA" if not mature else "POUCOS_ROTULOS"
        rows.append(row)
    return pd.DataFrame(rows)
