"""Métricas técnicas e de negócio.

Métrica primária: AUPRC (Average Precision). Com 0,17% de positivos, a ROC-AUC
é dominada pelos verdadeiros negativos e fica otimista (Saito & Rehmsmeier,
2015). A AUPRC mede diretamente o trade-off que o negócio sente: quantas
fraudes pegamos (recall) versus quantos clientes bons incomodamos (precision).

Métrica de decisão: CUSTO ESPERADO em EUR, com custo dependente do valor
da transação (Bahnsen et al., 2014). O limiar operacional é escolhido para
minimizar esse custo — não para maximizar F1, que assume erros simétricos.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
)

from fraudguard.config import CostModel


@dataclass
class CostBreakdown:
    total_cost: float
    baseline_cost: float  # sem modelo: toda fraude passa
    savings: float
    savings_pct: float
    fraud_loss_prevented: float
    fraud_loss_missed: float
    false_positive_cost: float
    review_cost: float
    n_flagged: int

    def to_dict(self) -> dict:
        return {k: round(v, 2) if isinstance(v, float) else v for k, v in asdict(self).items()}


def business_cost(y_true, y_pred, amount, cost: CostModel) -> CostBreakdown:
    y_true = np.asarray(y_true).astype(bool)
    y_pred = np.asarray(y_pred).astype(bool)
    amount = np.asarray(amount, dtype=float)
    fn_cost = cost.false_negative_cost(amount)

    tp, fp, fn = y_true & y_pred, ~y_true & y_pred, y_true & ~y_pred
    missed = float(fn_cost[fn].sum())
    fp_total = float(fp.sum() * cost.false_positive_cost)
    review = float(tp.sum() * cost.true_positive_cost)
    total = missed + fp_total + review
    baseline = float(fn_cost[y_true].sum())
    savings = baseline - total
    return CostBreakdown(
        total_cost=total,
        baseline_cost=baseline,
        savings=savings,
        savings_pct=100.0 * savings / baseline if baseline else 0.0,
        fraud_loss_prevented=float(amount[tp].sum()),
        fraud_loss_missed=float(amount[fn].sum()),
        false_positive_cost=fp_total,
        review_cost=review,
        n_flagged=int(y_pred.sum()),
    )


def optimize_threshold(y_true, proba, amount, cost: CostModel, grid: np.ndarray | None = None):
    """Limiar que minimiza o custo esperado total. Retorna (limiar, custo, curva)."""
    grid = np.unique(np.concatenate([np.linspace(0.005, 0.995, 199), np.asarray(proba)])) if grid is None else grid
    grid = grid[(grid > 0) & (grid < 1)]
    y_true = np.asarray(y_true).astype(bool)
    proba = np.asarray(proba, dtype=float)
    amount = np.asarray(amount, dtype=float)
    fn_cost = cost.false_negative_cost(amount)

    # Vetorizado via ordenação: custo(t) para todos os t em O(n log n)
    order = np.argsort(-proba)
    p_sorted, y_sorted, fn_sorted = proba[order], y_true[order], fn_cost[order]
    cum_tp = np.cumsum(y_sorted)
    cum_fp = np.cumsum(~y_sorted)
    cum_fn_avoided = np.cumsum(np.where(y_sorted, fn_sorted, 0.0))
    total_fn = fn_sorted[y_sorted].sum()

    # nº de itens com proba >= t
    k = np.searchsorted(-p_sorted, -grid, side="right")
    idx = np.clip(k - 1, 0, None)
    tp = np.where(k > 0, cum_tp[idx], 0)
    fp = np.where(k > 0, cum_fp[idx], 0)
    avoided = np.where(k > 0, cum_fn_avoided[idx], 0.0)
    costs = (total_fn - avoided) + fp * cost.false_positive_cost + tp * cost.true_positive_cost
    best = int(np.argmin(costs))
    return float(grid[best]), float(costs[best]), (grid, costs)


def classification_report_at(y_true, proba, threshold: float) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(proba) >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    f2 = 5 * precision * recall / (4 * precision + recall) if precision + recall else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return {
        "threshold": round(float(threshold), 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "f2": round(f2, 4),
        "false_positive_rate": round(fpr, 6),
        "alert_rate": round((tp + fp) / len(y_true), 6),
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }


def recall_at_precision(y_true, proba, min_precision: float) -> float:
    precision, recall, _ = precision_recall_curve(y_true, proba)
    ok = precision >= min_precision
    return float(recall[ok].max()) if ok.any() else 0.0


def ranking_metrics(y_true, proba) -> dict:
    return {
        "auprc": round(float(average_precision_score(y_true, proba)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, proba)), 4),
        "brier": round(float(brier_score_loss(y_true, proba)), 6),
        "recall_at_precision_80": round(recall_at_precision(y_true, proba, 0.80), 4),
        "recall_at_precision_90": round(recall_at_precision(y_true, proba, 0.90), 4),
    }
