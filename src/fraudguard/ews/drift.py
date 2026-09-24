"""Detecção de drift via Population Stability Index (PSI).

Escolha do PSI (Siddiqi, 2006): é o padrão de fato em risco de crédito,
interpretável por times de Risco/Compliance e barato o bastante para rodar
online a cada janela. Faixas usuais: < 0,10 estável; 0,10–0,25 atenção;
> 0,25 mudança relevante. Os bins são definidos por QUANTIS da referência,
o que torna o índice robusto às caudas pesadas destas variáveis.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-4
MONITORED_FEATURES = ["Amount", "V14", "V17", "V12", "V10", "V4", "V11", "V3"]


def quantile_edges(values, n_bins: int = 10) -> list[float]:
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    edges = np.unique(np.quantile(values, np.linspace(0, 1, n_bins + 1)[1:-1]))
    return edges.tolist()


def bin_proportions(values, edges) -> list[float]:
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return [0.0] * (len(edges) + 1)
    counts = np.bincount(np.searchsorted(edges, values, side="right"), minlength=len(edges) + 1)
    return (counts / counts.sum()).tolist()


def psi(expected: list[float], actual: list[float]) -> float:
    e = np.clip(np.asarray(expected, dtype=float), EPS, None)
    a = np.clip(np.asarray(actual, dtype=float), EPS, None)
    e, a = e / e.sum(), a / a.sum()
    return float(np.sum((a - e) * np.log(a / e)))


def build_reference_profile(features, scores, threshold: float, n_bins: int = 10) -> dict:
    """Perfil estatístico do período de validação (dados 'saudáveis' mais recentes
    antes do teste) — base de comparação do Early Warning System."""
    profile = {"features": {}, "score": {}}
    for col in MONITORED_FEATURES:
        edges = quantile_edges(features[col], n_bins)
        profile["features"][col] = {"edges": edges, "proportions": bin_proportions(features[col], edges)}
    scores = np.asarray(scores, dtype=float)
    # Scores são muito concentrados perto de 0: bins fixos em escala log são mais informativos
    score_edges = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1, 0.3, 0.5, 0.8]
    profile["score"] = {
        "edges": score_edges,
        "proportions": bin_proportions(scores, score_edges),
        "mean": float(scores.mean()),
        "suspicious_rate": float((scores >= threshold).mean()),
    }
    return profile
