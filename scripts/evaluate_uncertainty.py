"""Intervalos de confiança (block bootstrap) das métricas de teste.

Uso: python scripts/evaluate_uncertainty.py [--n-boot 1000] [--block-size 500]
Saída: reports/uncertainty.json, reports/UNCERTAINTY.md, reports/figures/uncertainty_*.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fraudguard.config import get_settings
from fraudguard.modeling.evaluation import load_eval_set
from fraudguard.modeling.metrics import optimize_threshold
from fraudguard.modeling.uncertainty import block_bootstrap_indices, bootstrap_metrics

ROOT = Path(__file__).resolve().parents[1]
LABELS = {
    "auprc": "AUPRC",
    "recall": "Recall",
    "precision": "Precisão",
    "false_positive_rate": "Taxa de falso positivo",
    "savings_eur": "Economia (EUR)",
    "savings_pct": "Redução de custo (%)",
}


def br(x: float, dec: int = 3) -> str:
    return f"{x:,.{dec}f}".replace(",", "\x00").replace(".", ",").replace("\x00", ".")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--block-size", type=int, default=500)
    ap.add_argument("--no-plots", action="store_true")
    a = ap.parse_args()
    s = get_settings()
    ev = load_eval_set(s)
    thr = ev.predictor.threshold
    ci = bootstrap_metrics(ev.y, ev.proba, ev.amount, thr, s.cost, n_boot=a.n_boot, block_size=a.block_size)
    iid = bootstrap_metrics(ev.y, ev.proba, ev.amount, thr, s.cost, n_boot=a.n_boot, block_size=1, seed=1)

    # Sensibilidade do custo ao limiar, com banda de 95%
    grid = np.geomspace(0.005, 0.9, 60)
    rng = np.random.default_rng(2)
    curves = []
    for _ in range(300):
        idx = block_bootstrap_indices(len(ev.y), a.block_size, rng)
        _, _, (_, c) = optimize_threshold(ev.y[idx], ev.proba[idx], ev.amount[idx], s.cost, grid=grid)
        curves.append(c)
    curves = np.array(curves)
    _, _, (_, point_curve) = optimize_threshold(ev.y, ev.proba, ev.amount, s.cost, grid=grid)

    out = {
        "model_version": ev.predictor.version,
        "threshold": thr,
        "n_test": len(ev.y),
        "frauds_test": int(ev.y.sum()),
        "method": f"block bootstrap temporal, blocos de {a.block_size} transações, {a.n_boot} réplicas, IC 95% percentílico",
        "block_bootstrap": {k: v.to_dict() for k, v in ci.items()},
        "iid_bootstrap": {k: v.to_dict() for k, v in iid.items()},
        "threshold_sensitivity": {
            "grid": grid.round(5).tolist(),
            "cost_point": point_curve.round(2).tolist(),
            "cost_low": np.quantile(curves, 0.025, axis=0).round(2).tolist(),
            "cost_high": np.quantile(curves, 0.975, axis=0).round(2).tolist(),
        },
    }
    (ROOT / "reports" / "uncertainty.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))

    lines = [
        "# Incerteza das métricas de teste",
        "",
        f"Modelo `{ev.predictor.version}`; limiar {br(thr, 4)}; {br(len(ev.y), 0)} transações e {int(ev.y.sum())} fraudes no teste.",
        f"Método: {out['method']}. Ver `docs/sdr/SDR-010-incerteza.md`.",
        "",
        "| Métrica | Pontual | IC 95% (block bootstrap) | IC 95% (i.i.d., para comparação) |",
        "|---|---|---|---|",
    ]
    for k, lab in LABELS.items():
        d = 0 if k == "savings_eur" else (1 if k == "savings_pct" else (5 if k == "false_positive_rate" else 3))
        b, i = ci[k], iid[k]
        lines.append(f"| {lab} | {br(b.point, d)} | {br(b.low, d)} a {br(b.high, d)} | {br(i.low, d)} a {br(i.high, d)} |")
    wb = ci["auprc"].high - ci["auprc"].low
    wi = iid["auprc"].high - iid["auprc"].low
    lines += [
        "",
        "**Leitura.**",
        (
            f"- O intervalo da AUPRC tem largura {br(wb, 3)} no block bootstrap contra {br(wi, 3)} no i.i.d.: "
            "ignorar a dependência temporal faria o resultado parecer mais preciso do que é."
        )
        if wb > 1.1 * wi
        else (
            f"- As larguras da AUPRC são próximas ({br(wb, 3)} com blocos, {br(wi, 3)} i.i.d.): nestes dados as fraudes "
            "não se concentram em rajadas. Em dados reais, onde ataques vêm em ondas, o block bootstrap tende a "
            "alargar o intervalo; por isso ele é o método padrão."
        ),
        f"- Mesmo no limite inferior, a redução de custo é de {br(ci['savings_pct'].low, 1)}%: "
        "o ganho do modelo é robusto, a magnitude exata é que é incerta.",
        "- Comunicar sempre a faixa, não só o ponto.",
    ]
    (ROOT / "reports" / "UNCERTAINTY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    if not a.no_plots:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig_dir = ROOT / "reports" / "figures"
        fig, ax = plt.subplots(figsize=(6.4, 3.4))
        keys = ["auprc", "recall", "precision"]
        for j, k in enumerate(keys):
            b = ci[k]
            ax.errorbar(b.point, j, xerr=[[b.point - b.low], [b.high - b.point]], fmt="o", color="#1F4E79", capsize=6, lw=2)
            ax.text(b.high + 0.01, j, f"{br(b.point, 3)}  [{br(b.low, 3)}; {br(b.high, 3)}]", va="center", fontsize=9)
        ax.set(
            yticks=range(len(keys)),
            yticklabels=[LABELS[k] for k in keys],
            xlim=(0.3, 1.15),
            title="Métricas de teste com IC 95% (block bootstrap)",
        )
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_dir / "uncertainty_intervals.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6.4, 3.6))
        ax.fill_between(
            grid,
            out["threshold_sensitivity"]["cost_low"],
            out["threshold_sensitivity"]["cost_high"],
            color="#B3261E",
            alpha=0.15,
            label="IC 95%",
        )
        ax.plot(grid, point_curve, color="#B3261E", lw=2, label="custo no teste")
        ax.axvline(thr, ls="--", color="#1F4E79", label=f"limiar escolhido na validação ({br(thr, 3)})")
        ax.set(xscale="log", xlabel="limiar de decisão", ylabel="custo total (EUR)", title="Custo × limiar no teste")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_dir / "uncertainty_threshold.png", dpi=150)
        plt.close(fig)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
