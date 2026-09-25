"""Figuras em qualidade de apresentação, geradas dos dados reais do pipeline.

Mesma paleta dos slides e do dashboard; texto grande para leitura à distância.
Uso: python scripts/make_presentation_figures.py [--font caminho/Archivo.ttf]
Saída: reports/figures/presentation/*.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager as fm
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve

from fraudguard.config import PCA_FEATURES, TARGET, get_settings
from fraudguard.data.loader import load_dataset
from fraudguard.modeling.evaluation import load_eval_set

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "figures" / "presentation"
BG, INK, SLATE, RULE = "#F2F4F6", "#152231", "#5A6878", "#D5DCE3"
BLUE, ALARM, SAFE = "#1F4E79", "#C4452B", "#1E7A5A"


def style(font: Path | None) -> None:
    family = "DejaVu Sans"
    if font and font.exists():
        fm.fontManager.addfont(str(font))
        family = fm.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update(
        {
            "font.family": family,
            "font.size": 20,
            "axes.titlesize": 22,
            "axes.labelsize": 20,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "legend.fontsize": 17,
            "figure.facecolor": BG,
            "axes.facecolor": BG,
            "savefig.facecolor": BG,
            "axes.edgecolor": RULE,
            "axes.labelcolor": INK,
            "xtick.color": SLATE,
            "ytick.color": SLATE,
            "text.color": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": RULE,
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
        }
    )


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", dpi=200)
    plt.close(fig)


def br(x: float, d: int = 1) -> str:
    return f"{x:.{d}f}".replace(".", ",")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--font", type=Path, default=None)
    style(ap.parse_args().font)
    s = get_settings()
    df, _ = load_dataset(s.raw_data_path)
    ev = load_eval_set(s, df=df)
    fraud, legit = df[df[TARGET] == 1], df[df[TARGET] == 0]

    # 1. Valor por classe (fração de cada classe por faixa; "density" com bins log distorce)
    fig, ax = plt.subplots(figsize=(10, 6))
    bins = np.logspace(-1, 4.3, 40)
    for part, color, alpha, label in ((legit, SLATE, 0.55, "legítimas"), (fraud, ALARM, 0.75, "fraudes")):
        v = part["Amount"].clip(0.1)
        ax.hist(v, bins=bins, weights=np.full(len(v), 100 / len(v)), color=color, alpha=alpha, label=label)
    ax.axvspan(0.1, 2, color=ALARM, alpha=0.07)
    ax.text(0.12, ax.get_ylim()[1] * 0.80, "teste de\ncartão\n(≤ €2)", color=ALARM, fontsize=18)
    ax.set(xscale="log", xlabel="valor da transação (EUR, escala log)", ylabel="% das transações da classe")
    ax.legend(frameon=False, loc="upper right")
    save(fig, "eda_amount")

    # 2. Hora do dia
    hour = ((df["Time"] % 86_400) // 3_600).astype(int)
    g = df.groupby(hour)[TARGET].agg(["count", "mean"])
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(g.index, g["count"] / 1000, color=RULE, width=0.8, label="volume (mil)")
    ax.set(xlabel="hora do dia", ylabel="transações (mil)", xticks=range(0, 24, 3))
    ax2 = ax.twinx()
    ax2.plot(g.index, g["mean"] * 100, color=ALARM, lw=4, marker="o", ms=8, label="taxa de fraude")
    ax2.set_ylabel("taxa de fraude (%)", color=ALARM)
    ax2.tick_params(axis="y", colors=ALARM)
    ax2.grid(False)
    ax2.spines["right"].set_visible(True)
    save(fig, "eda_hour")

    # 3. KS
    from scipy.stats import ks_2samp

    ks = sorted(((c, ks_2samp(fraud[c], legit[c]).statistic) for c in [*PCA_FEATURES, "Amount"]), key=lambda t: -t[1])[:10][::-1]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh([k for k, _ in ks], [v for _, v in ks], color=[BLUE if v > 0.5 else SLATE for _, v in ks])
    for i, (_, v) in enumerate(ks):
        ax.text(v + 0.01, i, br(v, 2), va="center", fontsize=17, color=INK)
    ax.set(xlabel="estatística KS (fraude vs legítima)", xlim=(0, 0.8))
    ax.grid(axis="y", visible=False)
    save(fig, "eda_ks")

    # 4. Curva PR com o ponto de operação
    prec, rec, _ = precision_recall_curve(ev.y, ev.proba)
    thr = ev.predictor.threshold
    pred = ev.proba >= thr
    op_r, op_p = (pred & (ev.y == 1)).sum() / ev.y.sum(), (pred & (ev.y == 1)).sum() / pred.sum()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.fill_between(rec, prec, color=BLUE, alpha=0.08)
    ax.plot(rec, prec, color=BLUE, lw=4)
    ax.axhline(ev.y.mean(), color=SLATE, ls=":", lw=2)
    ax.text(0.02, ev.y.mean() + 0.03, "classificador aleatório (0,17%)", color=SLATE, fontsize=17)
    ax.scatter([op_r], [op_p], s=260, color=ALARM, zorder=5)
    ax.annotate(
        f"ponto de operação\nrecall {br(op_r * 100)}%, precisão {br(op_p * 100)}%",
        (op_r, op_p),
        xytext=(0.08, 0.35),
        fontsize=18,
        color=ALARM,
        arrowprops={"arrowstyle": "-", "color": ALARM},
    )
    ax.set(xlabel="recall (fraudes capturadas)", ylabel="precisão", xlim=(0, 1.02), ylim=(0, 1.05))
    save(fig, "pr_curve")

    # 5. Custo × limiar com IC
    unc = json.loads((ROOT / "reports" / "uncertainty.json").read_text())["threshold_sensitivity"]
    grid = np.array(unc["grid"])
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.fill_between(grid, unc["cost_low"], unc["cost_high"], color=ALARM, alpha=0.15, label="IC 95%")
    ax.plot(grid, unc["cost_point"], color=ALARM, lw=4, label="custo total no teste")
    ax.axvline(thr, color=BLUE, ls="--", lw=3)
    ax.text(thr * 1.08, max(unc["cost_high"]) * 0.92, f"limiar escolhido\n{br(thr, 3)}", color=BLUE, fontsize=18)
    ax.axvline(0.5, color=SLATE, ls=":", lw=2)
    ax.text(0.52, max(unc["cost_high"]) * 0.92, "limiar\ningênuo 0,5", color=SLATE, fontsize=18)
    ax.set(xscale="log", xlabel="limiar de decisão (escala log)", ylabel="custo total (EUR)")
    ax.legend(frameon=False, loc="lower left")
    save(fig, "cost_threshold")

    # 6. Calibração
    frac, mean = calibration_curve(ev.y, ev.proba, n_bins=8, strategy="quantile")
    fig, ax = plt.subplots(figsize=(8, 6))
    lim = [1e-5, 1]
    ax.plot(lim, lim, color=SLATE, ls=":", lw=2, label="calibração perfeita")
    ax.plot(mean, np.clip(frac, 1e-5, 1), color=BLUE, lw=4, marker="o", ms=10, label="modelo calibrado")
    ax.set(xscale="log", yscale="log", xlabel="probabilidade prevista", ylabel="frequência observada", xlim=lim, ylim=lim)
    ax.legend(frameon=False)
    save(fig, "calibration")

    # 7. Maturação de rótulos
    import pandas as pd

    from fraudguard.ews.batch_monitor import simulate_label_arrival

    ts_h = ev.test["Time"].to_numpy() / 3600
    dec = pd.DataFrame({"ts_h": ts_h, "flagged": ev.proba >= thr, "label": ev.y})
    dec["arr"] = simulate_label_arrival(dec, np.random.default_rng(0))
    end, frauds = ts_h.max(), dec["label"].astype(bool)
    days = np.linspace(0.02, 15, 150)
    cov = np.array([(dec.loc[frauds, "arr"] <= end + d * 24).mean() for d in days]) * 100
    app = np.array([dec.loc[frauds & (dec["arr"] <= end + d * 24), "flagged"].mean() for d in days]) * 100
    real = dec.loc[frauds, "flagged"].mean() * 100
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(days, app, color=ALARM, lw=4, label="recall aparente")
    ax.axhline(real, color=ALARM, ls=":", lw=3, label=f"recall real ({br(real)}%)")
    ax.plot(days, cov, color=BLUE, lw=4, label="fraudes já rotuladas")
    ax.axvline(4, color=SLATE, ls="--", lw=2)
    ax.text(4.2, 12, "maturação\n(96 h)", color=SLATE, fontsize=17)
    ax.set(xlabel="dias após a transação", ylabel="%", ylim=(0, 105))
    ax.legend(frameon=False, loc="center right")
    save(fig, "label_maturation")
    print(f"figuras em {OUT}")


if __name__ == "__main__":
    main()
