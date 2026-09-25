"""Demonstra a camada 3 do monitoramento sobre a partição de teste.

Trata o teste como "produção", simula a chegada atrasada dos rótulos
(revisão ~1 h para sinalizadas; chargeback ~72 h para fraudes aprovadas) e
mostra o que o time veria em diferentes datas de consulta.

Uso: python scripts/batch_monitor.py
Saída: reports/BATCH_MONITORING.md, reports/figures/label_maturation.png
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fraudguard.config import get_settings
from fraudguard.ews.batch_monitor import BatchMonitorConfig, cohort_report, simulate_label_arrival
from fraudguard.modeling.evaluation import load_eval_set

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    s = get_settings()
    ev = load_eval_set(s)
    ts_h = ev.test["Time"].to_numpy() / 3600
    dec = pd.DataFrame(
        {
            "ts_h": ts_h,
            "cohort": np.floor(ts_h - ts_h.min()).astype(int),
            "probability": ev.proba,
            "flagged": ev.proba >= ev.predictor.threshold,
            "amount": ev.amount,
            "label": ev.y,
        }
    )
    # Coortes de 3 h: com ~10 fraudes por hora, coortes horárias teriam rótulos demais escassos
    dec["cohort"] = (dec["cohort"] // 3) * 3
    dec["label_arrival_h"] = simulate_label_arrival(dec, np.random.default_rng(0))
    ref = ev.metadata["test_metrics"]["auprc"]
    cfg = BatchMonitorConfig(maturity_h=96, min_frauds=5)
    end = dec["ts_h"].max()

    lines = [
        "# Monitoramento com rótulos atrasados (camada 3)",
        "",
        "Partição de teste tratada como produção, em coortes de 3 horas pela data da transação. "
        "Rótulos simulados: sinalizadas são revisadas em ~1 h; fraudes aprovadas só aparecem no "
        "chargeback, em ~72 h (exponencial). Maturação exigida: 96 h.",
        "",
    ]
    reports = {}
    for label, delta in (("6 horas", 6), ("2 dias", 48), ("5 dias", 120), ("15 dias", 360)):
        rep = cohort_report(dec, end + delta, s.cost, ref, cfg)
        reports[delta] = rep
        lines += [f"## Consulta {label} após o fim do período", "", rep.to_markdown(index=False), ""]

    frauds = dec["label"].astype(bool)
    true_recall = dec.loc[frauds, "flagged"].mean()
    known6 = frauds & (dec["label_arrival_h"] <= end + 6)
    apparent6 = dec.loc[known6, "flagged"].mean()
    r5, r15 = reports[120].set_index("cohort"), reports[360].set_index("cohort")
    worst = (r5["recall"] - r15["recall"]).idxmax()
    delays = dec.loc[frauds & ~dec["flagged"], "label_arrival_h"] - dec.loc[frauds & ~dec["flagged"], "ts_h"]
    p95 = float(np.quantile(delays, 0.95))

    def br(x, d=3):
        return f"{x:.{d}f}".replace(".", ",")

    lines += [
        "**Leitura.**",
        f"- Seis horas após o período, o recall aparente é {br(apparent6)} contra {br(true_recall)} real: as fraudes "
        "que o modelo pegou são rotuladas em minutos, as que ele perdeu só aparecem no chargeback. Por isso nenhuma "
        "métrica é calculada antes da maturação.",
        f"- Mesmo após 96 h a cobertura não é completa. Na coorte {worst}, o recall medido cai de "
        f"{br(r5.loc[worst, 'recall'])} (5 dias) para {br(r15.loc[worst, 'recall'])} (15 dias), quando todos os "
        f"rótulos chegam. O p95 do atraso de chargeback nesta simulação é {br(p95 / 24, 1)} dias: a janela de "
        "maturação deve ser calibrada por esse percentil, não pela média, e a cobertura sempre é publicada ao lado.",
        "- Coortes `DEGRADADO` (AUPRC mais de 10% abaixo da referência) ou `DESCALIBRADO` (ECE acima do limite) "
        "disparam investigação, não retreino automático.",
    ]
    (ROOT / "reports" / "BATCH_MONITORING.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    days = np.linspace(0, 15, 120)
    cov = [(dec.loc[frauds, "label_arrival_h"] <= end + d * 24).mean() for d in days]
    apparent_recall = []
    for d in days:
        known = frauds & (dec["label_arrival_h"] <= end + d * 24)
        apparent_recall.append((dec.loc[known, "flagged"]).mean() if known.any() else np.nan)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.plot(days, np.array(cov) * 100, color="#1F4E79", lw=2, label="fraudes com rótulo conhecido")
    ax.plot(days, np.array(apparent_recall) * 100, color="#B3261E", lw=2, label="recall aparente")
    ax.axhline(true_recall * 100, color="#B3261E", ls=":", label=f"recall real ({true_recall:.1%})".replace(".", ","))
    ax.axvline(4, color="#5A6878", ls="--", lw=1)
    ax.text(4.2, 8, "maturação (96 h)", color="#5A6878", fontsize=8)
    ax.set(xlabel="dias após a transação", ylabel="%", ylim=(0, 105), title="Por que esperar a maturação dos rótulos")
    ax.legend(fontsize=8, loc="center right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(ROOT / "reports" / "figures" / "label_maturation.png", dpi=150)
    plt.close(fig)
    print("\n".join(lines[:3] + lines[-4:]))


if __name__ == "__main__":
    main()
