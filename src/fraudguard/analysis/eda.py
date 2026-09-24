"""EDA + Missing Analysis reprodutíveis -> reports/EDA.md e reports/figures/.

Cada seção termina com a DECISÃO de engenharia que ela motiva, para que o
relatório seja rastreável até o código (docs/SDD.md §3).

Uso:  python -m fraudguard.analysis.eda [--data creditcard.csv]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from fraudguard.config import PCA_FEATURES, PROJECT_ROOT, RAW_FEATURES, TARGET, get_settings
from fraudguard.data.loader import load_dataset

FIG_DIR = PROJECT_ROOT / "reports" / "figures"
INK, FRAUD, LEGIT = "#1B3A5C", "#B3261E", "#6B7F99"


def _fmt_table(df: pd.DataFrame, floatfmt: str = "{:,.4f}") -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join([df.index.name or "", *cols]) + " |", "|" + "---|" * (len(cols) + 1)]
    for idx, row in df.iterrows():
        cells = [floatfmt.format(v) if isinstance(v, float | np.floating) else str(v) for v in row]
        lines.append("| " + " | ".join([str(idx), *cells]) + " |")
    return "\n".join(lines)


def run_eda(df: pd.DataFrame, source: str, make_plots: bool = True) -> str:
    out: list[str] = [f"# Análise Exploratória de Dados (EDA)\n\nFonte: `{source}` — {len(df):,} linhas × {df.shape[1]} colunas.\n"]
    y = df[TARGET]
    fraud, legit = df[y == 1], df[y == 0]

    # 1. Desbalanceamento
    rate = y.mean()
    out += [
        "## 1. Desbalanceamento de classes",
        f"- Fraudes: **{int(y.sum()):,}** ({rate:.4%}) — razão ≈ 1:{(1 - rate) / rate:,.0f}.",
        "- **Decisão:** acurácia é inútil (99,8% prevendo 'tudo legítimo'). Métrica primária = AUPRC; "
        "decisão por custo esperado; split temporal estratificado por construção.\n",
    ]

    # 2. Missing
    miss = df.isna().sum()
    miss_pct = miss / len(df) * 100
    out += [
        "## 2. Análise de valores ausentes (Missing Analysis)",
        f"- Total de células ausentes: **{int(miss.sum()):,}** ({miss.sum() / df.size:.4%}).",
    ]
    if miss.sum() == 0:
        out.append("- Nenhuma coluna com ausentes. Mecanismo MCAR/MAR/MNAR não se aplica a este extrato.")
    else:
        tbl = pd.DataFrame({"ausentes": miss[miss > 0], "%": miss_pct[miss > 0].round(4)})
        tbl.index.name = "coluna"
        out.append(_fmt_table(tbl))
        has_miss = df[RAW_FEATURES].isna().any(axis=1)
        out.append(
            f"- Taxa de fraude com ausente: {y[has_miss].mean():.4%} vs sem: {y[~has_miss].mean():.4%} "
            "(diferença grande sugere MNAR → ausência é sinal e mereceria indicador)."
        )
    out.append(
        "- **Decisão:** mesmo sem ausentes no treino, produção terá (falhas upstream). "
        "`SimpleImputer(median)` dentro do pipeline (ajustado só no treino) + rejeição de nulos na "
        "API (Pydantic). Mediana por robustez às caudas pesadas.\n"
    )

    # 3. Duplicatas
    dup_mask = df.duplicated(keep=False)
    n_dup = int(df.duplicated().sum())
    out += [
        "## 3. Duplicatas",
        f"- Linhas duplicadas exatas: **{n_dup:,}** ({n_dup / len(df):.3%}); "
        f"fraudes entre elas: {int(df.loc[dup_mask, TARGET].sum())}.",
        "- **Decisão:** deduplicar ANTES do split — cópias em treino e teste inflam métricas (memorização).\n",
    ]

    # 4. Amount
    amt = pd.DataFrame(
        {
            "legítima": legit["Amount"].describe(percentiles=[0.5, 0.9, 0.99]),
            "fraude": fraud["Amount"].describe(percentiles=[0.5, 0.9, 0.99]),
        }
    )
    amt.index.name = "Amount (EUR)"
    zero_share = (df["Amount"] == 0).mean()
    micro_fraud = ((fraud["Amount"] > 0) & (fraud["Amount"] <= 2)).mean()
    micro_legit = ((legit["Amount"] > 0) & (legit["Amount"] <= 2)).mean()
    skew = df["Amount"].skew()
    out += [
        "## 4. Valor da transação (`Amount`)",
        _fmt_table(amt, "{:,.2f}"),
        f"\n- Assimetria (skew) = {skew:.2f}; {zero_share:.2%} das transações têm valor 0.",
        f"- Micro-valores (≤ €2): {micro_fraud:.1%} das fraudes vs {micro_legit:.1%} das legítimas — assinatura de *card testing*.",
        "- **Decisão:** `log1p(Amount)` + `RobustScaler`; flags `amount_is_zero` e `amount_is_micro`; "
        "custo de falso negativo proporcional ao valor.\n",
    ]

    # 5. Tempo
    hour = ((df["Time"] % 86_400) // 3_600).astype(int)
    by_hour = df.groupby(hour)[TARGET].agg(["count", "mean"]).rename(columns={"count": "transações", "mean": "taxa_fraude"})
    night = by_hour.loc[0:5, "taxa_fraude"].mean()
    day = by_hour.loc[8:22, "taxa_fraude"].mean()
    out += [
        "## 5. Padrão temporal",
        f"- Taxa média de fraude na madrugada (0–5h): {night:.3%} vs dia (8–22h): {day:.3%} (≈ {night / day:.1f}×).",
        "- **Decisão:** `Time` bruto é um contador desde o início da coleta e não generaliza; usamos só "
        "a hora do dia em codificação cíclica (sin/cos) e a flag `is_night`. Split temporal.\n",
    ]

    # 6. Poder discriminativo univariado
    ks = {c: ks_2samp(fraud[c].dropna(), legit[c].dropna()).statistic for c in [*PCA_FEATURES, "Amount"]}
    ks_s = pd.Series(ks).sort_values(ascending=False)
    top = pd.DataFrame(
        {
            "KS": ks_s.head(10).round(4),
            "média_fraude": fraud[ks_s.head(10).index].mean().round(3),
            "média_legítima": legit[ks_s.head(10).index].mean().round(3),
        }
    )
    top.index.name = "feature"
    out += [
        "## 6. Poder discriminativo (KS entre classes)",
        _fmt_table(top, "{:,.3f}"),
        f"\n- {int((ks_s > 0.5).sum())} variáveis com KS > 0,5; {int((ks_s < 0.1).sum())} com KS < 0,1.",
        "- **Decisão:** sinal forte e não linear em poucas componentes → gradient boosting; "
        "as top-KS são monitoradas pelo EWS (drift das variáveis que mais importam).\n",
    ]

    # 7. Correlação
    corr = df[PCA_FEATURES].corr().abs().to_numpy()
    off = corr[~np.eye(len(PCA_FEATURES), dtype=bool)]
    amt_corr = df[PCA_FEATURES].corrwith(df["Amount"]).abs().sort_values(ascending=False)
    out += [
        "## 7. Correlação",
        f"- |corr| máxima entre componentes V: {off.max():.3f} (média {off.mean():.4f}) — consistente com PCA.",
        f"- Maiores |corr| com Amount: {', '.join(f'{k}={v:.2f}' for k, v in amt_corr.head(3).items())}.",
        "- **Decisão:** sem multicolinearidade; nenhuma remoção de features necessária.\n",
    ]

    # 8. Outliers
    q1, q3 = df[PCA_FEATURES].quantile(0.25), df[PCA_FEATURES].quantile(0.75)
    iqr = q3 - q1
    outlier = ((df[PCA_FEATURES] < q1 - 3 * iqr) | (df[PCA_FEATURES] > q3 + 3 * iqr)).any(axis=1)
    out += [
        "## 8. Outliers",
        f"- Linhas com algum valor além de 3×IQR: legítimas {outlier[y == 0].mean():.2%}, fraudes {outlier[y == 1].mean():.2%}.",
        "- **Decisão:** NÃO remover outliers — em fraude, o outlier frequentemente É o sinal. "
        "Tratamento via escalonamento robusto e modelo baseado em árvores.\n",
    ]

    if make_plots:
        _plots(df, by_hour, ks_s)
        out += [
            "## Figuras",
            "![Amount por classe](figures/eda_amount.png)",
            "![Taxa de fraude por hora](figures/eda_hour.png)",
            "![Ranking KS](figures/eda_ks.png)",
        ]
    return "\n".join(out)


def _plots(df, by_hour, ks_s) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    bins = np.logspace(-1, 4.5, 50)
    ax.hist(df.loc[df[TARGET] == 0, "Amount"].clip(0.1), bins=bins, density=True, alpha=0.6, color=LEGIT, label="legítima")
    ax.hist(df.loc[df[TARGET] == 1, "Amount"].clip(0.1), bins=bins, density=True, alpha=0.7, color=FRAUD, label="fraude")
    ax.set(xscale="log", xlabel="Amount (EUR, log)", ylabel="densidade", title="Distribuição do valor por classe")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "eda_amount.png", dpi=130)
    plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(6, 3.6))
    ax1.bar(by_hour.index, by_hour["transações"], color=LEGIT, alpha=0.5, label="volume")
    ax2 = ax1.twinx()
    ax2.plot(by_hour.index, by_hour["taxa_fraude"] * 100, color=FRAUD, marker="o", lw=2, label="taxa de fraude (%)")
    ax1.set(xlabel="hora do dia", ylabel="transações", title="Volume × taxa de fraude por hora")
    ax2.set_ylabel("taxa de fraude (%)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "eda_hour.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4.2))
    top = ks_s.head(15)[::-1]
    ax.barh(top.index, top.values, color=INK)
    ax.set(xlabel="estatística KS (fraude vs legítima)", title="Poder discriminativo univariado")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "eda_ks.png", dpi=130)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=None)
    args = parser.parse_args()
    df, source = load_dataset(args.data or get_settings().raw_data_path)
    report = run_eda(df, source)
    path = PROJECT_ROOT / "reports" / "EDA.md"
    path.write_text(report, encoding="utf-8")
    print(f"EDA salva em {path}")


if __name__ == "__main__":
    main()
