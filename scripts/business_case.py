"""Business case: projeta o impacto financeiro medido no teste para o volume da plataforma.

Método (transparente e conservador):
* Parte do custo medido no conjunto de TESTE (nunca visto pelo modelo).
* Decompõe o ganho em (a) parcela que escala com o volume de FRAUDE e
  (b) custo de falsos positivos, que escala com o volume de LEGÍTIMAS.
* Mantém recall e FPR do teste constantes ao variar a prevalência de fraude.
  Aproximação de primeira ordem; a precisão cai com prevalência menor, e isso
  já está capturado porque o custo de FP permanece fixo por transação legítima.
* Tudo é ESTIMATIVA sob as premissas do CostModel (config.py), que precisam ser
  calibradas com Finanças/Risco.

Uso: python scripts/business_case.py --tx-per-minute 3000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def br(x: float, dec: int = 0) -> str:
    return f"{x:,.{dec}f}".replace(",", "\x00").replace(".", ",").replace("\x00", ".")


def project(meta: dict, tx_per_minute: float, prevalence_factors=(0.25, 0.5, 1.0)) -> dict:
    tb, tm, cost = meta["test_business"], meta["test_metrics"], meta["cost_model"]
    n_test = meta["data"]["split"]["test"]["rows"]
    cm = tm["confusion_matrix"]
    missed_cost = tb["total_cost"] - tb["false_positive_cost"] - tb["review_cost"]
    fraud_side_gain = tb["baseline_cost"] - missed_cost - tb["review_cost"]  # escala com fraude
    fp_cost = tb["false_positive_cost"]  # escala com legítimas

    tx_year = tx_per_minute * 60 * 24 * 365
    scale = tx_year / n_test
    scenarios = []
    for k in prevalence_factors:
        saving = (k * fraud_side_gain - fp_cost) * scale
        baseline = k * tb["baseline_cost"] * scale
        scenarios.append(
            {
                "prevalencia_relativa": k,
                "custo_sem_modelo_eur_ano": round(baseline),
                "economia_eur_ano": round(saving),
                "economia_pct": round(100 * saving / baseline, 1) if baseline else 0.0,
            }
        )

    churn_fraud, churn_fp = cost["churn_prob_after_fraud"], cost["churn_prob_after_false_decline"]
    retained = cm["tp"] * churn_fraud * scale  # clientes que teriam saído após sofrer fraude
    lost_friction = cm["fp"] * churn_fp * scale  # clientes perdidos por bloqueio indevido
    return {
        "volume_tx_ano": int(tx_year),
        "economia_por_1000_tx_eur": round(1000 * tb["savings"] / n_test, 2),
        "cenarios": scenarios,
        "risco_marca": {
            "clientes_preservados_ano": round(retained),
            "clientes_perdidos_por_atrito_ano": round(lost_friction),
            "saldo_liquido_clientes_ano": round(retained - lost_friction),
            "clv_eur": cost["customer_lifetime_value"],
            "valor_liquido_clv_eur_ano": round((retained - lost_friction) * cost["customer_lifetime_value"]),
        },
        "operacao": {
            "alertas_por_minuto": round(tm["alert_rate"] * tx_per_minute, 2),
            "fraudes_capturadas_pct": round(100 * tm["recall"], 1),
            "clientes_legitimos_incomodados_pct": round(100 * tm["false_positive_rate"], 3),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tx-per-minute", type=float, default=3000)
    ap.add_argument("--metadata", type=Path, default=ROOT / "artifacts" / "model_metadata.json")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    res = project(json.loads(a.metadata.read_text()), a.tx_per_minute)
    if a.json:
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return
    print(f"Volume: {br(a.tx_per_minute)} transações/min = {br(res['volume_tx_ano'])} por ano")
    print(f"Economia medida no teste: €{br(res['economia_por_1000_tx_eur'], 2)} por 1.000 transações\n")
    print("| Prevalência de fraude vs. dataset | Custo sem modelo (€/ano) | Economia (€/ano) | Redução |")
    print("|---|---|---|---|")
    for s in res["cenarios"]:
        print(
            f"| {br(100 * s['prevalencia_relativa'])}% | €{br(s['custo_sem_modelo_eur_ano'])} "
            f"| €{br(s['economia_eur_ano'])} | {br(s['economia_pct'], 1)}% |"
        )
    r, o = res["risco_marca"], res["operacao"]
    print(
        f"\nMarca: ~{br(r['clientes_preservados_ano'])} clientes/ano que sairiam após sofrer fraude são preservados; "
        f"~{br(r['clientes_perdidos_por_atrito_ano'])} saem por bloqueio indevido. "
        f"Saldo: {br(r['saldo_liquido_clientes_ano'])} clientes (€{br(r['valor_liquido_clv_eur_ano'])} em CLV)."
    )
    print(
        f"Operação: {br(o['alertas_por_minuto'], 1)} alertas/min para revisão; "
        f"{br(o['fraudes_capturadas_pct'], 1)}% das fraudes capturadas; "
        f"{br(o['clientes_legitimos_incomodados_pct'], 3)}% dos clientes legítimos incomodados."
    )


if __name__ == "__main__":
    main()
