# SLOs e política de orçamento de erro

## SLIs e SLOs

| SLI | Fonte | SLO (30 dias) |
|---|---|---|
| Disponibilidade: fração de predições sem erro interno | `fraudguard_predictions_total`, `fraudguard_errors_total{kind!="validation"}` | 99,9% |
| Latência: fração de inferências < 50 ms | `fraudguard_inference_latency_seconds_bucket{le="0.05"}` | 99% |
| Frescor do modelo: idade do champion | `model_metadata.json` | < 45 dias |
| Qualidade: AUPRC da última coorte madura | `scripts/batch_monitor.py` | ≥ 90% da referência |

## Política de orçamento

- **Orçamento saudável (> 50% restante):** deploys normais.
- **Orçamento em risco (< 50%):** deploys só com correções; mudança de modelo exige aprovação de SRE.
- **Orçamento esgotado:** congelamento de mudanças até recuperar; postmortem obrigatório para incidentes que consumiram mais de 20%.

Alertas de burn rate: ver [SDR-013](sdr/SDR-013-slo-burn-rate.md). Procedimentos: [RUNBOOK](RUNBOOK.md).
