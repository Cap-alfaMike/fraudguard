# KPIs

Os indicadores formam uma árvore: os de negócio respondem "vale a pena?", os de modelo respondem "a decisão é boa?", os operacionais respondem "a equipe consegue agir?" e os de sistema respondem "o serviço aguenta?". Cada KPI tem uma fonte que pode ser consultada, não só uma meta.

Valores "medidos" vêm do conjunto de teste temporal (dados sintéticos com perfil ULB, 56.745 transações, 94 fraudes) ou do benchmark de carga. Metas são propostas iniciais para validar com as áreas.

## Negócio

| KPI | Definição | Meta | Medido | Fonte |
|---|---|---|---|---|
| Redução do custo de fraude | 1 − custo com modelo / custo sem modelo | ≥ 60% | **76,5%** | `test_business.savings_pct` |
| Economia por 1.000 transações | Economia / volume × 1.000 | ≥ €50 | **€134,13** | `scripts/business_case.py` |
| Perda de fraude evitada | Valor das fraudes bloqueadas / valor total de fraude | ≥ 75% | **86,9%** (€5.780 de €6.655) | `fraud_loss_prevented` |
| Saldo de clientes preservados | Churn evitado por fraude − churn por atrito | > 0 | **+2,9 a cada 56,7 mil tx** | `business_case.py` (risco_marca) |
| Taxa de bloqueio indevido | FP / legítimas | ≤ 0,2% | **0,111%** | `test_metrics.false_positive_rate` |

A perda evitada em valor (86,9%) supera o recall em contagem (75,5%) porque o custo por valor faz o modelo priorizar as fraudes caras.

## Modelo

| KPI | Meta | Medido | Observação |
|---|---|---|---|
| AUPRC (teste) | ≥ 0,70 | **0,774** | Classificador aleatório = 0,0017 (≈ 467× melhor) |
| Recall no limiar | ≥ 70% | **75,5%** | 71 de 94 fraudes |
| Precisão no limiar | ≥ 40% | **53,0%** | 1 alerta falso para cada alerta verdadeiro |
| Recall com precisão ≥ 90% | referência | 66,0% | Opção se a operação exigir alertas mais certeiros |
| Brier (calibração) | menor que o não calibrado | 0,000604 vs 0,000741 | Probabilidades confiáveis para decisão por custo |
| ROC-AUC | referência | 0,990 | Reportada, mas não usada para decidir (SDR-001) |

## Operação

| KPI | Meta | Medido/estimado | Fonte |
|---|---|---|---|
| Alertas para revisão | cabe na capacidade da equipe | 0,24% do volume; ~7 por minuto a 3.000 tx/min | `alert_rate` |
| Tempo até detecção de ataque | < 2 min | teste de cartão detectado com 10 eventos em 60 s | `CARD_TESTING_BURST` |
| Alertas críticos por dia | acompanhar tendência | — | `fraudguard_ews_alerts_total` |
| Relatórios com grounding ≥ 80% | 100% | 100% nos testes | `grounding_ratio` |

## Sistema

| KPI | SLO | Medido | Fonte |
|---|---|---|---|
| Latência de inferência p99 | < 50 ms | **0,84 ms** | `fraudguard_inference_latency_seconds` |
| Latência ponta a ponta p50 | < 10 ms | **2,6 ms** | benchmark, concorrência 1 |
| Throughput por réplica | > 3.000 tx/min | **~22.000 tx/min** | benchmark |
| Erros internos | < 0,1% | 0% | `fraudguard_errors_total{kind!="validation"}` |
| Disponibilidade | 99,9% | — | probes do orquestrador |
| PSI dos scores | < 0,10 | recalculado a cada 100 eventos | `/ews/status` |
