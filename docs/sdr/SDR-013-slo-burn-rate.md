# SDR-013: SLOs com alertas de burn rate em múltiplas janelas

**Contexto.** Alertas de limiar simples ("p99 > 50 ms por 5 min") disparam demais em picos inofensivos e de menos em degradações lentas.

**Decisão.** Dois SLOs com orçamento de erro mensal e alertas de *burn rate* em pares de janelas (Google SRE Workbook, cap. 5), em `ops/prometheus/slo.yml`:

| SLO | Meta | Orçamento |
|---|---|---|
| Disponibilidade: predições sem erro interno | 99,9% | 0,1% |
| Latência: inferências abaixo de 50 ms | 99% | 1% |

| Alerta | Condição | Significado | Ação |
|---|---|---|---|
| Fast burn | 14,4× o orçamento em 1 h **e** em 5 min | 2% do orçamento mensal por hora | Page |
| Slow burn | 6× em 6 h **e** em 1 h | 5% do orçamento a cada 6 h | Ticket |

A janela curta confirma que o problema ainda está acontecendo, o que evita alertas sobre incidentes já resolvidos. Erros de validação (422) não consomem orçamento: são problema do cliente, não do serviço, e têm alerta próprio.

**Consequências.** Todas as regras são checadas no CI com `promtool`. O dashboard Grafana provisionado mostra o burn rate ao lado das métricas de negócio.
