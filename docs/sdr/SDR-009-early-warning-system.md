# SDR-009: Early Warning System in-process, com agregação no Prometheus

**Contexto.** Rótulos de fraude chegam com dias ou semanas de atraso (chargebacks). Esperar pelos rótulos para descobrir que o modelo degradou ou que um ataque está em curso é tarde demais (Dal Pozzolo et al., 2018).

**Decisão.** Monitorar sinais disponíveis **imediatamente**, em janela deslizante de 2.000 predições:

| Família | Sinal | Regra padrão | Severidade |
|---|---|---|---|
| Ataque | Teste de cartão | ≥ 10 micro-transações (≤ €2) suspeitas em 60 s | Crítica |
| Ataque | Pico de suspeitas | Taxa ≥ 3× o baseline da validação | Atenção; crítica se ≥ 6× |
| Ataque | Alto valor | Suspeita ≥ €1.000 com p ≥ 0,5 | Crítica |
| Modelo | Drift de predição | PSI dos scores ≥ 0,10 (crítico ≥ 0,25) | Atenção/crítica |
| Modelo | Drift de dados | PSI ≥ 0,25 nas 8 variáveis mais discriminativas | Atenção |
| Sistema | Latência | p99 acima do SLO (50 ms) | Atenção |

PSI foi escolhido por ser o padrão de mercado em risco de crédito, interpretável por Risco e Compliance, e barato (Siddiqi, 2006). Os bins vêm de quantis da validação, o que dá robustez às caudas pesadas. Os alertas têm deduplicação por tipo (cooldown de 5 min; 60 s para alto valor, que é pontual).

Custo no caminho quente: O(1) por predição (append em `deque` e contadores). Regras de janela rodam a cada 100 eventos.

**Alternativas descartadas.** Evidently ou Alibi Detect no processo: excelentes para relatórios batch e testes estatísticos mais ricos (MMD, KS multivariado), mas pesados para o caminho de cada requisição; recomendados em um job batch diário, complementar. Streaming dedicado (Kafka + Flink): arquitetura certa em escala de plataforma; aqui seria infraestrutura desproporcional.

**Consequências.** Cada réplica vê só seu tráfego. Para ataques distribuídos entre réplicas, as regras equivalentes rodam no Prometheus sobre métricas agregadas (`ops/prometheus/alerts.yml`). Os limiares das regras são parâmetros configuráveis e devem ser ajustados com o histórico de alertas.
