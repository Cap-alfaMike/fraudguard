# Monitoramento em produção

O desafio central: **os rótulos chegam atrasados.** Uma fraude só é confirmada quando o cliente contesta, dias ou semanas depois. Por isso o monitoramento tem três camadas, da mais rápida (sem rótulos) à mais precisa (com rótulos).

## Camada 1: segundos (sem rótulos), no Early Warning System

- **Drift de predição:** PSI da distribuição de scores contra a validação. É o sinal mais precoce de que algo mudou, seja nos dados, seja no comportamento dos fraudadores.
- **Drift de dados:** PSI das oito variáveis mais discriminativas (`Amount`, V14, V17, V12, V10, V4, V11, V3). Escolhemos as que mais importam para o modelo; drift em variável irrelevante não merece alarme.
- **Padrões de ataque:** teste de cartão, pico de taxa de suspeitas, tentativas de alto valor.
- **Saúde do sistema:** latência, erros, taxa de requisições inválidas (proxy de mudança de contrato no upstream).

## Camada 2: minutos a horas, no Prometheus

Métricas agregadas de todas as réplicas, com regras em `ops/prometheus/alerts.yml`: latência p99, taxa de erro, pico de suspeitas contra a média de 7 dias, alertas críticos do EWS. Isso cobre ataques distribuídos que uma réplica sozinha não percebe.

## Camada 3: dias a semanas (com rótulos), job batch

Quando os chargebacks chegam, um job diário junta decisões registradas e rótulos confirmados e calcula:
- AUPRC, recall e precisão por coorte semanal, **por data da transação** (não da chegada do rótulo), para não confundir atraso com degradação.
- Calibração: a fração de fraudes entre transações com p ≈ 0,1 continua perto de 10%?
- Custo realizado contra custo previsto.
- Taxa de falso positivo por segmento, quando houver features interpretáveis.

Ferramentas adequadas para essa camada: Evidently ou Alibi Detect para relatórios e testes estatísticos multivariados.

**Feedback loop.** Transações sinalizadas são revisadas e ganham rótulo rápido; as aprovadas só ganham rótulo se houver contestação. Isso enviesa o conjunto rotulado. Mitigação: aprovar e acompanhar uma pequena amostra aleatória de transações que seriam sinalizadas (*exploration*), com limite de valor, para ter uma estimativa não enviesada do desempenho.

## Gatilhos de retreino

| Gatilho | Ação |
|---|---|
| PSI dos scores ≥ 0,25 por mais de 24 h | Investigar a causa; retreinar se não for problema de upstream |
| AUPRC da coorte cai mais de 10% contra a referência | Retreinar com a janela mais recente |
| Calendário | Retreino mensal com janela deslizante, mesmo sem alarme |
| Mudança nas premissas de custo | Apenas recalcular o limiar (sem retreino) |

**Promoção de modelo.** O novo modelo roda em *shadow* (pontua sem decidir) por pelo menos uma semana. É promovido se superar o atual em AUPRC e custo na mesma coorte, sem aumentar a taxa de falso positivo além do limite acordado. Como a imagem é imutável e versionada, o rollback é trocar a tag.
