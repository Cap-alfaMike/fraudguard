# Runbook

## Ataque

**Sinais:** `CARD_TESTING_BURST`, `SUSPICIOUS_RATE_SPIKE`, `HIGH_VALUE_FRAUD_ATTEMPT`; faixa do dashboard em "Sob ataque".

1. Gerar o relatório de operações (`POST /ews/report` com `{"audience": "operacoes"}`) e acionar a equipe de prevenção à fraude.
2. Teste de cartão: identificar no sistema transacional os estabelecimentos ou BINs de origem e aplicar bloqueio temporário ou limite de velocidade no gateway.
3. Priorizar a fila de revisão por `risk_level=CRITICO` e valor.
4. Se o volume de alertas exceder a capacidade da equipe, **não** subir o limiar às cegas: consultar o `recall_at_precision_90` em `/model/info` e decidir com Risco o novo ponto de operação.
5. Registrar o incidente; padrões novos entram no próximo retreino.

## Drift

**Sinais:** `PREDICTION_DRIFT`, `DATA_DRIFT`.

1. Checar primeiro o upstream: houve deploy no produtor, mudança de moeda, escala ou ordem de colunas? Picos de `fraudguard_errors_total{kind="validation"}` reforçam essa hipótese.
2. Ver em `/ews/status` quais variáveis têm PSI alto. Drift só em `Amount` pode ser sazonal (Black Friday); em várias componentes V ao mesmo tempo sugere mudança de pipeline.
3. Se o drift for real e persistente, iniciar retreino (ver MONITORING.md) e acompanhar a coorte com rótulos.

## Latência

**Sinais:** `LATENCY_SLO_BREACH`, alerta `FraudGuardLatencyP99High`.

1. Ver o uso de CPU das réplicas; escalar horizontalmente.
2. Verificar nos logs se `fast_path` está `false` após o último deploy: significa que a verificação de paridade falhou e o serviço caiu para o sklearn (~9 ms por transação). Investigar o artefato antes de promover.
3. Lotes grandes em `/predict/batch` competem com o tráfego unitário; considerar réplicas separadas para batch.

## Erros

**Sinais:** alerta `FraudGuardErrorRate`, respostas 500 ou 503.

1. 503 em `/health`: modelo não carregado. Os logs de startup (`falha ao carregar modelo`) mostram a causa. A réplica continua viva e fora do balanceador.
2. 500 em `/predict`: buscar `falha de inferência` nos logs pelo `request_id` retornado no cabeçalho `x-request-id`.
3. Rollback: publicar a tag anterior da imagem.

## LLM

**Sinais:** relatórios com `source: "template"`, log `falha no LLM`.

Nenhuma ação urgente: o scoring não depende do LLM. Verificar a chave `ANTHROPIC_API_KEY`, conectividade e o status do provedor. Log `relatório LLM com baixo grounding` indica que o texto citou números que não estão nos dados: revisar o relatório antes de circular.
