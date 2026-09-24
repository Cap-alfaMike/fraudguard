# SDR-004: Probabilidades calibradas e limiar que minimiza o custo esperado

**Contexto.** Aprovar uma fraude custa o valor da transação, mais chargeback, mais o risco de o cliente abandonar o banco por sentir-se inseguro. Bloquear um cliente legítimo custa a revisão e o risco de ele abandonar por atrito. Esses custos são assimétricos e o primeiro depende do valor, o que é conhecido como classificação *example-dependent cost-sensitive* (Bahnsen et al., 2014; Elkan, 2001).

**Decisão.**
1. **Modelo de custo explícito** (`CostModel`), com premissas visíveis e ajustáveis:

   | Premissa | Valor padrão |
   |---|---|
   | Custo de revisão/contato | €3 |
   | Taxa de chargeback e disputa | €15 |
   | CLV (valor do cliente no tempo) | €400 |
   | Prob. de churn após sofrer fraude | 5% |
   | Prob. de churn após bloqueio indevido | 1% |

   Custo de falso negativo = valor + €15 + 5% × €400 = **valor + €35**. Custo de falso positivo = €3 + 1% × €400 = **€7**. O churn por fraude é o componente de **dano à marca**, agora quantificado.
2. **Calibração Platt (sigmoid)** na validação, sobre o modelo congelado (`FrozenEstimator`). Isotônica foi descartada: com apenas 64 fraudes na validação ela vira uma função em degraus que sobreajusta. A calibração melhorou o Brier no teste de 0,000741 para 0,000604.
3. **Limiar** = argmin do custo total na validação, calculado de forma vetorizada para todos os limiares candidatos (O(n log n)). Resultado: **0,055**.

**Por que o limiar é tão baixo.** Porque uma fraude média custa muitas vezes mais que um falso alarme; vale a pena revisar uma transação com 5,5% de chance de fraude. Um limiar de 0,5 (padrão ingênuo) deixaria passar fraudes que custam caro para evitar alarmes que custam pouco.

**Resultado no teste:** custo cai de €9.945 (sem modelo) para €2.334, redução de 76,5%.

**Consequências.** As premissas precisam ser calibradas com Finanças e Risco antes de produção. Como o limiar é derivado, não treinado, mudar uma premissa leva minutos. Evolução natural: limiar dependente do valor (regra de Bayes por transação: sinalizar se p × custo_FN(valor) > (1−p) × custo_FP), que exige calibração mais robusta e mais dados rotulados.
