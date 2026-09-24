# SDR-003: LightGBM, com o tratamento de desbalanceamento escolhido empiricamente

**Contexto.** A EDA mostrou sinal forte, não linear e concentrado em poucas componentes (KS de 0,63 em V17, 0,61 em V14, 0,57 em V12), caudas pesadas e outliers que são o próprio sinal de fraude (70% das fraudes têm algum valor além de 3×IQR, contra 10% das legítimas). O serviço precisa de latência de milissegundos.

**Decisão.** Comparar quatro candidatos na mesma validação temporal, pela AUPRC:

| Candidato | AUPRC validação |
|---|---|
| Regressão logística, `class_weight="balanced"` (baseline interpretável) | 0,551 |
| **LightGBM sem pesos** | **0,763** |
| LightGBM, `scale_pos_weight = √(neg/pos)` | 0,672 |
| LightGBM, `scale_pos_weight = neg/pos` | 0,660 |

O vencedor foi o LightGBM sem ponderação. O resultado é coerente com a literatura: pesos de classe e reamostragem alteram a distribuição de treino e deslocam as probabilidades, e para *ranking* um boosting bem regularizado costuma lidar bem com o desbalanceamento (Dal Pozzolo et al., 2015). O desbalanceamento ainda é tratado explicitamente em três pontos: na métrica (AUPRC), no early stopping (monitorando `average_precision`) e, principalmente, na decisão (limiar por custo, SDR-004), onde ele realmente importa.

Por que gradient boosting: invariante a escala e robusto a outliers, captura interações sem engenharia manual, estado da arte consolidado em dados tabulares, explicações exatas via TreeSHAP nativo (`pred_contrib`) e inferência em microssegundos. LightGBM em vez de XGBoost: desempenho equivalente em dados tabulares, treino mais rápido por histogramas e crescimento *leaf-wise* (Ke et al., 2017).

**Alternativas descartadas.**
- *SMOTE:* gera fraudes sintéticas interpolando vizinhos, o que em espaço PCA pode criar pontos sem significado; distorce a calibração; precisaria estar dentro da validação cruzada para não vazar. Ganho raramente supera o de pesos de classe em boosting.
- *Undersampling:* descarta 99% dos dados legítimos e exige correção das probabilidades (Dal Pozzolo et al., 2015).
- *Redes neurais/autoencoders:* sem ganho esperado em tabular de 30 colunas, com mais custo de operação e menos explicabilidade.
- *Isolation Forest/não supervisionado:* útil quando não há rótulos; aqui há, e o supervisionado domina.

**Consequências.** Seleção reprodutível e auditável (resultados de todos os candidatos ficam em `model_metadata.json`). A escolha é refeita a cada retreino: se, com o CSV real, uma versão ponderada vencer, ela será selecionada automaticamente pelo mesmo critério.
