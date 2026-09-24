# SDR-001: AUPRC para selecionar, custo em EUR para decidir

**Contexto.** Com 0,17% de fraudes, um classificador que aprova tudo tem 99,83% de acurácia. A ROC-AUC também engana: ela é dominada pelos verdadeiros negativos, e diferenças enormes em quantos clientes bons são incomodados aparecem como variações na terceira casa decimal. No nosso teste, ROC-AUC = 0,990 enquanto a precisão no limiar é 0,53; a ROC sozinha esconde que metade dos alertas são falsos.

**Decisão.** Dois níveis de métrica, cada um com seu papel:
1. **Seleção de modelo:** AUPRC (Average Precision), independente de limiar e sensível exatamente à região que importa (Saito & Rehmsmeier, 2015).
2. **Decisão operacional:** custo esperado total em EUR (SDR-004), porque o negócio não sente "F1": sente dinheiro perdido e clientes irritados.

Métricas complementares reportadas: recall com precisão ≥ 80% e ≥ 90% (quanto conseguimos capturar se a operação exigir alertas confiáveis), Brier score (qualidade da calibração), F2 (dá mais peso ao recall, útil como referência).

**Alternativas descartadas.** Acurácia (inútil). F1 como critério de limiar (assume custo simétrico entre FP e FN, o que é falso: uma fraude de €2.000 não custa o mesmo que um cliente incomodado). ROC-AUC como critério primário (otimista sob desbalanceamento).

**Consequências.** O limiar ótimo depende das premissas de custo, que viram parâmetros de negócio versionados (`CostModel` em `config.py`). Mudar uma premissa exige apenas recalcular o limiar, não retreinar.
