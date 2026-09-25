# SDR-010: Intervalos de confiança por block bootstrap temporal

**Contexto.** O teste tem 94 fraudes. Reportar "AUPRC = 0,774" sem faixa sugere uma precisão que não existe e leva a decisões ruins, como promover um modelo por uma diferença que é ruído.

**Decisão.** Toda métrica de teste é publicada com IC 95% percentílico por *block bootstrap* temporal: blocos contíguos de 500 transações (~5 minutos de tráfego) sorteados com reposição, 1.000 réplicas. Comparações entre modelos usam *bootstrap pareado*: as mesmas reamostragens para os dois, o que cancela a variância comum.

| Métrica | Pontual | IC 95% |
|---|---|---|
| AUPRC | 0,774 | 0,691 a 0,848 |
| Recall | 0,755 | 0,667 a 0,838 |
| Precisão | 0,530 | 0,448 a 0,609 |
| Redução de custo | 76,5% | 61,2% a 86,1% |

**Por que blocos.** Fraude vem em rajadas; reamostrar transações individuais trata eventos correlacionados como independentes e estreita o intervalo. Nos dados sintéticos as larguras ficaram quase iguais (0,157 com blocos contra 0,156 i.i.d.), porque o gerador não produz rajadas. Mantemos blocos como padrão: com dados reais, o método i.i.d. seria otimista exatamente quando mais importa.

**Alternativas descartadas.** Intervalo analítico para AUPRC: depende de suposições de distribuição frágeis com poucos positivos. Validação cruzada temporal: estima variabilidade entre períodos, não a incerteza do teste; complementar, recomendada com meses de dados.

**Consequências.** A leitura muda: o modelo **com certeza** reduz o custo em mais de 60%; a magnitude exata é incerta. Toda comunicação a stakeholders passa a usar a faixa.
