# Incerteza das métricas de teste

Modelo `20260924114649-4a06a7b0`; limiar 0,0550; 56.745 transações e 94 fraudes no teste.
Método: block bootstrap temporal, blocos de 500 transações, 1000 réplicas, IC 95% percentílico. Ver `docs/sdr/SDR-010-incerteza.md`.

| Métrica | Pontual | IC 95% (block bootstrap) | IC 95% (i.i.d., para comparação) |
|---|---|---|---|
| AUPRC | 0,774 | 0,691 a 0,848 | 0,695 a 0,851 |
| Recall | 0,755 | 0,667 a 0,838 | 0,670 a 0,844 |
| Precisão | 0,530 | 0,448 a 0,609 | 0,446 a 0,617 |
| Taxa de falso positivo | 0,00111 | 0,00083 a 0,00141 | 0,00083 a 0,00138 |
| Economia (EUR) | 7.611 | 4.013 a 12.854 | 4.095 a 13.011 |
| Redução de custo (%) | 76,5 | 61,2 a 86,1 | 61,7 a 86,2 |

**Leitura.**
- As larguras da AUPRC são próximas (0,157 com blocos, 0,156 i.i.d.): nestes dados as fraudes não se concentram em rajadas. Em dados reais, onde ataques vêm em ondas, o block bootstrap tende a alargar o intervalo; por isso ele é o método padrão.
- Mesmo no limite inferior, a redução de custo é de 61,2%: o ganho do modelo é robusto, a magnitude exata é que é incerta.
- Comunicar sempre a faixa, não só o ponto.
