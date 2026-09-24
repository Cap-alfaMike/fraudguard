# SDR-002: Deduplicação antes do split e split temporal

**Contexto.** O dataset ULB tem 1.081 linhas duplicadas exatas (o sintético reproduz 1.082). Fraude acontece em rajadas: o mesmo fraudador testa vários cartões em minutos. Em produção, o modelo sempre prevê o futuro a partir do passado.

**Decisão.**
1. Remover duplicatas exatas **antes** de qualquer divisão, para que cópias não apareçam em treino e teste ao mesmo tempo (o modelo seria premiado por memorizar).
2. Split cronológico por `Time`: treino (60%) | validação (20%) | teste (20%).
3. Os últimos 15% do treino servem de conjunto de early stopping. Assim, a validação fica limpa para seleção de modelo, calibração e limiar, e o teste é usado **uma única vez**, no final.

| Partição | Linhas | Fraudes |
|---|---|---|
| Treino (inclui early stopping) | 170.235 | 334 |
| Validação | 56.745 | 64 |
| Teste | 56.745 | 94 |

**Alternativas descartadas.** Split aleatório estratificado: vaza padrões de ataques que se repetem no tempo e superestima o desempenho, efeito documentado no *Fraud Detection Handbook* da ULB (Le Borgne et al., 2022). Validação cruzada k-fold aleatória: mesmo problema. Validação cruzada temporal (*rolling origin*): mais robusta, mas com apenas ~48 h de dados cada dobra teria poucas fraudes; recomendada quando houver meses de histórico.

**Consequências.** As métricas reportadas são mais pessimistas e mais honestas. A validação tem só 64 fraudes, o que pesa nas decisões de calibração (SDR-004). O mesmo princípio orienta o retreino em produção: janela deslizante, sempre avaliando no período mais recente.
