# Champion × challenger

- Champion: `20260924114649-4a06a7b0` (lgbm_unweighted, limiar 0,0550)
- Challenger: `20260925122050-4a06a7b0` (teto de 300 árvores, limiar 0,0590)
- Comparação: bootstrap pareado por blocos temporais na mesma partição de teste, 1.000 réplicas.

| Diferença (challenger − champion) | Pontual | IC 95% |
|---|---|---|
| AUPRC (maior é melhor) | +0,0075 | +0,0024 a +0,0136 |
| Custo em EUR (menor é melhor) | -42 | -84 a -7 |
| Taxa de falso positivo (menor é melhor) | -0,00011 | -0,00021 a -0,00002 |

P(AUPRC do challenger maior) = 100%; P(custo do challenger menor) = 99%.

## Decisão: PROMOVIDO

- AUPRC não inferior (IC inferior de Δ = +0.0024)
- Custo menor com probabilidade 99%
- Taxa de falsos positivos sem aumento relevante

Política: margem de não inferioridade de AUPRC 0.02; P(custo menor) ≥ 80%; aumento máximo de FPR 0.0005.

## Cuidados na leitura

- **Significância não é relevância.** A diferença de custo é de €42 em 8,9 h de tráfego. Antes de promover em produção, a política deve exigir também um efeito mínimo que compense o risco operacional da troca.
- **Reuso do teste.** Aqui o teste histórico serve de coorte de comparação. Usá-lo repetidamente para escolher modelos o transforma em validação. Em produção, a comparação usa a coorte coletada durante o período em sombra.
- **Promoção não é deploy.** A promoção fica registrada no registry; o artefato servido nesta entrega continua o champion original. Em produção, a promoção dispara o build de uma nova imagem.

## Histórico do registry

| Versão | Evento | Motivo |
|---|---|---|
| `20260924114649-4a06a7b0` | registered |  |
| `20260924114649-4a06a7b0` | stage:champion | modelo inicial em produção |
| `20260925122050-4a06a7b0` | registered |  |
| `20260925122050-4a06a7b0` | stage:shadow | challenger: teto de 300 árvores |
| `20260924114649-4a06a7b0` | archived | substituído por 20260925122050-4a06a7b0 |
| `20260925122050-4a06a7b0` | stage:champion | aprovado no portão de promoção |
