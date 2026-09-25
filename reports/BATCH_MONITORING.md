# Monitoramento com rótulos atrasados (camada 3)

Partição de teste tratada como produção, em coortes de 3 horas pela data da transação. Rótulos simulados: sinalizadas são revisadas em ~1 h; fraudes aprovadas só aparecem no chargeback, em ~72 h (exponencial). Maturação exigida: 96 h.

## Consulta 6 horas após o fim do período

|   cohort |   transactions |   age_h | mature   |   frauds_true |   frauds_labeled |   label_coverage | status   |
|---------:|---------------:|--------:|:---------|--------------:|-----------------:|-----------------:|:---------|
|        0 |          29030 |    11.9 | False    |            38 |               32 |            0.842 | IMATURA  |
|        3 |          19206 |     8.9 | False    |            31 |               24 |            0.774 | IMATURA  |
|        6 |           8509 |     6   | False    |            25 |               18 |            0.72  | IMATURA  |

## Consulta 2 dias após o fim do período

|   cohort |   transactions |   age_h | mature   |   frauds_true |   frauds_labeled |   label_coverage | status   |
|---------:|---------------:|--------:|:---------|--------------:|-----------------:|-----------------:|:---------|
|        0 |          29030 |    53.9 | False    |            38 |               36 |            0.947 | IMATURA  |
|        3 |          19206 |    50.9 | False    |            31 |               24 |            0.774 | IMATURA  |
|        6 |           8509 |    48   | False    |            25 |               20 |            0.8   | IMATURA  |

## Consulta 5 dias após o fim do período

|   cohort |   transactions |   age_h | mature   |   frauds_true |   frauds_labeled |   label_coverage |   auprc |   recall |      ece |   realized_cost_eur | status   |
|---------:|---------------:|--------:|:---------|--------------:|-----------------:|-----------------:|--------:|---------:|---------:|--------------------:|:---------|
|        0 |          29030 |   125.9 | True     |            38 |               38 |            1     |  0.816  |   0.8158 | 8.1e-05  |              570.42 | OK       |
|        3 |          19206 |   122.9 | True     |            31 |               28 |            0.903 |  0.8112 |   0.7857 | 9.4e-05  |              815.46 | OK       |
|        6 |           8509 |   120   | True     |            25 |               22 |            0.88  |  0.7878 |   0.8182 | 0.000156 |              554.72 | OK       |

## Consulta 15 dias após o fim do período

|   cohort |   transactions |   age_h | mature   |   frauds_true |   frauds_labeled |   label_coverage |   auprc |   recall |      ece |   realized_cost_eur | status   |
|---------:|---------------:|--------:|:---------|--------------:|-----------------:|-----------------:|--------:|---------:|---------:|--------------------:|:---------|
|        0 |          29030 |   365.9 | True     |            38 |               38 |                1 |  0.816  |   0.8158 | 8.1e-05  |              570.42 | OK       |
|        3 |          19206 |   362.9 | True     |            31 |               31 |                1 |  0.7554 |   0.7097 | 0.000177 |              927.93 | OK       |
|        6 |           8509 |   360   | True     |            25 |               25 |                1 |  0.7324 |   0.72   | 0.000487 |              835.2  | OK       |

**Leitura.**
- Seis horas após o período, o recall aparente é 0,959 contra 0,755 real: as fraudes que o modelo pegou são rotuladas em minutos, as que ele perdeu só aparecem no chargeback. Por isso nenhuma métrica é calculada antes da maturação.
- Mesmo após 96 h a cobertura não é completa. Na coorte 6, o recall medido cai de 0,818 (5 dias) para 0,720 (15 dias), quando todos os rótulos chegam. O p95 do atraso de chargeback nesta simulação é 10,5 dias: a janela de maturação deve ser calibrada por esse percentil, não pela média, e a cobertura sempre é publicada ao lado.
- Coortes `DEGRADADO` (AUPRC mais de 10% abaixo da referência) ou `DESCALIBRADO` (ECE acima do limite) disparam investigação, não retreino automático.
