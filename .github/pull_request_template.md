## O que muda e por quê

## Tipo
- [ ] Correção
- [ ] Nova funcionalidade
- [ ] Mudança de modelo ou de limiar (exige seção abaixo)
- [ ] Infraestrutura / plataforma

## Checklist
- [ ] `make lint` e `make test` passam localmente
- [ ] Testes novos cobrem o comportamento alterado
- [ ] Contrato da API inalterado, ou mudança versionada e comunicada aos consumidores
- [ ] Documentação e SDR atualizados quando a decisão muda

## Se mudar o modelo ou o limiar
- [ ] `scripts/champion_challenger.py` executado; portão de promoção aprovado
- [ ] `scripts/evaluate_uncertainty.py` executado; ICs anexados
- [ ] Testes comportamentais passam (`pytest tests/behavioral`)
- [ ] Período em modo sombra planejado antes da promoção
