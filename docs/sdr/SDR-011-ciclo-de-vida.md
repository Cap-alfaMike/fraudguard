# SDR-011: Registry imutável, modo sombra e portão de promoção estatístico

**Contexto.** Trocar o modelo de um sistema de fraude é uma mudança de alto risco: um modelo pior custa dinheiro, e um modelo mais agressivo bloqueia clientes. "O número subiu no notebook" não é critério.

**Decisão.** Ciclo em quatro estágios:

1. **Registry** (`modeling/registry.py`): versões imutáveis endereçadas por SHA-256; estágios `candidate → shadow → champion → archived`; histórico append-only; escrita atômica do índice. A interface espelha a do MLflow Model Registry para facilitar a migração.
2. **Modo sombra** (`FRAUDGUARD_SHADOW_MODEL_PATH`): o challenger pontua cada transação em paralelo, a comparação vai para logs e métricas (`fraudguard_shadow_*`) e para `GET /model/shadow`, e **a resposta nunca muda**. Falhas do challenger são isoladas (testado).
3. **Portão de promoção** (`promotion_gate`), sobre bootstrap pareado na mesma coorte:
   - não inferioridade de AUPRC: IC inferior de Δ > −0,02;
   - custo menor com probabilidade ≥ 80%;
   - aumento de falsos positivos: IC superior de Δ ≤ +0,05 p.p.
4. **Promoção = nova imagem**, com rollback por troca de tag.

**Resultado da demonstração** (`reports/CHAMPION_CHALLENGER.md`): um challenger com teto de 300 árvores foi **promovido** (ΔAUPRC +0,0075 [+0,0024; +0,0136]; Δcusto −€42 [−84; −7]). O relatório registra três ressalvas que fazem parte da decisão:
- **Significância não é relevância.** €42 em 8,9 h é pouco. A próxima versão da política deve exigir um efeito mínimo.
- **Reuso do teste.** Em produção, a comparação usa a coorte do período em sombra, não o teste histórico.
- **Promoção não é deploy.** O artefato servido continua o original até a nova imagem ser publicada.

**Alternativas descartadas.** A/B test com tráfego real decidindo: expõe clientes ao challenger antes de haver evidência; sombra primeiro, A/B depois se necessário. Comparar métricas pontuais: sujeito a promover ruído.

**Consequências.** Promoções são auditáveis (quem, quando, por quê, com que evidência). O custo é tempo: pelo menos uma semana de sombra e a espera pela maturação dos rótulos (SDR-009, camada 3).
