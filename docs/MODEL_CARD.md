# Model Card: FraudGuard lgbm_unweighted

Formato inspirado em Mitchell et al. (2019), *Model Cards for Model Reporting*.

**Detalhes.** LightGBM (234 árvores após early stopping, `num_leaves=31`, `learning_rate=0.03`) com calibração Platt e limiar de 0,055. Versão `20260924114649-4a06a7b0`. Treino determinístico (semente 42).

**Uso pretendido.** Pontuar transações de cartão em tempo real e encaminhar as suspeitas para revisão ou autenticação adicional. **Não** é destinado a decisões automáticas e irreversíveis sobre clientes (encerramento de conta, restrição de crédito) sem revisão humana.

**Dados.** Schema do dataset ULB (Kaggle `mlg-ulb/creditcardfraud`): `Time`, `V1`–`V28` (componentes PCA anonimizadas), `Amount`. Nesta entrega, dados sintéticos com o mesmo perfil estatístico, porque o dataset real não estava acessível no ambiente de desenvolvimento. O pipeline usa o CSV real automaticamente quando presente.

**Features.** 28 componentes PCA, `log_amount`, hora do dia (seno e cosseno), `is_night`, `amount_is_zero`, `amount_is_micro`. `Time` bruto não é usado.

**Desempenho (teste temporal, nunca visto na seleção).** AUPRC 0,774; recall 75,5%; precisão 53,0%; taxa de falso positivo 0,111%; redução de custo de 76,5%.

**Explicabilidade.** Cada predição pode retornar os cinco fatores com maior contribuição TreeSHAP exata (em log-odds do modelo base). As features V são anonimizadas, então a explicação diz *quais* sinais pesaram, mas não o que eles significam no mundo real; o significado depende de quem tem a matriz de PCA original.

**Limitações.**
- Métricas em dados sintéticos; precisam ser revalidadas no CSV real.
- Apenas 64 fraudes na validação: o limiar e a calibração têm incerteza considerável.
- Sem features de comportamento por cartão, o modelo não enxerga padrões como "primeira compra no exterior".
- Janela de dados de cerca de dois dias: sem sazonalidade semanal ou mensal.

**Considerações éticas.** As componentes PCA não permitem auditar vieses por grupo demográfico. Em produção, com features interpretáveis, deve-se medir taxa de falso positivo por segmento (região, faixa de renda, idade da conta) para evitar que um grupo seja bloqueado desproporcionalmente. Falsos positivos têm custo real para as pessoas: por isso o nível `ALTO` deve preferir autenticação adicional a bloqueio.
