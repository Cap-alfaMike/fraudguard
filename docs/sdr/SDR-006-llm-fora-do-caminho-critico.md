# SDR-006: LLM só para relatórios, fora do caminho crítico

**Contexto.** O desafio pede um EWS com LLM para *smart reporting*. LLMs têm latência de segundos, são não determinísticos e podem inventar números. Decisões de crédito e fraude precisam ser auditáveis e explicáveis a reguladores.

**Decisão.**
1. **O LLM nunca decide.** Aprovar ou sinalizar é papel do modelo calibrado. O LLM traduz sinais técnicos em narrativa e ações para três públicos: diretoria, prevenção à fraude e ML/SRE.
2. **Fora do caminho crítico.** Relatórios são gerados sob demanda (`POST /ews/report`) ou em background após alerta crítico, com limite de uma geração automática a cada 5 minutos. Indisponibilidade do provedor jamais afeta o `/predict`.
3. **Privacidade por construção.** O prompt recebe apenas agregados (taxas, PSI, somas, contagens). Nenhuma transação individual nem identificador sai do serviço (LGPD, PCI-DSS). Há teste que verifica isso.
4. **Grounding.** Temperatura 0; instrução para usar somente os números fornecidos; e uma verificação pós-geração que extrai os números do texto (em formatos pt-BR e en-US) e mede a fração rastreável aos fatos (`grounding_ratio`). Abaixo de 80%, o evento é registrado em log.
5. **Degradação elegante.** Timeout de 15 s, uma nova tentativa, e fallback para um relatório determinístico por template. Sem chave de API, o sistema funciona integralmente com o template.

**Alternativas descartadas.** LLM como classificador ou "segunda opinião" em tempo real: latência incompatível, custo por transação alto, sem ganho sobre um modelo tabular treinado. Enviar exemplos de transações suspeitas ao LLM para análise: risco de privacidade sem benefício proporcional.

**Consequências.** O modelo padrão (`claude-sonnet-5`) é configurável por variável de ambiente. O `grounding_ratio` é uma heurística, não uma prova; relatórios para decisões de alto impacto continuam exigindo leitura humana.
