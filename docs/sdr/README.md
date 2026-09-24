# Solution Decision Records (SDR)

Cada decisão relevante tem um registro curto: contexto, decisão, alternativas descartadas e consequências. O formato segue os *Architecture Decision Records* de Michael Nygard. Registros não são reescritos; uma decisão revista gera um novo SDR que substitui o anterior.

| SDR | Decisão | Status |
|---|---|---|
| [001](SDR-001-metrica-primaria.md) | AUPRC como métrica de seleção; custo em EUR como métrica de decisão | Aceita |
| [002](SDR-002-split-temporal.md) | Deduplicação antes do split e split temporal 60/20/20 | Aceita |
| [003](SDR-003-modelo-e-desbalanceamento.md) | LightGBM; desbalanceamento tratado por seleção empírica de pesos, sem SMOTE | Aceita |
| [004](SDR-004-modelo-de-custo.md) | Calibração Platt e limiar que minimiza custo esperado dependente do valor | Aceita |
| [005](SDR-005-fastapi.md) | FastAPI com validação Pydantic estrita | Aceita |
| [006](SDR-006-llm-fora-do-caminho-critico.md) | LLM apenas para relatórios, fora do caminho crítico, com fallback | Aceita |
| [007](SDR-007-container.md) | Imagem multi-stage que treina no build; 1 processo por container | Aceita |
| [008](SDR-008-inferencia-compilada.md) | Caminho de inferência NumPy com paridade verificada no startup | Aceita |
| [009](SDR-009-early-warning-system.md) | EWS in-process com PSI e regras de ataque; agregação no Prometheus | Aceita |
