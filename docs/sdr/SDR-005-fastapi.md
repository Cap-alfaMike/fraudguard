# SDR-005: FastAPI com validação Pydantic estrita

**Contexto.** O requisito dá preferência ao FastAPI. A API é a fronteira de confiança: dados ruins que entram viram decisões ruins silenciosas.

**Decisão.** FastAPI + Pydantic v2, com:
- `strict=True`: `"10.0"` (string) é rejeitado; um produtor com bug deve falhar de forma visível.
- `extra="forbid"`: campos desconhecidos são rejeitados, detectando mudanças de contrato no upstream (e evitando que alguém envie, por exemplo, nome do cliente por engano).
- `allow_inf_nan=False` e limites físicos (`Amount` ∈ [0, 10⁶], componentes em ±250). Valores fora disso indicam corrupção, não fraude.
- Endpoints de inferência como `def` síncronos: o FastAPI os executa em threadpool, então a inferência (CPU) não bloqueia o event loop que responde `/health` e `/metrics`.
- Readiness (`/health`, 503 sem modelo) separado de liveness (`/health/live`): um modelo ausente tira a réplica do balanceador sem que o orquestrador a reinicie em loop.

**Alternativas descartadas.** Flask: sem validação nem OpenAPI nativos, WSGI. gRPC: melhor para comunicação interna de altíssimo volume, mas eleva a barreira de integração; pode ser adicionado depois sobre o mesmo núcleo (`FraudPredictor`). Servidores de modelo dedicados (BentoML, Seldon, KServe): ótimos em escala de plataforma, excessivos para um único modelo tabular.

**Consequências.** Contrato OpenAPI gerado automaticamente em `/docs`, versionável com os consumidores. O custo de validação (~0,1 ms) é pequeno frente ao benefício.
