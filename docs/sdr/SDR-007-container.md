# SDR-007: Imagem multi-stage que treina no build; 1 processo por container

**Contexto.** A imagem precisa ser leve, segura e reprodutível, e subir com um único comando.

**Decisão.**
- **Três estágios.** `builder` compila wheels; `trainer` treina o modelo; `runtime` recebe só wheels e artefatos. Compiladores e cache de build não chegam à produção.
- **Treinar no build.** Código e modelo nascem juntos a partir do mesmo commit, o que garante reprodutibilidade (o treino é determinístico: mesmo fingerprint de dados, mesmas métricas, em ambientes diferentes). Se `data/raw/creditcard.csv` estiver no contexto, o modelo é treinado com os dados reais.
- **Segurança.** Base `python:3.12-slim`; usuário sem privilégios (UID 10001) e sem shell de login; filesystem somente leitura com `/tmp` em tmpfs; `cap_drop: ALL`; `no-new-privileges`; healthcheck em Python puro (sem instalar curl); scan Trivy no CI; `DEMO_MODE=false` por padrão na imagem.
- **Um processo uvicorn por container.** Escala-se por réplicas (Kubernetes HPA ou `docker compose --scale`). Motivos: o estado do EWS fica coerente por réplica; o orquestrador enxerga cada processo; a falha de um processo não degrada os outros silenciosamente. A visão global vem do Prometheus.

**Alternativas descartadas.** Gunicorn com vários workers por container: esconde processos do orquestrador e fragmenta o estado do EWS dentro do mesmo pod. Baixar o modelo de um registry no startup: é o caminho certo em escala (MLflow/S3), mas adiciona dependência externa ao boot; está no roadmap.

**Consequências.** O build leva cerca de um minuto a mais (treino). A imagem é imutável: trocar de modelo significa publicar uma nova imagem, o que facilita rollback e auditoria.
