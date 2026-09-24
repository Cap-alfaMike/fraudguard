# syntax=docker/dockerfile:1.7
# =============================================================================
# FraudGuard: imagem multi-stage (ver docs/sdr/SDR-007-container.md)
#   1. builder  -> compila wheels (ferramentas de build NÃO vão para produção)
#   2. trainer  -> treina o modelo DENTRO do build: imagem = código + modelo
#                  versionados juntos, reprodutível a partir do commit
#   3. runtime  -> python slim, usuário não-root, sem compiladores, sem shell tools
# =============================================================================
ARG PYTHON_VERSION=3.12

FROM python:${PYTHON_VERSION}-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /build
COPY requirements.txt .
RUN pip wheel --wheel-dir /wheels -r requirements.txt

FROM python:${PYTHON_VERSION}-slim AS trainer
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONPATH=/app/src
# libgomp1: runtime OpenMP exigido pelo LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /wheels /wheels
RUN pip install --no-index --find-links=/wheels /wheels/*
WORKDIR /app
COPY src/ src/
# Se data/raw/creditcard.csv (Kaggle) existir no contexto, é usado; senão, dados sintéticos
COPY data/ data/
ARG TRAIN_ARGS="--no-plots"
RUN python -m fraudguard.modeling.train ${TRAIN_ARGS}

FROM python:${PYTHON_VERSION}-slim AS runtime
LABEL org.opencontainers.image.title="fraudguard" \
      org.opencontainers.image.description="Detecção de fraude em tempo real com Early Warning System" \
      org.opencontainers.image.licenses="MIT"
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PIP_NO_CACHE_DIR=1 \
    FRAUDGUARD_ARTIFACTS_DIR=/app/artifacts \
    FRAUDGUARD_DEMO_MODE=false
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --no-create-home --shell /usr/sbin/nologin app
COPY --from=builder /wheels /wheels
RUN pip install --no-index --find-links=/wheels /wheels/* && rm -rf /wheels
WORKDIR /app
COPY --chown=app:app src/ src/
COPY --from=trainer --chown=app:app /app/artifacts/ artifacts/
USER 10001:10001
EXPOSE 8000
# Probe sem curl (menos superfície de ataque): usa o próprio Python
HEALTHCHECK --interval=15s --timeout=3s --start-period=20s --retries=3 \
  CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).status == 200 else 1)"]
# 1 processo por container: o estado do EWS fica coerente por réplica; escala-se
# horizontalmente (réplicas), agregando métricas no Prometheus. Ver SDR-007.
CMD ["uvicorn", "fraudguard.api.main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", "--no-access-log", "--timeout-keep-alive", "5", "--limit-concurrency", "512"]
