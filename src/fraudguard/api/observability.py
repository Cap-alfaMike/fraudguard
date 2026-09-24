"""Observabilidade: logs estruturados (JSON) e métricas Prometheus.

* Logs em JSON, uma linha por evento, prontos para Loki/ELK/Datadog.
  ``request_id`` via ContextVar correlaciona todos os logs de uma requisição.
* Nunca logamos o vetor de features nem identificadores de cliente:
  logs são o vazamento de dados mais comum em sistemas de pagamento (PCI-DSS).
* Métricas seguem RED (Rate, Errors, Duration) + métricas de modelo.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from contextvars import ContextVar

from prometheus_client import Counter, Gauge, Histogram

request_id_ctx: ContextVar[str] = ContextVar("request_id", default="-")

_RESERVED = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)) + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "request_id": request_id_ctx.get(),
        }
        payload.update({k: v for k, v in record.__dict__.items() if k not in _RESERVED})
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(level: str = "INFO", as_json: bool = True) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter() if as_json else logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())
    for noisy in ("uvicorn.access", "httpx", "httpx2", "httpcore"):
        logging.getLogger(noisy).setLevel("WARNING")


PREDICTIONS = Counter("fraudguard_predictions_total", "Predições realizadas", ["decision", "risk_level"])
LATENCY = Histogram(
    "fraudguard_inference_latency_seconds",
    "Latência de inferência (servidor)",
    buckets=(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)
PROBABILITY = Histogram(
    "fraudguard_fraud_probability",
    "Distribuição das probabilidades previstas",
    buckets=(0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 0.5, 0.8, 0.95, 1.0),
)
ERRORS = Counter("fraudguard_errors_total", "Erros por tipo", ["kind"])
ALERTS = Counter("fraudguard_ews_alerts_total", "Alertas emitidos pelo EWS", ["type", "severity"])
VALUE_BLOCKED = Counter("fraudguard_value_blocked_eur_total", "Valor (EUR) de transações marcadas como suspeitas")
MODEL_INFO = Gauge("fraudguard_model_info", "Modelo em produção", ["version", "name"])
THRESHOLD = Gauge("fraudguard_decision_threshold", "Limiar de decisão ativo")
