"""FraudGuard API — microsserviço de detecção de fraude em tempo real.

Escolha do FastAPI (docs/sdr/SDR-005-fastapi.md): validação Pydantic nativa,
OpenAPI automático (contrato versionável com consumidores), ASGI com
threadpool para trabalho CPU-bound e o melhor desempenho entre frameworks
Python maduros.

Os endpoints de inferência são funções ``def`` síncronas de propósito: a
inferência é CPU-bound e o FastAPI as executa no threadpool, sem bloquear o
event loop que atende /health e /metrics. LightGBM libera o GIL na predição.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from fraudguard import __version__
from fraudguard.api.observability import (
    ALERTS,
    ERRORS,
    LATENCY,
    MODEL_INFO,
    PREDICTIONS,
    PROBABILITY,
    THRESHOLD,
    VALUE_BLOCKED,
    configure_logging,
    request_id_ctx,
)
from fraudguard.api.schemas import (
    BatchRequest,
    BatchResponse,
    HealthResponse,
    PredictionResponse,
    ReportRequest,
    SimulationRequest,
    Transaction,
)
from fraudguard.config import RAW_FEATURES, Settings, get_settings
from fraudguard.ews.drift import MONITORED_FEATURES
from fraudguard.ews.llm_reporter import LLMReporter, build_facts
from fraudguard.ews.monitor import Alert, EarlyWarningSystem
from fraudguard.modeling.predictor import FraudPredictor

logger = logging.getLogger("fraudguard.api")
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


class AppState:
    """Estado por instância da aplicação (nunca compartilhado entre apps/testes)."""

    def __init__(self) -> None:
        self.predictor: FraudPredictor | None = None
        self.ews: EarlyWarningSystem | None = None
        self.reporter: LLMReporter | None = None
        self.metadata: dict = {}
        self.started_at: float = time.time()
        self.load_error: str | None = None


def _load(state: AppState, settings: Settings) -> None:
    try:
        state.predictor = FraudPredictor.load(settings.model_path)
        state.metadata = json.loads(settings.metadata_path.read_text()) if settings.metadata_path.exists() else {}
        reference = json.loads(settings.reference_path.read_text()) if settings.reference_path.exists() else None
        if reference is None:
            logger.warning("perfil de referência ausente: drift desabilitado")
        state.ews = EarlyWarningSystem(settings, reference, state.predictor.threshold)
        state.reporter = LLMReporter(settings)

        def on_alert(alert: Alert) -> None:
            ALERTS.labels(alert.type, alert.severity).inc()
            if alert.severity == "CRITICAL":
                state.reporter.generate_in_background(lambda: _facts(state), audience="operacoes")

        state.ews.subscribe(on_alert)
        MODEL_INFO.labels(state.predictor.version, state.predictor.model_name).set(1)
        THRESHOLD.set(state.predictor.threshold)
        state.load_error = None
        logger.info("serviço pronto", extra={"model_version": state.predictor.version, "llm_available": state.reporter.llm_available})
    except Exception as exc:  # readiness falha, liveness continua ok -> orquestrador não mata em loop
        state.load_error = f"{type(exc).__name__}: {exc}"
        logger.exception("falha ao carregar modelo")


def _facts(state: AppState) -> dict:
    return build_facts(state.ews.status(), state.metadata, state.ews.alerts(limit=10))


def _frame(transactions) -> np.ndarray:
    # Array NumPy direto: evita o custo de montar DataFrame no caminho quente
    return np.array([[getattr(t, c) for c in RAW_FEATURES] for t in transactions], dtype=float)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level, settings.log_json)
    state = AppState()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        _load(state, settings)
        yield
        logger.info("encerrando serviço")

    app = FastAPI(
        title="FraudGuard — Detecção de Fraude em Tempo Real",
        version=__version__,
        description="Scoring de risco com modelo calibrado, explicabilidade TreeSHAP e "
        "Early Warning System com relatórios inteligentes via LLM.",
        lifespan=lifespan,
    )
    app.state.fg = state
    app.state.settings = settings

    # ---------------------------------------------------------- middleware
    @app.middleware("http")
    async def request_context(request: Request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex
        token = request_id_ctx.set(rid[:64])
        t0 = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            request_id_ctx.reset(token)
        response.headers["x-request-id"] = rid[:64]
        response.headers["x-process-time-ms"] = f"{(time.perf_counter() - t0) * 1000:.3f}"
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError):
        ERRORS.labels("validation").inc()
        errors = [{"loc": e.get("loc"), "msg": e.get("msg"), "type": e.get("type")} for e in exc.errors()]
        logger.info("requisição inválida", extra={"path": request.url.path, "n_errors": len(errors)})
        return JSONResponse(status_code=422, content={"detail": errors})

    def _require_model() -> FraudPredictor:
        if state.predictor is None:
            ERRORS.labels("model_unavailable").inc()
            raise HTTPException(status_code=503, detail="Modelo indisponível")
        return state.predictor

    def _score(transactions, explain: bool) -> tuple[list[PredictionResponse], float]:
        predictor = _require_model()
        t0 = time.perf_counter()
        try:
            frame = _frame(transactions)
            preds = predictor.predict(frame, explain=explain)
        except Exception:
            ERRORS.labels("inference").inc()
            logger.exception("falha de inferência")
            raise HTTPException(status_code=500, detail="Falha interna na inferência") from None
        latency_ms = (time.perf_counter() - t0) * 1000
        per_item_ms = latency_ms / len(preds)
        out = []
        for tx, p in zip(transactions, preds, strict=True):
            decision = "SUSPEITO" if p.is_suspicious else "APROVADO"
            out.append(
                PredictionResponse(
                    transaction_id=tx.transaction_id,
                    fraud_probability=round(p.probability, 6),
                    decision=decision,
                    risk_level=p.risk_level,
                    threshold=round(predictor.threshold, 6),
                    model_version=predictor.version,
                    latency_ms=round(latency_ms, 3),
                    explanation=p.factors,
                )
            )
            PREDICTIONS.labels(decision, p.risk_level).inc()
            PROBABILITY.observe(p.probability)
            if p.is_suspicious:
                VALUE_BLOCKED.inc(tx.Amount)
            state.ews.record(p.probability, p.is_suspicious, tx.Amount, per_item_ms, {f: getattr(tx, f) for f in MONITORED_FEATURES})
            logger.info(
                "predição",
                extra={
                    "transaction_id": tx.transaction_id,
                    "fraud_probability": round(p.probability, 6),
                    "decision": decision,
                    "risk_level": p.risk_level,
                    "latency_ms": round(latency_ms, 3),
                    "model_version": predictor.version,
                },
            )
        LATENCY.observe(latency_ms / 1000)
        return out, latency_ms

    # ---------------------------------------------------------- saúde
    @app.get("/health", response_model=HealthResponse, tags=["saúde"])
    async def health():
        """Readiness: 200 somente se o modelo está carregado e pronto para servir."""
        ok = state.predictor is not None
        body = HealthResponse(
            status="ok" if ok else "unavailable",
            model_loaded=ok,
            model_version=state.predictor.version if ok else None,
            uptime_s=round(time.time() - state.started_at, 1),
        )
        return JSONResponse(status_code=200 if ok else 503, content=body.model_dump())

    @app.get("/health/live", tags=["saúde"])
    async def liveness():
        """Liveness: processo vivo e respondendo (não depende do modelo)."""
        return {"status": "alive"}

    # ---------------------------------------------------------- inferência
    @app.post("/predict", response_model=PredictionResponse, tags=["inferência"])
    def predict(tx: Transaction, explain: bool = Query(False, description="Incluir fatores TreeSHAP")):  # type: ignore[valid-type]
        results, _ = _score([tx], explain)
        return results[0]

    @app.post("/predict/batch", response_model=BatchResponse, tags=["inferência"])
    def predict_batch(req: BatchRequest, explain: bool = Query(False)):
        if len(req.transactions) > settings.max_batch_size:
            raise HTTPException(status_code=413, detail=f"Máximo de {settings.max_batch_size} transações por lote")
        results, latency_ms = _score(req.transactions, explain)
        return BatchResponse(
            model_version=state.predictor.version,
            count=len(results),
            suspicious_count=sum(r.decision == "SUSPEITO" for r in results),
            latency_ms=round(latency_ms, 3),
            results=results,
        )

    @app.get("/model/info", tags=["modelo"])
    async def model_info():
        _require_model()
        return {k: v for k, v in state.metadata.items() if k != "feature_names"}

    # ---------------------------------------------------------- EWS
    @app.get("/ews/status", tags=["early warning"])
    async def ews_status():
        _require_model()
        return state.ews.status()

    @app.get("/ews/alerts", tags=["early warning"])
    async def ews_alerts(limit: int = Query(50, ge=1, le=500), min_severity: str = Query("INFO", pattern="^(INFO|WARNING|CRITICAL)$")):
        _require_model()
        return state.ews.alerts(limit=limit, min_severity=min_severity)

    @app.post("/ews/report", tags=["early warning"])
    def ews_report(req: ReportRequest):
        """Relatório inteligente (LLM com fallback determinístico)."""
        _require_model()
        return state.reporter.generate(_facts(state), req.audience).to_dict()

    @app.get("/ews/report/latest", tags=["early warning"])
    async def ews_report_latest():
        _require_model()
        rep = state.reporter.last_report
        if rep is None:
            raise HTTPException(status_code=404, detail="Nenhum relatório gerado ainda")
        return rep.to_dict()

    @app.post("/ews/simulate", tags=["early warning"])
    def ews_simulate(req: SimulationRequest):
        """Injeta um cenário sintético pelo pipeline real (somente modo demo)."""
        if not settings.demo_mode:
            raise HTTPException(status_code=404, detail="Não encontrado")
        predictor = _require_model()
        from fraudguard.ews.simulator import scenario_frame

        frame = scenario_frame(req.scenario, req.n)
        t0 = time.perf_counter()
        probs = predictor.predict_proba(frame)
        per_item = (time.perf_counter() - t0) * 1000 / len(frame)
        fired = []
        for row, p in zip(frame.itertuples(index=False), probs, strict=True):
            susp = bool(p >= predictor.threshold)
            fired += state.ews.record(float(p), susp, float(row.Amount), per_item, {f: getattr(row, f) for f in MONITORED_FEATURES})
        return {
            "scenario": req.scenario,
            "n": len(frame),
            "suspicious": int((probs >= predictor.threshold).sum()),
            "alerts_fired": [a.to_dict() for a in fired],
        }

    @app.post("/ews/reset", tags=["early warning"])
    async def ews_reset():
        if not settings.demo_mode:
            raise HTTPException(status_code=404, detail="Não encontrado")
        _require_model()
        state.ews.reset()
        return {"status": "reset"}

    # ---------------------------------------------------------- ops / UI
    @app.get("/metrics", include_in_schema=False)
    async def metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/", include_in_schema=False)
    async def dashboard():
        return FileResponse(STATIC_DIR / "dashboard.html")

    return app


app = create_app()
