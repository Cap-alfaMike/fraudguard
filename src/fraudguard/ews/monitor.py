"""Early Warning System (EWS).

Monitora o fluxo de predições em uma janela deslizante e emite alertas
ANTES que um problema vire prejuízo. Três famílias de sinais:

1. Sinais de ATAQUE (negócio): pico da taxa de suspeitas vs. baseline,
   rajada de micro-transações suspeitas (card testing — prelúdio clássico
   de fraude em massa) e transação de alto valor com risco crítico.
2. Sinais de MODELO (ML): drift de predição (PSI dos scores) e drift de
   dados (PSI das features mais discriminativas). Drift de score é o
   indicador mais precoce disponível, pois rótulos de fraude chegam com
   dias/semanas de atraso (chargeback) — Dal Pozzolo et al., 2018.
3. Sinais de SISTEMA (SRE): violação do SLO de latência p99.

Custo por requisição: O(1) (append em deque + contadores). As regras de
janela (PSI, percentis) rodam a cada ``eval_every`` eventos, amortizando o
custo fora do caminho quente.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import asdict, dataclass, field

import numpy as np

from fraudguard.config import Settings
from fraudguard.ews.drift import MONITORED_FEATURES, bin_proportions, psi

logger = logging.getLogger("fraudguard.ews")

SEVERITY_RANK = {"INFO": 0, "WARNING": 1, "CRITICAL": 2}


def _br(x: float, dec: int = 2) -> str:
    """Formatação numérica pt-BR (1.234,56) para mensagens a stakeholders."""
    return f"{x:,.{dec}f}".replace(",", "\x00").replace(".", ",").replace("\x00", ".")


def _pct(x: float, dec: int = 2) -> str:
    return _br(100 * x, dec) + "%"


@dataclass
class Alert:
    type: str
    severity: str
    title: str
    message: str
    metrics: dict
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Event:
    ts: float
    probability: float
    suspicious: bool
    amount: float
    latency_ms: float
    features: tuple[float, ...]


class EarlyWarningSystem:
    def __init__(self, settings: Settings, reference: dict | None, threshold: float, eval_every: int = 100):
        self.s = settings
        self.reference = reference or {}
        self.threshold = threshold
        self.eval_every = eval_every
        self._events: deque[Event] = deque(maxlen=settings.ews_window_size)
        self._micro_suspicious: deque[float] = deque(maxlen=1000)
        self._alerts: deque[Alert] = deque(maxlen=500)
        self._last_fired: dict[str, float] = {}
        self._listeners: list[Callable[[Alert], None]] = []
        self._lock = threading.Lock()
        self.total_events = 0
        self.total_suspicious = 0
        self.value_blocked = 0.0
        self.expected_loss_avoided = 0.0
        self._last_snapshot: dict = {}

    # ------------------------------------------------------------ API pública
    def subscribe(self, callback: Callable[[Alert], None]) -> None:
        self._listeners.append(callback)

    def record(self, probability: float, suspicious: bool, amount: float, latency_ms: float, features: dict) -> list[Alert]:
        ev = Event(
            time.time(), probability, suspicious, amount, latency_ms, tuple(float(features.get(f, np.nan)) for f in MONITORED_FEATURES)
        )
        fired: list[Alert] = []
        with self._lock:
            self._events.append(ev)
            self.total_events += 1
            if suspicious:
                self.total_suspicious += 1
                self.value_blocked += amount
                self.expected_loss_avoided += probability * amount
            fired += self._point_rules(ev)
            if self.total_events % self.eval_every == 0:
                fired += self._window_rules()
        for alert in fired:
            self._notify(alert)
        return fired

    def status(self) -> dict:
        with self._lock:
            self._window_rules(emit=False)
            snap = dict(self._last_snapshot)
            snap.update(
                {
                    "total_events": self.total_events,
                    "total_suspicious": self.total_suspicious,
                    "value_blocked_eur": round(self.value_blocked, 2),
                    "expected_loss_avoided_eur": round(self.expected_loss_avoided, 2),
                    "active_alerts": [a.to_dict() for a in list(self._alerts)[-10:]][::-1],
                    "health": self._health(),
                }
            )
        return snap

    def alerts(self, limit: int = 50, min_severity: str = "INFO") -> list[dict]:
        rank = SEVERITY_RANK[min_severity]
        with self._lock:
            items = [a for a in self._alerts if SEVERITY_RANK[a.severity] >= rank]
        return [a.to_dict() for a in items[-limit:]][::-1]

    def reset(self) -> None:
        with self._lock:
            self._events.clear()
            self._micro_suspicious.clear()
            self._alerts.clear()
            self._last_fired.clear()
            self._last_snapshot = {}
            self.total_events = self.total_suspicious = 0
            self.value_blocked = self.expected_loss_avoided = 0.0

    # ---------------------------------------------------------------- regras
    def _point_rules(self, ev: Event) -> list[Alert]:
        out = []
        if ev.suspicious and ev.amount >= self.s.ews_high_value_amount and ev.probability >= 0.5:
            out += self._fire(
                Alert(
                    "HIGH_VALUE_FRAUD_ATTEMPT",
                    "CRITICAL",
                    "Tentativa de fraude de alto valor bloqueada",
                    f"Transação de €{_br(ev.amount)} com probabilidade de fraude {_pct(ev.probability, 1)}.",
                    {"amount": ev.amount, "probability": round(ev.probability, 4)},
                ),
                cooldown=60,
            )
        if ev.suspicious and 0 < ev.amount <= self.s.ews_card_testing_amount:
            self._micro_suspicious.append(ev.ts)
            recent = sum(1 for t in self._micro_suspicious if ev.ts - t <= 60)
            if recent >= self.s.ews_card_testing_burst:
                out += self._fire(
                    Alert(
                        "CARD_TESTING_BURST",
                        "CRITICAL",
                        "Padrão de teste de cartão detectado",
                        f"{recent} micro-transações suspeitas (até €{_br(self.s.ews_card_testing_amount, 0)}) no último minuto. "
                        "Padrão típico de validação de cartões roubados antes de fraude em massa.",
                        {"micro_suspicious_last_60s": recent},
                    )
                )
        return out

    def _window_rules(self, emit: bool = True) -> list[Alert]:
        n = len(self._events)
        if n == 0:
            self._last_snapshot = {"window_size": 0}
            return []
        probs = np.fromiter((e.probability for e in self._events), float, n)
        lat = np.fromiter((e.latency_ms for e in self._events), float, n)
        susp_rate = float(np.mean(probs >= self.threshold))
        ref_score = self.reference.get("score", {})
        base_rate = float(ref_score.get("suspicious_rate", 0.0))
        p50, p95, p99 = (float(x) for x in np.percentile(lat, [50, 95, 99]))

        score_psi = None
        feature_psi: dict[str, float] = {}
        if ref_score.get("edges") and n >= self.s.ews_min_samples:
            score_psi = psi(ref_score["proportions"], bin_proportions(probs, ref_score["edges"]))
            feats = np.array([e.features for e in self._events])
            for j, name in enumerate(MONITORED_FEATURES):
                ref = self.reference.get("features", {}).get(name)
                if ref:
                    feature_psi[name] = round(psi(ref["proportions"], bin_proportions(feats[:, j], ref["edges"])), 4)

        self._last_snapshot = {
            "window_size": n,
            "suspicious_rate": round(susp_rate, 6),
            "baseline_suspicious_rate": round(base_rate, 6),
            "mean_probability": round(float(probs.mean()), 6),
            "latency_ms": {"p50": round(p50, 3), "p95": round(p95, 3), "p99": round(p99, 3)},
            "latency_slo_ms": self.s.latency_slo_ms,
            "score_psi": None if score_psi is None else round(score_psi, 4),
            "feature_psi": feature_psi,
        }
        if not emit or n < self.s.ews_min_samples:
            return []

        out: list[Alert] = []
        if base_rate > 0 and susp_rate >= self.s.ews_rate_spike_factor * base_rate:
            factor = susp_rate / base_rate
            out += self._fire(
                Alert(
                    "SUSPICIOUS_RATE_SPIKE",
                    "CRITICAL" if factor >= 2 * self.s.ews_rate_spike_factor else "WARNING",
                    "Pico na taxa de transações suspeitas",
                    f"Taxa de suspeitas em {_pct(susp_rate)} ({_br(factor, 1)}× o baseline de {_pct(base_rate)}). "
                    "Possível ataque coordenado ou mudança de comportamento.",
                    {"suspicious_rate": round(susp_rate, 6), "baseline": round(base_rate, 6), "factor": round(factor, 2)},
                )
            )
        if score_psi is not None and score_psi >= self.s.ews_psi_warning:
            sev = "CRITICAL" if score_psi >= self.s.ews_psi_critical else "WARNING"
            out += self._fire(
                Alert(
                    "PREDICTION_DRIFT",
                    sev,
                    "Distribuição dos scores mudou",
                    f"PSI dos scores = {_br(score_psi, 3)}. O modelo está vendo um perfil de risco diferente do treino.",
                    {"score_psi": round(score_psi, 4)},
                )
            )
        drifted = {k: v for k, v in feature_psi.items() if v >= self.s.ews_psi_critical}
        if drifted:
            out += self._fire(
                Alert(
                    "DATA_DRIFT",
                    "WARNING",
                    "Drift nas variáveis de entrada",
                    f"Variáveis com PSI ≥ {_br(self.s.ews_psi_critical)}: {', '.join(sorted(drifted))}. "
                    "Avaliar retreino e checar mudanças no upstream.",
                    {"feature_psi": drifted},
                )
            )
        if p99 > self.s.latency_slo_ms:
            out += self._fire(
                Alert(
                    "LATENCY_SLO_BREACH",
                    "WARNING",
                    "Latência p99 acima do SLO",
                    f"p99 = {_br(p99, 1)} ms (SLO {_br(self.s.latency_slo_ms, 0)} ms). Verificar saturação de CPU/réplicas.",
                    {"p99_ms": round(p99, 2)},
                )
            )
        return out

    # --------------------------------------------------------------- helpers
    def _fire(self, alert: Alert, cooldown: float | None = None) -> list[Alert]:
        now = time.time()
        cd = self.s.ews_alert_cooldown_s if cooldown is None else cooldown
        if now - self._last_fired.get(alert.type, 0.0) < cd:
            return []
        self._last_fired[alert.type] = now
        self._alerts.append(alert)
        logger.warning("ews_alert", extra={"alert": alert.to_dict()})
        return [alert]

    def _notify(self, alert: Alert) -> None:
        for cb in self._listeners:
            try:
                cb(alert)
            except Exception:  # listener nunca derruba o fluxo de predição
                logger.exception("falha em listener do EWS")

    def _health(self) -> str:
        recent = [a for a in self._alerts if time.time() - a.created_at < 900]
        if any(a.severity == "CRITICAL" for a in recent):
            return "VERMELHO"
        if any(a.severity == "WARNING" for a in recent):
            return "AMARELO"
        return "VERDE"
