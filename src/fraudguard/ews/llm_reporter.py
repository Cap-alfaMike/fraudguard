"""Smart reporting com LLM para o Early Warning System.

Decisões de AI Engineering (docs/sdr/SDR-006-llm-fora-do-caminho-critico.md):

* O LLM NUNCA decide aprovar/negar. A decisão é do modelo calibrado, que
  é determinístico, auditável e responde em milissegundos. O LLM traduz
  sinais técnicos em narrativa acionável para cada público.
* Fora do caminho crítico: relatórios são gerados sob demanda ou em
  background após alerta crítico. Latência/indisponibilidade do provedor
  jamais afetam o /predict.
* Privacidade: o prompt recebe apenas AGREGADOS (taxas, PSI, somas),
  nunca dados de transação individual ou identificadores (LGPD/PCI-DSS).
* Grounding: temperatura 0, instrução para usar somente os números
  fornecidos, e verificação pós-geração que mede a fração de números do
  texto rastreáveis aos fatos (``grounding_ratio``).
* Resiliência: timeout, e fallback para relatório determinístico por
  template — o sistema degrada com elegância e nunca fica sem relatório.
"""

from __future__ import annotations

import contextlib
import json
import logging
import re
import threading
import time
from dataclasses import asdict, dataclass

from fraudguard.config import Settings

logger = logging.getLogger("fraudguard.llm")

AUDIENCES = {
    "executivo": (
        "Diretoria e C-level. Foque em impacto financeiro, risco reputacional da marca "
        "(confiança do cliente na segurança das transações) e decisões necessárias. "
        "Linguagem de negócio, sem jargão de ML."
    ),
    "tecnico": (
        "Times de ML Engineering e SRE. Foque em drift (PSI), calibração, latência, "
        "causas prováveis e passos de diagnóstico e mitigação."
    ),
    "operacoes": (
        "Analistas de prevenção à fraude. Foque em padrões de ataque observados, fila de revisão e ações imediatas de contenção."
    ),
}

SYSTEM_PROMPT = """Você é o analista sênior de risco de fraude de um banco digital.
Escreva um relatório do Early Warning System em português do Brasil.

Regras invioláveis:
1. Use SOMENTE os números presentes no JSON de fatos. Não invente, não extrapole.
   Se algo não estiver nos fatos, escreva "não disponível".
2. Valores monetários em EUR, com o prefixo €.
3. Deixe explícito quando um número é ESTIMATIVA (ex.: perdas evitadas estimadas).
4. Não inclua dados pessoais nem sugira decisões sobre transações individuais.
5. Seja conciso: no máximo ~350 palavras.

Estrutura (títulos em markdown ##):
## Situação
## O que os sinais indicam
## Impacto financeiro e na marca
## Ações recomendadas (priorizadas, com responsável sugerido)
## Confiabilidade desta análise"""

_NUM = re.compile(r"-?\d+(?:[.,]\d+)*")


@dataclass
class Report:
    audience: str
    source: str  # "llm" | "template"
    content: str
    generated_at: float
    grounding_ratio: float | None
    model: str | None
    latency_ms: float

    def to_dict(self) -> dict:
        return asdict(self)


def _candidates(num: str) -> list[float]:
    """Interpretações numéricas possíveis de um token (pt-BR e en-US).

    "39.623" pode ser 39,623 (en) ou 39.623 (pt, milhar); "1.234,56" é pt;
    "1,234.56" é en. Retornamos todas as leituras plausíveis.
    """
    s = num.strip()
    out: list[str] = []
    if "," in s and "." in s:
        out.append(s.replace(".", "").replace(",", ".") if s.rfind(",") > s.rfind(".") else s.replace(",", ""))
    elif "," in s:
        out += [s.replace(",", "."), s.replace(",", "")]
    elif s.count(".") > 1:
        out.append(s.replace(".", ""))
    elif "." in s:
        out += [s, s.replace(".", "")]
    else:
        out.append(s)
    vals = []
    for c in out:
        with contextlib.suppress(ValueError):
            vals.append(float(c))
    return vals


def _normalize(num: str) -> float | None:
    c = _candidates(num)
    return c[0] if c else None


def _fact_numbers(obj) -> list[float]:
    out: list[float] = []
    if isinstance(obj, dict):
        for v in obj.values():
            out += _fact_numbers(v)
    elif isinstance(obj, list):
        for v in obj:
            out += _fact_numbers(v)
    elif isinstance(obj, bool):
        pass
    elif isinstance(obj, int | float):
        out += [float(obj), float(obj) * 100]  # aceita taxa expressa em %
    elif isinstance(obj, str):
        for m in _NUM.findall(obj):
            out += _candidates(m)
    return out


def grounding_ratio(text: str, facts: dict) -> float:
    """Fração de números do texto (exceto triviais 0–10, que costumam ser
    numeração/contagens) que batem com algum número dos fatos (tolerância 1%)."""
    refs = _fact_numbers(facts)
    found = [c for c in (_candidates(m) for m in _NUM.findall(text)) if c and max(abs(v) for v in c) > 10]
    if not found:
        return 1.0

    def matches(v: float) -> bool:
        return any(abs(v - r) <= max(0.01 * abs(r), 0.05) for r in refs)

    ok = sum(1 for cands in found if any(matches(v) for v in cands))
    return round(ok / len(found), 3)


def build_facts(status: dict, model_info: dict, alerts: list[dict]) -> dict:
    """Somente agregados — nenhum dado transacional individual."""
    tb = model_info.get("test_business", {})
    tm = model_info.get("test_metrics", {})
    return {
        "janela_monitorada": {
            "transacoes_na_janela": status.get("window_size"),
            "total_transacoes_processadas": status.get("total_events"),
            "total_suspeitas": status.get("total_suspicious"),
            "taxa_suspeitas_atual": status.get("suspicious_rate"),
            "taxa_suspeitas_baseline": status.get("baseline_suspicious_rate"),
            "psi_scores": status.get("score_psi"),
            "psi_features": status.get("feature_psi"),
            "latencia_ms": status.get("latency_ms"),
            "slo_latencia_ms": status.get("latency_slo_ms"),
            "valor_bloqueado_eur": status.get("value_blocked_eur"),
            "perda_evitada_estimada_eur": status.get("expected_loss_avoided_eur"),
            "saude": status.get("health"),
        },
        "alertas_recentes": [{"tipo": a["type"], "severidade": a["severity"], "mensagem": a["message"]} for a in alerts[:8]],
        "modelo": {
            "versao": model_info.get("version"),
            "auprc_teste": tm.get("auprc"),
            "recall_teste": tm.get("recall"),
            "precisao_teste": tm.get("precision"),
            "taxa_falso_positivo_teste": tm.get("false_positive_rate"),
            "economia_teste_eur": tb.get("savings"),
            "economia_teste_pct": tb.get("savings_pct"),
            "horas_do_periodo_de_teste": tb.get("period_hours"),
        },
    }


def template_report(facts: dict, audience: str) -> str:
    j, m, alerts = facts["janela_monitorada"], facts["modelo"], facts["alertas_recentes"]

    def br(x: float, dec: int = 2) -> str:
        return f"{x:,.{dec}f}".replace(",", "\x00").replace(".", ",").replace("\x00", ".")

    def pct(x):
        return "não disponível" if x is None else br(100 * x, 3) + "%"

    lines = [
        "## Situação",
        f"Saúde do sistema: **{j.get('saude', 'não disponível')}**. "
        f"{j.get('total_transacoes_processadas') or 0} transações avaliadas, "
        f"{j.get('total_suspeitas') or 0} marcadas como suspeitas.",
        "",
        "## O que os sinais indicam",
        f"Taxa atual de suspeitas: {pct(j.get('taxa_suspeitas_atual'))} "
        f"(baseline: {pct(j.get('taxa_suspeitas_baseline'))}). "
        f"PSI dos scores: {br(j['psi_scores'], 3) if j.get('psi_scores') is not None else 'não disponível'}.",
    ]
    if alerts:
        lines += [f"- [{a['severidade']}] {a['tipo']}: {a['mensagem']}" for a in alerts]
    else:
        lines.append("Nenhum alerta ativo.")
    lines += [
        "",
        "## Impacto financeiro e na marca",
        f"Valor retido para revisão: €{br(j.get('valor_bloqueado_eur') or 0)}. "
        f"Perda evitada (estimativa ponderada pela probabilidade): €{br(j.get('perda_evitada_estimada_eur') or 0)}. "
        + (
            f"Em teste offline, o modelo reduziu em {br(m['economia_teste_pct'], 1)}% o custo total de fraude."
            if m.get("economia_teste_pct") is not None
            else "Economia offline não disponível."
        ),
        "Cada fraude evitada preserva a confiança do cliente na segurança da plataforma.",
        "",
        "## Ações recomendadas",
    ]
    types = {a["tipo"] for a in alerts}
    actions = []
    if {"CARD_TESTING_BURST", "SUSPICIOUS_RATE_SPIKE", "HIGH_VALUE_FRAUD_ATTEMPT"} & types:
        actions.append("Operações: acionar protocolo de contenção e priorizar fila de revisão.")
    if {"PREDICTION_DRIFT", "DATA_DRIFT"} & types:
        actions.append("ML: investigar upstream e avaliar retreino com dados recentes rotulados.")
    if "LATENCY_SLO_BREACH" in types:
        actions.append("SRE: escalar réplicas e verificar saturação de CPU.")
    actions = actions or ["Manter monitoramento de rotina."]
    lines += [f"{i}. {a}" for i, a in enumerate(actions, 1)]
    lines += [
        "",
        "## Confiabilidade desta análise",
        f"Relatório determinístico gerado por template (LLM indisponível ou desativado). Público: {audience}.",
    ]
    return "\n".join(lines)


class LLMReporter:
    def __init__(self, settings: Settings):
        self.s = settings
        self._client = None
        self._lock = threading.Lock()
        self._last_auto = 0.0
        self.last_report: Report | None = None
        if settings.llm_enabled and settings.anthropic_api_key:
            try:
                import anthropic

                self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key, timeout=settings.llm_timeout_s, max_retries=1)
            except Exception:  # pragma: no cover - dependência opcional
                logger.exception("cliente Anthropic indisponível; usando template")

    @property
    def llm_available(self) -> bool:
        return self._client is not None

    def generate(self, facts: dict, audience: str = "executivo") -> Report:
        if audience not in AUDIENCES:
            raise ValueError(f"Público inválido: {audience}. Opções: {sorted(AUDIENCES)}")
        t0 = time.perf_counter()
        if self._client is not None:
            try:
                resp = self._client.messages.create(
                    model=self.s.llm_model,
                    max_tokens=self.s.llm_max_tokens,
                    temperature=0,
                    system=SYSTEM_PROMPT,
                    messages=[
                        {
                            "role": "user",
                            "content": f"Público: {AUDIENCES[audience]}\n\nFatos (JSON):\n"
                            f"{json.dumps(facts, ensure_ascii=False, indent=2)}",
                        }
                    ],
                )
                text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip()
                if text:
                    report = Report(
                        audience,
                        "llm",
                        text,
                        time.time(),
                        grounding_ratio(text, facts),
                        self.s.llm_model,
                        round((time.perf_counter() - t0) * 1000, 1),
                    )
                    if report.grounding_ratio is not None and report.grounding_ratio < 0.8:
                        logger.warning("relatório LLM com baixo grounding", extra={"grounding_ratio": report.grounding_ratio})
                    self.last_report = report
                    return report
            except Exception as exc:
                logger.warning("falha no LLM; fallback para template", extra={"error": type(exc).__name__})
        text = template_report(facts, audience)
        report = Report(
            audience, "template", text, time.time(), grounding_ratio(text, facts), None, round((time.perf_counter() - t0) * 1000, 1)
        )
        self.last_report = report
        return report

    def generate_in_background(self, facts_factory, audience: str = "operacoes", min_interval_s: float = 300) -> bool:
        """Dispara geração assíncrona (ex.: após alerta crítico), com rate limit."""
        with self._lock:
            if time.time() - self._last_auto < min_interval_s:
                return False
            self._last_auto = time.time()
        threading.Thread(target=lambda: self.generate(facts_factory(), audience), daemon=True).start()
        return True
