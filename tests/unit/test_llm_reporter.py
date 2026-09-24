from types import SimpleNamespace

import pytest

from fraudguard.config import Settings
from fraudguard.ews.llm_reporter import LLMReporter, build_facts, grounding_ratio, template_report

STATUS = {
    "window_size": 1000,
    "total_events": 5000,
    "total_suspicious": 42,
    "suspicious_rate": 0.0084,
    "baseline_suspicious_rate": 0.0021,
    "score_psi": 0.31,
    "feature_psi": {"V14": 0.4},
    "latency_ms": {"p99": 1.2},
    "latency_slo_ms": 50,
    "value_blocked_eur": 12345.67,
    "expected_loss_avoided_eur": 9876.5,
    "health": "VERMELHO",
}
ALERTS = [{"type": "CARD_TESTING_BURST", "severity": "CRITICAL", "message": "12 micro-transações"}]
META = {"version": "v1", "test_metrics": {"auprc": 0.77, "recall": 0.75}, "test_business": {"savings_pct": 76.5}}


@pytest.fixture
def facts():
    return build_facts(STATUS, META, ALERTS)


def test_facts_contain_only_aggregates(facts):
    flat = str(facts)
    assert "transaction_id" not in flat and "V1'" not in flat
    assert facts["janela_monitorada"]["valor_bloqueado_eur"] == 12345.67


def test_template_report_covers_sections_and_actions(facts):
    text = template_report(facts, "executivo")
    for section in ("## Situação", "## Impacto financeiro", "## Ações recomendadas"):
        assert section in text
    assert "contenção" in text and "retreino" not in text  # só ações pertinentes aos alertas


def test_template_is_fully_grounded(facts):
    assert grounding_ratio(template_report(facts, "tecnico"), facts) == 1.0


def test_grounding_flags_hallucinated_numbers(facts):
    assert grounding_ratio("Perdemos €98.765,43 e 777 clientes.", facts) == 0.0
    assert grounding_ratio("Valor bloqueado €12.345,67.", facts) == 1.0


def test_fallback_to_template_without_api_key(facts):
    rep = LLMReporter(Settings(anthropic_api_key=None)).generate(facts, "executivo")
    assert rep.source == "template" and rep.content


def test_invalid_audience(facts):
    with pytest.raises(ValueError):
        LLMReporter(Settings()).generate(facts, "marketing")


class _FakeMessages:
    def __init__(self, text=None, exc=None):
        self.text, self.exc, self.calls = text, exc, []

    def create(self, **kw):
        self.calls.append(kw)
        if self.exc:
            raise self.exc
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=self.text)])


def test_llm_path_uses_temperature_zero_and_no_pii(facts):
    rep = LLMReporter(Settings(anthropic_api_key=None))
    fake = _FakeMessages(text="## Situação\nValor bloqueado €12.345,67.")
    rep._client = SimpleNamespace(messages=fake)
    out = rep.generate(facts, "operacoes")
    assert out.source == "llm" and out.grounding_ratio == 1.0
    assert fake.calls[0]["temperature"] == 0
    assert "tx-" not in fake.calls[0]["messages"][0]["content"]


def test_llm_failure_degrades_to_template(facts):
    rep = LLMReporter(Settings(anthropic_api_key=None))
    rep._client = SimpleNamespace(messages=_FakeMessages(exc=TimeoutError()))
    assert rep.generate(facts).source == "template"


def test_background_generation_is_rate_limited(facts):
    rep = LLMReporter(Settings(anthropic_api_key=None))
    assert rep.generate_in_background(lambda: facts) is True
    assert rep.generate_in_background(lambda: facts) is False


@pytest.mark.parametrize("text", ["Retidos €12.345,67.", "Retidos €12,345.67.", "Perda evitada de €9.876,50."])
def test_grounding_handles_ptbr_and_en_formats(facts, text):
    assert grounding_ratio(text, facts) == 1.0


def test_grounding_handles_ambiguous_thousands():
    facts = {"valor": 39623.4}
    assert grounding_ratio("Foram €39.623 retidos.", facts) == 1.0
