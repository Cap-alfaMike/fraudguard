"""Configuração central do FraudGuard.

Todas as constantes de negócio (custos, SLOs, limiares do EWS) vivem aqui e
podem ser sobrescritas por variáveis de ambiente com prefixo ``FRAUDGUARD_``.
Isso mantém o código livre de "números mágicos" e torna cada premissa
auditável (ver docs/sdr/SDR-004-modelo-de-custo.md).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Schema do dataset ULB (Kaggle mlg-ulb/creditcardfraud)
PCA_FEATURES: list[str] = [f"V{i}" for i in range(1, 29)]
RAW_FEATURES: list[str] = ["Time", *PCA_FEATURES, "Amount"]
TARGET: str = "Class"


class CostModel(BaseModel):
    """Modelo de custo *example-dependent* (Bahnsen et al., 2014/2016).

    Custo de um falso negativo (fraude aprovada) depende do valor da transação;
    custo de um falso positivo (cliente legítimo bloqueado) é quase fixo, mas
    inclui o risco de churn por atrito — o componente de *dano à marca*.
    Valores em EUR (moeda do dataset). São PREMISSAS e devem ser calibradas
    com Finanças/Risco antes de produção.
    """

    review_cost: float = Field(3.0, description="Custo operacional de revisar/contatar o cliente")
    chargeback_fee: float = Field(15.0, description="Taxa de chargeback + custo de disputa")
    customer_lifetime_value: float = Field(400.0, description="CLV médio de um cliente")
    churn_prob_after_fraud: float = Field(0.05, description="Prob. de churn após o cliente sofrer fraude não detectada (dano à marca)")
    churn_prob_after_false_decline: float = Field(0.01, description="Prob. de churn após bloqueio indevido (atrito)")

    @property
    def false_positive_cost(self) -> float:
        return self.review_cost + self.churn_prob_after_false_decline * self.customer_lifetime_value

    def false_negative_cost(self, amount):  # aceita escalar ou array
        return amount + self.chargeback_fee + self.churn_prob_after_fraud * self.customer_lifetime_value

    @property
    def true_positive_cost(self) -> float:
        # Fraude bloqueada ainda custa a revisão, mas evita a perda.
        return self.review_cost


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FRAUDGUARD_", env_file=".env", extra="ignore")

    # Artefatos
    artifacts_dir: Path = PROJECT_ROOT / "artifacts"
    model_filename: str = "model.joblib"
    metadata_filename: str = "model_metadata.json"
    reference_filename: str = "reference_profile.json"
    raw_data_path: Path = PROJECT_ROOT / "data" / "raw" / "creditcard.csv"

    # API / SLO
    latency_slo_ms: float = 50.0
    max_batch_size: int = 1000
    log_level: str = "INFO"
    log_json: bool = True
    demo_mode: bool = True  # habilita /ews/simulate (desligar em produção)

    # Early Warning System
    ews_window_size: int = 2000
    ews_min_samples: int = 200
    ews_rate_spike_factor: float = 3.0
    ews_psi_warning: float = 0.10
    ews_psi_critical: float = 0.25
    ews_alert_cooldown_s: float = 300.0
    ews_high_value_amount: float = 1000.0
    ews_card_testing_amount: float = 2.0
    ews_card_testing_burst: int = 10

    # LLM (fora do caminho crítico da predição)
    llm_enabled: bool = True
    llm_model: str = "claude-sonnet-5"
    llm_timeout_s: float = 15.0
    llm_max_tokens: int = 1200
    anthropic_api_key: str | None = Field(default=None, validation_alias="ANTHROPIC_API_KEY")

    cost: CostModel = CostModel()

    @property
    def model_path(self) -> Path:
        return self.artifacts_dir / self.model_filename

    @property
    def metadata_path(self) -> Path:
        return self.artifacts_dir / self.metadata_filename

    @property
    def reference_path(self) -> Path:
        return self.artifacts_dir / self.reference_filename


@lru_cache
def get_settings() -> Settings:
    return Settings()
