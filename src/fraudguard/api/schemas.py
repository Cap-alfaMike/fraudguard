"""Contratos da API (Pydantic v2).

Validação rigorosa por design (fail fast na borda):
* ``strict=True``: não converte "1.5" (string) em float — upstream com bug
  deve falhar alto, não ser silenciosamente aceito.
* ``extra="forbid"``: campos desconhecidos são rejeitados (detecta mudança de
  contrato do produtor).
* ``allow_inf_nan=False``: NaN/Infinity não entram no modelo.
* Limites físicos: ``Amount`` ∈ [0, 1e6]; componentes PCA em ±250 (os
  extremos do dataset real ficam em ~±120). Valores fora disso indicam
  corrupção, não fraude.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, create_model

from fraudguard.config import PCA_FEATURES

_STRICT = ConfigDict(strict=True, extra="forbid", allow_inf_nan=False)
PCAValue = Annotated[float, Field(ge=-250.0, le=250.0)]

_EXAMPLE = {
    "transaction_id": "tx-000123",
    "Time": 406.0,
    **{f: 0.0 for f in PCA_FEATURES},
    "Amount": 149.62,
}
_EXAMPLE.update({"V1": -1.3598, "V2": -0.0728, "V3": 2.5363, "V4": 1.3782, "V14": -0.3112, "V17": 0.2079})


class _TransactionBase(BaseModel):
    model_config = ConfigDict(**_STRICT, json_schema_extra={"example": _EXAMPLE})

    transaction_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9_\-:.]+$",
        description="Identificador opaco para rastreio (não é usado pelo modelo)",
    )
    Time: float = Field(ge=0, le=1e10, description="Segundos desde a referência (schema ULB)")
    Amount: float = Field(ge=0, le=1_000_000, description="Valor da transação em EUR")


Transaction = create_model(
    "Transaction",
    __base__=_TransactionBase,
    **{f: (PCAValue, Field(description=f"Componente {f} (PCA, anonimizada)")) for f in PCA_FEATURES},
)


class Factor(BaseModel):
    feature: str
    description: str
    contribution: float
    direction: Literal["aumenta risco", "reduz risco"]


class PredictionResponse(BaseModel):
    transaction_id: str | None
    fraud_probability: float = Field(ge=0, le=1)
    decision: Literal["APROVADO", "SUSPEITO"]
    risk_level: Literal["BAIXO", "MEDIO", "ALTO", "CRITICO"]
    threshold: float
    model_version: str
    latency_ms: float
    explanation: list[Factor] | None = None


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    transactions: list[Transaction] = Field(min_length=1)  # type: ignore[valid-type]


class BatchResponse(BaseModel):
    model_version: str
    count: int
    suspicious_count: int
    latency_ms: float
    results: list[PredictionResponse]


class HealthResponse(BaseModel):
    status: Literal["ok", "unavailable"]
    model_loaded: bool
    model_version: str | None
    uptime_s: float


class ReportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    audience: Literal["executivo", "tecnico", "operacoes"] = "executivo"


class SimulationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scenario: Literal["normal", "card_testing", "drift", "high_value"] = "normal"
    n: int = Field(default=300, ge=1, le=5000)
