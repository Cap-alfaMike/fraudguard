"""Registro de modelos local e portão de promoção (docs/sdr/SDR-011-ciclo-de-vida.md).

Registro mínimo, sem servidor, com as garantias que importam para auditoria:
* cada versão é imutável e endereçada por conteúdo (SHA-256 do artefato);
* estágios explícitos: ``candidate`` -> ``shadow`` -> ``champion`` -> ``archived``;
* todo movimento de estágio fica num histórico append-only.

Em escala, a mesma interface mapeia para MLflow Model Registry ou SageMaker
Model Registry; o formato de manifest foi escolhido para facilitar essa migração.

O portão de promoção é estatístico, não "o número subiu": exige não
inferioridade em AUPRC, redução de custo com probabilidade alta e não
aumento relevante de falsos positivos, tudo medido por bootstrap PAREADO
na mesma coorte.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from fraudguard.modeling.uncertainty import PairedComparison

STAGES = ("candidate", "shadow", "champion", "archived")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class ModelRegistry:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "index.json"
        if not self.index_path.exists():
            self._write({"versions": {}, "history": []})

    def _read(self) -> dict:
        return json.loads(self.index_path.read_text())

    def _write(self, index: dict) -> None:
        tmp = self.index_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(index, indent=2, ensure_ascii=False))
        tmp.replace(self.index_path)  # escrita atômica

    def register(self, model_path: Path, metadata_path: Path, reference_path: Path | None = None) -> str:
        meta = json.loads(Path(metadata_path).read_text())
        version = meta["version"]
        index = self._read()
        digest = _sha256(Path(model_path))
        if version in index["versions"]:
            if index["versions"][version]["sha256"] != digest:
                raise ValueError(f"versão {version} já registrada com conteúdo diferente (imutabilidade)")
            return version
        target = self.root / version
        target.mkdir(parents=True, exist_ok=False)
        shutil.copy2(model_path, target / "model.joblib")
        shutil.copy2(metadata_path, target / "model_metadata.json")
        if reference_path and Path(reference_path).exists():
            shutil.copy2(reference_path, target / "reference_profile.json")
        index["versions"][version] = {
            "sha256": digest,
            "stage": "candidate",
            "model_name": meta.get("model_name"),
            "registered_at": time.time(),
            "test_auprc": meta.get("test_metrics", {}).get("auprc"),
            "data_fingerprint": meta.get("data", {}).get("fingerprint"),
        }
        index["history"].append({"ts": time.time(), "version": version, "event": "registered"})
        self._write(index)
        return version

    def set_stage(self, version: str, stage: str, reason: str = "") -> None:
        if stage not in STAGES:
            raise ValueError(f"estágio inválido: {stage}")
        index = self._read()
        if version not in index["versions"]:
            raise KeyError(version)
        if stage == "champion":  # só um champion por vez; o anterior é arquivado
            for v, info in index["versions"].items():
                if info["stage"] == "champion" and v != version:
                    info["stage"] = "archived"
                    index["history"].append(
                        {"ts": time.time(), "version": v, "event": "archived", "reason": f"substituído por {version}"}
                    )
        index["versions"][version]["stage"] = stage
        index["history"].append({"ts": time.time(), "version": version, "event": f"stage:{stage}", "reason": reason})
        self._write(index)

    def get(self, stage: str) -> str | None:
        matches = [v for v, i in self._read()["versions"].items() if i["stage"] == stage]
        return max(matches, key=lambda v: self._read()["versions"][v]["registered_at"]) if matches else None

    def path(self, version: str) -> Path:
        return self.root / version / "model.joblib"

    def verify(self, version: str) -> bool:
        """Confere se o artefato no disco é o mesmo que foi registrado."""
        return _sha256(self.path(version)) == self._read()["versions"][version]["sha256"]

    def history(self) -> list[dict]:
        return self._read()["history"]


@dataclass(frozen=True)
class GatePolicy:
    auprc_non_inferiority_margin: float = 0.02  # IC inferior de ΔAUPRC deve ser > -margem
    min_prob_cost_lower: float = 0.80  # P(custo do challenger < champion)
    max_fpr_increase: float = 0.0005  # +0,05 p.p. no máximo (IC superior)


@dataclass
class GateResult:
    passed: bool
    reasons: list[str] = field(default_factory=list)
    comparison: dict | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def promotion_gate(cmp: PairedComparison, policy: GatePolicy | None = None) -> GateResult:
    policy = policy or GatePolicy()
    reasons, ok = [], True
    if cmp.delta_auprc.low <= -policy.auprc_non_inferiority_margin:
        ok = False
        reasons.append(
            f"AUPRC: IC inferior de Δ ({cmp.delta_auprc.low:+.4f}) não descarta piora maior que {policy.auprc_non_inferiority_margin}"
        )
    else:
        reasons.append(f"AUPRC não inferior (IC inferior de Δ = {cmp.delta_auprc.low:+.4f})")
    if cmp.prob_cost_lower < policy.min_prob_cost_lower:
        ok = False
        reasons.append(f"Custo: P(challenger mais barato) = {cmp.prob_cost_lower:.0%} < {policy.min_prob_cost_lower:.0%}")
    else:
        reasons.append(f"Custo menor com probabilidade {cmp.prob_cost_lower:.0%}")
    if cmp.delta_fpr.high > policy.max_fpr_increase:
        ok = False
        reasons.append(f"FPR: IC superior de Δ ({cmp.delta_fpr.high:+.5f}) acima do limite {policy.max_fpr_increase}")
    else:
        reasons.append("Taxa de falsos positivos sem aumento relevante")
    return GateResult(passed=ok, reasons=reasons, comparison=cmp.to_dict())
