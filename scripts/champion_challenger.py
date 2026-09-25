"""Ciclo champion/challenger de ponta a ponta.

1. Registra o modelo atual (champion) no registry.
2. Treina um challenger (por padrão: variante com teto de 300 árvores, modo --fast).
3. Compara os dois na MESMA partição de teste por bootstrap pareado.
4. Aplica o portão de promoção e registra a decisão.

Uso: python scripts/champion_challenger.py
Saída: reports/CHAMPION_CHALLENGER.md; registry em artifacts/registry/
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from fraudguard.config import Settings, get_settings
from fraudguard.modeling.evaluation import load_eval_set
from fraudguard.modeling.registry import GatePolicy, ModelRegistry, promotion_gate
from fraudguard.modeling.train import train
from fraudguard.modeling.uncertainty import paired_bootstrap

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    s = get_settings()
    reg = ModelRegistry(s.artifacts_dir / "registry")
    champ = reg.register(s.model_path, s.metadata_path, s.reference_path)
    if reg.get("champion") is None:
        reg.set_stage(champ, "champion", "modelo inicial em produção")

    with tempfile.TemporaryDirectory() as tmp:
        cs = Settings(artifacts_dir=Path(tmp), llm_enabled=False)
        train(settings=cs, fast=True, make_plots=False)
        chall = reg.register(cs.model_path, cs.metadata_path, cs.reference_path)
    reg.set_stage(chall, "shadow", "challenger: teto de 300 árvores")

    ev_champ = load_eval_set(s, model_path=reg.path(champ))
    ev_chall_p = load_eval_set(s, model_path=reg.path(chall))
    cmp = paired_bootstrap(
        ev_champ.y,
        ev_champ.proba,
        ev_chall_p.proba,
        ev_champ.amount,
        ev_champ.predictor.threshold,
        ev_chall_p.predictor.threshold,
        s.cost,
        n_boot=1000,
    )
    policy = GatePolicy()
    gate = promotion_gate(cmp, policy)
    if gate.passed:
        reg.set_stage(chall, "champion", "aprovado no portão de promoção")
    else:
        reg.set_stage(chall, "archived", "reprovado no portão: " + "; ".join(gate.reasons))

    d = cmp.to_dict()
    fmt = lambda x, n=4: f"{x:+.{n}f}".replace(".", ",")
    lines = [
        "# Champion × challenger",
        "",
        f"- Champion: `{champ}` ({ev_champ.predictor.model_name}, limiar {ev_champ.predictor.threshold:.4f})".replace(".", ",", 1),
        f"- Challenger: `{chall}` (teto de 300 árvores, limiar {ev_chall_p.predictor.threshold:.4f})".replace(".", ",", 1),
        "- Comparação: bootstrap pareado por blocos temporais na mesma partição de teste, 1.000 réplicas.",
        "",
        "| Diferença (challenger − champion) | Pontual | IC 95% |",
        "|---|---|---|",
        f"| AUPRC (maior é melhor) | {fmt(d['delta_auprc']['point'])} | "
        f"{fmt(d['delta_auprc']['low'])} a {fmt(d['delta_auprc']['high'])} |",
        f"| Custo em EUR (menor é melhor) | {fmt(d['delta_cost_eur']['point'], 0)} | "
        f"{fmt(d['delta_cost_eur']['low'], 0)} a {fmt(d['delta_cost_eur']['high'], 0)} |",
        f"| Taxa de falso positivo (menor é melhor) | {fmt(d['delta_fpr']['point'], 5)} | "
        f"{fmt(d['delta_fpr']['low'], 5)} a {fmt(d['delta_fpr']['high'], 5)} |",
        "",
        f"P(AUPRC do challenger maior) = {d['prob_auprc_better']:.0%}; P(custo do challenger menor) = {d['prob_cost_lower']:.0%}.",
        "",
        f"## Decisão: {'PROMOVIDO' if gate.passed else 'REPROVADO'}",
        "",
        *[f"- {r}" for r in gate.reasons],
        "",
        f"Política: margem de não inferioridade de AUPRC {policy.auprc_non_inferiority_margin}; "
        f"P(custo menor) ≥ {policy.min_prob_cost_lower:.0%}; aumento máximo de FPR {policy.max_fpr_increase}.",
        "",
        "## Cuidados na leitura",
        "",
        f"- **Significância não é relevância.** A diferença de custo é de €{-d['delta_cost_eur']['point']:.0f} em "
        f"{str(round(ev_champ.metadata['test_business']['period_hours'], 1)).replace('.', ',')} h de tráfego. Antes de "
        "promover em produção, a política deve exigir também um efeito mínimo que compense o risco operacional da troca.",
        "- **Reuso do teste.** Aqui o teste histórico serve de coorte de comparação. Usá-lo repetidamente para escolher "
        "modelos o transforma em validação. Em produção, a comparação usa a coorte coletada durante o período em sombra.",
        "- **Promoção não é deploy.** A promoção fica registrada no registry; o artefato servido nesta entrega continua o "
        "champion original. Em produção, a promoção dispara o build de uma nova imagem.",
        "",
        "## Histórico do registry",
        "",
        "| Versão | Evento | Motivo |",
        "|---|---|---|",
        *[f"| `{h['version']}` | {h['event']} | {h.get('reason', '')} |" for h in reg.history()],
    ]
    (ROOT / "reports" / "CHAMPION_CHALLENGER.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (ROOT / "reports" / "champion_challenger.json").write_text(json.dumps(gate.to_dict(), indent=2, ensure_ascii=False))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
