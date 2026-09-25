# Changelog

Formato: [Keep a Changelog](https://keepachangelog.com/pt-BR/1.1.0/). Versionamento semântico.

## [1.1.0]
Somente adições; nenhum comportamento existente foi alterado.

### Adicionado
- Intervalos de confiança por block bootstrap temporal (`modeling/uncertainty.py`, `scripts/evaluate_uncertainty.py`).
- Registro de modelos imutável por SHA-256 e portão de promoção champion/challenger com bootstrap pareado (`modeling/registry.py`, `scripts/champion_challenger.py`).
- Modo sombra na API (`FRAUDGUARD_SHADOW_MODEL_PATH`, `GET /model/shadow`), desligado por padrão.
- Monitoramento com rótulos atrasados e ECE por coorte (`ews/batch_monitor.py`, `scripts/batch_monitor.py`).
- Testes comportamentais do modelo (`tests/behavioral/`) e políticas de plataforma como código.
- Manifestos Kubernetes com Kustomize (HPA, PDB, NetworkPolicy, PSA `restricted`).
- SLOs com alertas de burn rate em múltiplas janelas; dashboard Grafana provisionado.
- Workflows `platform` e `ml-quality`; Dependabot, pre-commit, CODEOWNERS, templates.
- Documentos: SLO, modelo de ameaças, contrato de dados, SDRs 010 a 013.
- Mockup de produto da fila de revisão do analista (`docs/product/case-review-mock.html`).
- Figuras em qualidade de apresentação geradas dos dados reais do pipeline (`scripts/make_presentation_figures.py`).

## [1.0.0]
- Versão inicial: pipeline de ML, API, Early Warning System, relatórios via LLM, container, CI e documentação.
