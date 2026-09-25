# Política de segurança

## Reportar uma vulnerabilidade

Não abra issue pública. Use o *Private vulnerability reporting* do GitHub (aba Security → Report a vulnerability). Resposta inicial em até 3 dias úteis.

## Práticas adotadas

- Container não-root, filesystem somente leitura, sem capabilities, `no-new-privileges`, seccomp `RuntimeDefault`.
- Kubernetes com Pod Security Admission `restricted`, NetworkPolicy de entrada e saída, sem token de service account.
- Scan de vulnerabilidades da imagem (Trivy) no CI; Dependabot para dependências, imagens e actions.
- Segredos só por `Secret`/variável de ambiente; nunca no repositório (`detect-private-key` no pre-commit).
- Dados: logs sem features nem dados do cliente; o LLM recebe apenas agregados.
- Modelo de ameaças: [`docs/THREAT_MODEL.md`](docs/THREAT_MODEL.md).
