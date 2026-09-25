# SDR-012: Kubernetes com Kustomize e hardening `restricted`

**Contexto.** Docker Compose cobre a execução local; produção exige alta disponibilidade, escala automática e isolamento.

**Decisão.** Manifestos em `deploy/k8s/` com Kustomize (base + overlay de produção), validados no CI com `kubeconform` em modo estrito contra os schemas oficiais do Kubernetes 1.31.

- **Disponibilidade:** 3 réplicas mínimas (6 em produção) espalhadas por zona e por nó; `maxUnavailable: 0` no rollout; PDB com `minAvailable: 2`; `preStop` de 5 s para drenar conexões.
- **Escala:** HPA por CPU a 60%, dobrando em 30 s na subida (picos e ataques) e descendo 20% por minuto após 5 min de estabilidade.
- **Recursos:** `requests` de CPU e memória; limite só de memória. Sem limite de CPU porque o throttling do CFS infla a cauda de latência, e o p99 é SLO.
- **Segurança:** namespace com Pod Security Admission `restricted`; não-root, seccomp `RuntimeDefault`, filesystem somente leitura, sem capabilities, sem token de service account; NetworkPolicy aceitando só o gateway e o monitoramento, e saída só DNS e HTTPS.
- **Probes:** startup e readiness em `/health` (depende do modelo), liveness em `/health/live` (não depende): um modelo ausente tira o pod do balanceador sem reiniciá-lo em loop.

**Alternativas descartadas.** Helm: mais poderoso, mas templating é desnecessário para um serviço; Kustomize é nativo no `kubectl`. Service mesh: mTLS e retries são valiosos em escala de plataforma, excessivos aqui.

**Consequências.** O `preStop` com `sleep` exige Kubernetes 1.30+. A tag da imagem no overlay deve ser trocada por digest no pipeline de deploy.
