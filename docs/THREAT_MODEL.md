# Modelo de ameaças (STRIDE)

Escopo: serviço de scoring, EWS, integração com LLM e pipeline de treino.

| Ameaça | Exemplo no FraudGuard | Mitigação | Residual |
|---|---|---|---|
| **S**poofing | Chamador não autorizado consulta scores para calibrar ataques | NetworkPolicy: só o gateway entra; autenticação no gateway | Depende do gateway |
| **T**ampering | Artefato de modelo trocado | Registry com SHA-256 e verificação; imagem imutável; filesystem somente leitura | Supply chain do build (assinar imagens com cosign é o próximo passo) |
| **T**ampering | *Data poisoning* no retreino (rótulos manipulados) | Portão de promoção; testes comportamentais; sombra antes de promover | Envenenamento lento e sutil exige auditoria de rótulos |
| **R**epudiation | "O modelo bloqueou minha transação sem motivo" | Log estruturado por decisão com versão, probabilidade e `request_id`; explicações TreeSHAP | — |
| **I**nformation disclosure | Features ou dados do cliente em logs ou no prompt do LLM | Logs sem features; LLM só recebe agregados (testado) | — |
| **I**nformation disclosure | *Model extraction* por consultas repetidas | Rate limiting no gateway; a API não expõe os fatores por padrão | Mitigação parcial |
| **D**enial of service | Rajada de requisições ou payloads enormes | `--limit-concurrency`, limite de lote (1.000), HPA, validação estrita | — |
| **E**levation of privilege | Escape do container | Não-root, sem capabilities, seccomp, PSA `restricted`, sem token de service account | — |
| Ameaça específica de ML | *Evasão adversarial*: fraudador ajusta a transação para ficar abaixo do limiar | EWS de drift e de ataque; retreino periódico; limiar não é público | Ameaça permanente; é a razão do monitoramento contínuo |
| Ameaça específica de LLM | *Prompt injection* via dados | O prompt só contém números e tipos de alerta gerados pelo próprio sistema; a saída é escapada no dashboard | — |
