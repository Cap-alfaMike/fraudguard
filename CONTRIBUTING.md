# Como contribuir

```bash
pip install -r requirements-dev.txt
pre-commit install          # lint e formatação a cada commit
make test                   # 160+ testes, cobertura mínima de 85%
```

**Convenções.** Commits no formato *Conventional Commits* (`feat:`, `fix:`, `ci:`, `docs:`). Toda decisão técnica relevante ganha um SDR em `docs/sdr/`; SDRs não são reescritos, são substituídos por um novo.

**Mudanças de modelo** seguem o ciclo do [SDR-011](docs/sdr/SDR-011-ciclo-de-vida.md): treino → testes comportamentais → portão de promoção com bootstrap pareado → modo sombra → promoção. O template de PR traz o checklist.

**Mudanças de contrato da API** exigem nova versão e comunicação aos consumidores: o OpenAPI em `/openapi.json` é o contrato.
