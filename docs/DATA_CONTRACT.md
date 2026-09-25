# Contrato de dados: `POST /predict`

O contrato executável é o schema Pydantic (`api/schemas.py`), publicado em `/openapi.json`. Este documento registra a semântica e as regras de evolução.

| Campo | Tipo | Faixa | Semântica | Se violado |
|---|---|---|---|---|
| `transaction_id` | string opcional | 1–64, `[A-Za-z0-9_\-:.]` | ID opaco para rastreio; não é usado pelo modelo | 422 |
| `Time` | float | 0 a 1e10 | Segundos desde a referência; só a hora do dia é usada | 422 |
| `V1`–`V28` | float | ±250 | Componentes PCA; a matriz de projeção é propriedade do produtor | 422 |
| `Amount` | float | 0 a 1.000.000 | Valor em EUR | 422 |

Regras gerais: tipos estritos (sem coerção de string), campos extras rejeitados, NaN e infinito rejeitados.

## Evolução

- **Compatível** (versão menor): novos campos de resposta; novos endpoints.
- **Incompatível** (versão maior, `/v2/predict`): mudança de campo de entrada, de faixa ou da matriz PCA. A matriz PCA mudar invalida o modelo inteiro: exige retreino e é tratada como nova versão do contrato.
- **Monitoramento do contrato:** o alerta `FraudGuardValidationSpike` detecta produtores enviando dados fora do contrato; o PSI do EWS detecta mudanças de distribuição dentro do contrato.

## Responsáveis

| Papel | Responsabilidade |
|---|---|
| Produtor (sistema transacional) | Enviar dados no contrato; avisar mudanças com antecedência |
| Consumidor (FraudGuard) | Validar, rejeitar dados inválidos de forma visível, versionar mudanças |
