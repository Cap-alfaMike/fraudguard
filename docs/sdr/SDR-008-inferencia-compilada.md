# SDR-008: Inferência compilada em NumPy com paridade verificada

**Contexto.** O profiling mostrou que o pipeline sklearn com saída pandas custava **6,9 ms** por chamada unitária, enquanto a predição do LightGBM custava **0,03 ms**. O overhead era do empacotamento em DataFrames, não do modelo.

**Decisão.** No carregamento, extrair os parâmetros aprendidos (medianas do imputador, centro e escala do RobustScaler, árvores do booster, coeficientes Platt) e aplicá-los com NumPy puro. O feature engineering é **uma única função** (`engineer_array`) chamada tanto pelo transformer do sklearn (treino) quanto pelo caminho rápido (produção), o que elimina divergência entre treino e serviço por construção.

Salvaguardas:
- No startup, 512 entradas aleatórias com caudas pesadas e 2% de valores ausentes são pontuadas pelos dois caminhos. Se a diferença máxima passar de 1e-9, o serviço usa o pipeline sklearn e registra o evento. A diferença medida foi **0,0**.
- Um bug real foi pego por essa verificação durante o desenvolvimento: o sklearn alimenta o calibrador com o log-odds (`decision_function`), não com a probabilidade. O caminho rápido agora replica essa escolha.
- Testes cobrem a paridade e o fallback.

**Resultado.** Predição unitária: 8,6 ms → **0,17 ms** (cerca de 50×). Com explicação TreeSHAP: 1,9 ms.

**Alternativas descartadas.** ONNX/Treelite: ganho similar, mas adiciona toolchain e uma segunda representação do modelo a validar. Manter o sklearn: ~9 ms por transação é aceitável, mas desperdiça CPU numa plataforma de alto volume.

**Consequências.** O pipeline sklearn continua sendo a fonte da verdade e o artefato exportado; o caminho rápido é uma otimização verificável e descartável.
