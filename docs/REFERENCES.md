# Referências

> **Aviso.** Estas referências foram citadas de memória, sem acesso a uma base bibliográfica no momento da escrita. Títulos, anos e veículos devem ser conferidos antes de qualquer uso formal.

## Artigos

- Bahnsen, A. C., Stojanovic, A., Aouada, D., & Ottersten, B. (2014). Improving credit card fraud detection with calibrated probabilities. *SIAM International Conference on Data Mining*. (SDR-004)
- Bahnsen, A. C., Aouada, D., Stojanovic, A., & Ottersten, B. (2016). Feature engineering strategies for credit card fraud detection. *Expert Systems with Applications*, 51. (features de tempo cíclicas)
- Breck, E., Cai, S., Nielsen, E., Salib, M., & Sculley, D. (2017). The ML Test Score: A rubric for ML production readiness and technical debt reduction. *IEEE Big Data*. (estratégia de testes)
- Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic minority over-sampling technique. *Journal of Artificial Intelligence Research*, 16. (alternativa descartada, SDR-003)
- Dal Pozzolo, A., Caelen, O., Le Borgne, Y.-A., Waterschoot, S., & Bontempi, G. (2014). Learned lessons in credit card fraud detection from a practitioner perspective. *Expert Systems with Applications*, 41(10).
- Dal Pozzolo, A., Caelen, O., Johnson, R. A., & Bontempi, G. (2015). Calibrating probability with undersampling for unbalanced classification. *IEEE Symposium Series on Computational Intelligence*. (origem do dataset; SDR-003)
- Dal Pozzolo, A., Boracchi, G., Caelen, O., Alippi, C., & Bontempi, G. (2018). Credit card fraud detection: A realistic modeling and a novel learning strategy. *IEEE Transactions on Neural Networks and Learning Systems*, 29(8). (atraso de rótulos; SDR-009)
- Elkan, C. (2001). The foundations of cost-sensitive learning. *IJCAI*. (SDR-004)
- Gama, J., Žliobaitė, I., Bifet, A., Pechenizkiy, M., & Bouchachia, A. (2014). A survey on concept drift adaptation. *ACM Computing Surveys*, 46(4). (MONITORING)
- Ke, G., et al. (2017). LightGBM: A highly efficient gradient boosting decision tree. *NeurIPS*. (SDR-003)
- Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *NeurIPS*.
- Lundberg, S. M., et al. (2020). From local explanations to global understanding with explainable AI for trees. *Nature Machine Intelligence*, 2. (TreeSHAP)
- Mitchell, M., et al. (2019). Model cards for model reporting. *FAccT*. (MODEL_CARD)
- Niculescu-Mizil, A., & Caruana, R. (2005). Predicting good probabilities with supervised learning. *ICML*. (calibração, SDR-004)
- Rabanser, S., Günnemann, S., & Lipton, Z. (2019). Failing loudly: An empirical study of methods for detecting dataset shift. *NeurIPS*. (MONITORING)
- Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLOS ONE*, 10(3). (SDR-001)
- Sculley, D., et al. (2015). Hidden technical debt in machine learning systems. *NeurIPS*. (arquitetura)

## Livros e guias

- Le Borgne, Y.-A., Siblini, W., Lebichot, B., & Bontempi, G. (2022). *Reproducible Machine Learning for Credit Card Fraud Detection: Practical Handbook*. Université Libre de Bruxelles. (SDR-002, split temporal)
- Siddiqi, N. (2006). *Credit Risk Scorecards*. Wiley. (PSI, SDR-009)
- Nygard, M. (2011). *Documenting Architecture Decisions* (post). (formato dos SDRs)

## Repositórios

- `Fraud-Detection-Handbook/fraud-detection-handbook`: código do handbook da ULB; referência para validação temporal e métricas.
- `microsoft/LightGBM`: modelo.
- `evidentlyai/evidently` e `SeldonIO/alibi-detect`: detecção de drift (recomendados para a camada batch).
- `fastapi/full-stack-fastapi-template`: práticas de estrutura para serviços FastAPI.
- `feast-dev/feast`: feature store (roadmap).

## Dados

- Kaggle, `mlg-ulb/creditcardfraud`: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
