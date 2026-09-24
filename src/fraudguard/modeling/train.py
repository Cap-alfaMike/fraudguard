"""Treino end-to-end reprodutível.

Fluxo (cada etapa justificada em docs/SDD.md §4):

1. Carga (CSV real ou sintético) -> validação de schema -> deduplicação.
2. Split temporal: treino (60%) | validação (20%) | teste (20%).
   O fim do treino (15%) é separado para early stopping — assim a validação
   fica limpa para seleção/calibração/limiar e o teste fica intocado.
3. Candidatos: LogReg balanceada (baseline interpretável) e LightGBM com três
   estratégias de desbalanceamento. Seleção pela AUPRC na validação.
4. Calibração Platt (sigmoid) na validação: pesos de classe inflam as
   probabilidades; a decisão por custo exige probabilidades calibradas
   (Niculescu-Mizil & Caruana, 2005; Dal Pozzolo et al., 2015).
   Sigmoid em vez de isotônica: ~100 fraudes na validação tornam a isotônica
   propensa a overfitting.
5. Limiar que minimiza custo esperado (EUR) na validação.
6. Avaliação final ÚNICA no teste + artefatos versionados.

Uso:  python -m fraudguard.modeling.train [--data caminho.csv] [--fast]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import time
from datetime import UTC, datetime
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline

from fraudguard.config import RAW_FEATURES, TARGET, Settings, get_settings
from fraudguard.data.loader import deduplicate, load_dataset, temporal_split
from fraudguard.data.synthetic import generate_synthetic_creditcard
from fraudguard.ews.drift import build_reference_profile
from fraudguard.features.transformers import build_preprocessor
from fraudguard.modeling.metrics import (
    business_cost,
    classification_report_at,
    optimize_threshold,
    ranking_metrics,
)

logger = logging.getLogger("fraudguard.train")
SEED = 42


def _lgbm(scale_pos_weight: float, n_estimators: int) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=n_estimators,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=40,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.7,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        random_state=SEED,
        n_jobs=-1,
        verbose=-1,
    )


def _fit_candidate(name, X_fit, y_fit, X_es, y_es, ratio, fast: bool):
    """Ajusta pré-processador + classificador SOMENTE em X_fit."""
    pre = build_preprocessor().fit(X_fit)
    Xt, Xes = pre.transform(X_fit), pre.transform(X_es)
    if name == "logreg_balanced":
        clf = LogisticRegression(class_weight="balanced", max_iter=2000, C=0.1, random_state=SEED)
        clf.fit(Xt, y_fit)
    else:
        spw = {"lgbm_unweighted": 1.0, "lgbm_sqrt_weight": float(np.sqrt(ratio)), "lgbm_full_weight": float(ratio)}[name]
        clf = _lgbm(spw, n_estimators=300 if fast else 3000)
        clf.fit(
            Xt,
            y_fit,
            eval_X=(Xes,),
            eval_y=(y_es,),
            eval_metric="average_precision",
            callbacks=[lgb.early_stopping(150, first_metric_only=False, verbose=False)],
        )
    return Pipeline([("preprocess", pre), ("classifier", clf)])


def _data_fingerprint(df: pd.DataFrame) -> str:
    h = hashlib.sha256(pd.util.hash_pandas_object(df, index=False).values.tobytes())
    return h.hexdigest()[:12]


def _plots(out_dir: Path, y_test, p_test, grid, costs, threshold) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.calibration import calibration_curve
        from sklearn.metrics import precision_recall_curve
    except ImportError:  # imagem de produção não tem matplotlib
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    files = []
    prec, rec, _ = precision_recall_curve(y_test, p_test)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(rec, prec, color="#1B3A5C", lw=2)
    ax.set(xlabel="Recall (fraudes capturadas)", ylabel="Precision", title="Curva Precision-Recall (teste)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "pr_curve.png", dpi=130)
    plt.close(fig)
    files.append("pr_curve.png")

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(grid, costs, color="#B3261E", lw=2)
    ax.axvline(threshold, ls="--", color="#1B3A5C", label=f"limiar ótimo = {threshold:.3f}")
    ax.set(xlabel="Limiar de decisão", ylabel="Custo total (EUR, validação)", title="Custo esperado × limiar", xscale="log")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "cost_curve.png", dpi=130)
    plt.close(fig)
    files.append("cost_curve.png")

    frac_pos, mean_pred = calibration_curve(y_test, p_test, n_bins=8, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot([0, 1], [0, 1], ls=":", color="grey")
    ax.plot(mean_pred, frac_pos, marker="o", color="#1B3A5C")
    ax.set(xlabel="Probabilidade prevista", ylabel="Frequência observada", title="Calibração (teste)", xscale="log", yscale="log")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "calibration.png", dpi=130)
    plt.close(fig)
    files.append("calibration.png")
    return files


def train(
    settings: Settings | None = None,
    data_path: Path | None = None,
    fast: bool = False,
    df: pd.DataFrame | None = None,
    make_plots: bool = True,
) -> dict:
    settings = settings or get_settings()
    t0 = time.perf_counter()

    if df is None:
        df, source = load_dataset(data_path or settings.raw_data_path, seed=SEED)
    else:
        source = "in-memory"
    fingerprint = _data_fingerprint(df)
    df, n_dups = deduplicate(df)
    split = temporal_split(df)

    n_fit = int(len(split.train) * 0.85)
    fit_df, es_df = split.train.iloc[:n_fit], split.train.iloc[n_fit:]
    X_fit, y_fit = fit_df[RAW_FEATURES], fit_df[TARGET].to_numpy()
    X_es, y_es = es_df[RAW_FEATURES], es_df[TARGET].to_numpy()
    X_val, y_val = split.valid[RAW_FEATURES], split.valid[TARGET].to_numpy()
    X_test, y_test = split.test[RAW_FEATURES], split.test[TARGET].to_numpy()
    ratio = float((y_fit == 0).sum() / max((y_fit == 1).sum(), 1))

    # ---------------- seleção de modelo ----------------
    candidates = {}
    fitted = {}
    for name in ["logreg_balanced", "lgbm_unweighted", "lgbm_sqrt_weight", "lgbm_full_weight"]:
        ts = time.perf_counter()
        model = _fit_candidate(name, X_fit, y_fit, X_es, y_es, ratio, fast)
        p_val = model.predict_proba(X_val)[:, 1]
        candidates[name] = {
            "valid_auprc": round(float(average_precision_score(y_val, p_val)), 4),
            "train_seconds": round(time.perf_counter() - ts, 2),
        }
        clf = model.named_steps["classifier"]
        if hasattr(clf, "best_iteration_") and clf.best_iteration_:
            candidates[name]["best_iteration"] = int(clf.best_iteration_)
        fitted[name] = model
        logger.info("candidato=%s auprc_valid=%.4f", name, candidates[name]["valid_auprc"])

    best_name = max(candidates, key=lambda n: candidates[n]["valid_auprc"])
    base_model = fitted[best_name]

    # ---------------- calibração + limiar ----------------
    calibrated = CalibratedClassifierCV(FrozenEstimator(base_model), method="sigmoid")
    calibrated.fit(X_val, y_val)
    p_val_cal = calibrated.predict_proba(X_val)[:, 1]
    threshold, val_cost, (grid, costs) = optimize_threshold(y_val, p_val_cal, split.valid["Amount"].to_numpy(), settings.cost)

    # ---------------- avaliação final (teste intocado) ----------------
    p_test = calibrated.predict_proba(X_test)[:, 1]
    p_test_uncal = base_model.predict_proba(X_test)[:, 1]
    y_pred = p_test >= threshold
    test_metrics = {
        **ranking_metrics(y_test, p_test),
        "brier_uncalibrated": round(float(np.mean((p_test_uncal - y_test) ** 2)), 6),
        **classification_report_at(y_test, p_test, threshold),
    }
    cost_test = business_cost(y_test, y_pred, split.test["Amount"].to_numpy(), settings.cost)

    # ---------------- artefatos ----------------
    trained_at = datetime.now(UTC)
    version = f"{trained_at:%Y%m%d%H%M%S}-{fingerprint[:8]}"
    feature_names = list(base_model.named_steps["preprocess"].get_feature_names_out())
    bundle = {
        "model": calibrated,  # pipeline completo calibrado: raw JSON -> probabilidade
        "base_model": base_model,  # usado para explicações (TreeSHAP nativo do LightGBM)
        "threshold": threshold,
        "version": version,
        "model_name": best_name,
        "feature_names": feature_names,
        "raw_features": RAW_FEATURES,
    }
    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, settings.model_path, compress=3)

    reference = build_reference_profile(split.valid, p_val_cal, threshold)
    reference["version"] = version
    settings.reference_path.write_text(json.dumps(reference, indent=2))

    period_hours = (split.test["Time"].max() - split.test["Time"].min()) / 3600
    metadata = {
        "version": version,
        "model_name": best_name,
        "trained_at": trained_at.isoformat(),
        "data": {
            "source": source,
            "fingerprint": fingerprint,
            "rows_after_dedup": len(df),
            "duplicates_removed": n_dups,
            "fraud_rate": round(float(df[TARGET].mean()), 6),
            "split": {
                k: {"rows": len(v), "frauds": int(v[TARGET].sum())}
                for k, v in (("train", split.train), ("valid", split.valid), ("test", split.test))
            },
        },
        "imbalance_ratio_train": round(ratio, 1),
        "candidates": candidates,
        "calibration": "sigmoid (Platt) na validação",
        "threshold": round(threshold, 6),
        "threshold_objective": "min custo esperado EUR (validação)",
        "valid_cost_at_threshold": round(val_cost, 2),
        "cost_model": settings.cost.model_dump(),
        "test_metrics": test_metrics,
        "test_business": {**cost_test.to_dict(), "period_hours": round(float(period_hours), 2)},
        "feature_names": feature_names,
        "environment": {
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "lightgbm": lgb.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "training_seconds": round(time.perf_counter() - t0, 1),
    }
    if make_plots:
        metadata["figures"] = _plots(settings.artifacts_dir.parent / "reports" / "figures", y_test, p_test, grid, costs, threshold)
    settings.metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    logger.info("modelo=%s versão=%s limiar=%.4f auprc_test=%.4f", best_name, version, threshold, test_metrics["auprc"])
    return metadata


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description="Treina o modelo FraudGuard")
    parser.add_argument("--data", type=Path, default=None, help="CSV do Kaggle (creditcard.csv)")
    parser.add_argument("--fast", action="store_true", help="Treino rápido (CI/smoke)")
    parser.add_argument("--synthetic-rows", type=int, default=None, help="Força dados sintéticos com N linhas")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    df = None
    if args.synthetic_rows:
        n_fr = max(int(args.synthetic_rows * 0.00172), 60)
        df = generate_synthetic_creditcard(n_samples=args.synthetic_rows, n_frauds=n_fr, seed=SEED)
    meta = train(data_path=args.data, fast=args.fast, df=df, make_plots=not args.no_plots)
    print(
        json.dumps(
            {k: meta[k] for k in ("version", "model_name", "threshold", "test_metrics", "test_business")}, indent=2, ensure_ascii=False
        )
    )


if __name__ == "__main__":
    main()
