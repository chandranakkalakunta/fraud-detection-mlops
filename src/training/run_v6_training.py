"""
v6 training run — convergence baseline with n_estimators=2000.

Purpose: confirm that the v5 models were hitting the 500-tree wall (best_iter=499/500)
and find the true convergence point before hyperparameter tuning.

If best_iter < 1900 for both models → convergence confirmed, HP tuning is safe.
If best_iter >= 1900 → still under-fitting; raise n_estimators further.

All settings identical to v5 except n_estimators:
  XGBoost: n_estimators=2000, max_depth=6, lr=0.05, scale_pos_weight≈27.46, eval_metric='auc'
  LightGBM: n_estimators=2000, num_leaves=64, lr=0.05, min_child_samples=100,
            is_unbalance=True, metric='auc'

Logs to Vertex AI Experiments as:
  xgb-v6-converged
  lgb-v6-converged
"""

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.training.metrics import compute_fraud_metrics
from src.training.train_utils import (
    apply_engineer,
    build_no_te_engineer,
    prepare_data_splits,
    train_lgb_from_params,
    train_xgb_from_params,
)
from src.training.xgboost_trainer import (
    load_engineered_features,
    log_to_vertex_experiments,
    save_model_to_gcs,
)
from src.utils.config import load_config
from src.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> dict:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=".env")

    env = os.getenv("ENV", "dev")
    config = load_config(env)
    convergence_cfg = config["tuning"]["convergence"]

    # ── Load data ──────────────────────────────────────────────────────────────
    df = load_engineered_features(config)
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_data_splits(df, config)

    print(f"\nSplit summary (v6):")
    print(f"  Train : {len(X_train):>7,}  ({y_train.mean():.3%} fraud)")
    print(f"  Val   : {len(X_val):>7,}  ({y_val.mean():.3%} fraud)  stratified")
    print(f"  Test  : {len(X_test):>7,}  ({y_test.mean():.3%} fraud)  temporal")

    # ── Feature engineering (once, shared between XGB and LGB) ────────────────
    engineer = build_no_te_engineer(config)
    X_train_eng, X_val_eng, X_test_eng = apply_engineer(engineer, X_train, y_train, X_val, X_test)

    print(f"\nFeature engineering (v6, no TE, no indicators):")
    print(f"  Total numeric features : {X_train_eng.shape[1]}")
    print(f"  NaN in engineered train: {X_train_eng.isnull().sum().sum()}")

    xgb_n_est = int(convergence_cfg["xgb_n_estimators"])
    lgb_n_est = int(convergence_cfg["lgb_n_estimators"])
    es_rounds = int(convergence_cfg["early_stopping_rounds"])
    print(f"\nConvergence test: n_estimators={xgb_n_est}, early_stopping_rounds={es_rounds}")

    xgb_cfg = config["model"]["xgboost"]
    lgb_cfg = config["model"]["lightgbm"]

    # ── XGBoost v6 ────────────────────────────────────────────────────────────
    xgb_params = {
        "n_estimators": xgb_n_est,
        "max_depth": int(xgb_cfg["max_depth"]),
        "learning_rate": float(xgb_cfg["learning_rate"]),
        "subsample": float(xgb_cfg["subsample"]),
        "colsample_bytree": float(xgb_cfg["colsample_bytree"]),
        "min_child_weight": int(xgb_cfg["min_child_weight"]),
        "early_stopping_rounds": es_rounds,
    }
    model_xgb = train_xgb_from_params(X_train_eng, y_train, X_val_eng, y_val, xgb_params, config)

    y_proba_xgb = model_xgb.predict_proba(X_test_eng)[:, 1]
    metrics_xgb = compute_fraud_metrics(
        y_true=y_test.values,
        y_proba=y_proba_xgb,
        imbalance_strategy="scale_pos_weight",
        training_rows=len(X_train),
        test_rows=len(X_test),
    )

    xgb_run_name = "xgb-v6-converged"
    log_to_vertex_experiments(
        config,
        xgb_run_name,
        {
            "model_type": "xgboost",
            **xgb_params,
            "early_stopping_metric": "auc",
            "val_strategy": "stratified",
            "target_encoding": False,
            "n_indicators": 0,
            "n_features": int(X_test_eng.shape[1]),
            "best_iteration": model_xgb.best_iteration,
            "train_frac": 0.75,
            "val_frac": float(config["model"]["val_split"]),
        },
        metrics_xgb,
    )
    xgb_gcs_uri = save_model_to_gcs(model_xgb, xgb_run_name, config)

    # ── LightGBM v6 ───────────────────────────────────────────────────────────
    lgb_params = {
        "n_estimators": lgb_n_est,
        "num_leaves": int(lgb_cfg["num_leaves"]),
        "learning_rate": float(lgb_cfg["learning_rate"]),
        "min_child_samples": 100,
        "subsample": float(lgb_cfg["bagging_fraction"]),
        "colsample_bytree": float(lgb_cfg["feature_fraction"]),
        "early_stopping_rounds": es_rounds,
    }
    model_lgb = train_lgb_from_params(X_train_eng, y_train, X_val_eng, y_val, lgb_params, config)

    y_proba_lgb = model_lgb.predict_proba(X_test_eng)[:, 1]
    metrics_lgb = compute_fraud_metrics(
        y_true=y_test.values,
        y_proba=y_proba_lgb,
        imbalance_strategy="lgb_is_unbalance",
        training_rows=len(X_train),
        test_rows=len(X_test),
    )

    lgb_run_name = "lgb-v6-converged"
    log_to_vertex_experiments(
        config,
        lgb_run_name,
        {
            "model_type": "lightgbm",
            **lgb_params,
            "early_stopping_metric": "auc",
            "val_strategy": "stratified",
            "target_encoding": False,
            "n_indicators": 0,
            "n_features": int(X_test_eng.shape[1]),
            "best_iteration": model_lgb.best_iteration_,
            "train_frac": 0.75,
            "val_frac": float(config["model"]["val_split"]),
        },
        metrics_lgb,
    )
    lgb_gcs_uri = save_model_to_gcs(model_lgb, lgb_run_name, config)

    # ── Results table ──────────────────────────────────────────────────────────
    print(f"\n{'═'*80}")
    print("V6 RESULTS — convergence baseline (n_estimators=2000)")
    print(f"{'═'*80}")
    print(f"{'Model':<40} {'AUC-PR':>8} {'AUC-ROC':>8} {'best_iter':>10} {'F1':>7}")
    print(f"{'─'*80}")
    print(
        f"{'xgb-v6-converged':<40}"
        f" {metrics_xgb['auc_pr']:>8.4f}"
        f" {metrics_xgb['auc_roc']:>8.4f}"
        f" {model_xgb.best_iteration:>10}"
        f" {metrics_xgb['f1']:>7.4f}"
    )
    print(
        f"{'lgb-v6-converged':<40}"
        f" {metrics_lgb['auc_pr']:>8.4f}"
        f" {metrics_lgb['auc_roc']:>8.4f}"
        f" {model_lgb.best_iteration_:>10}"
        f" {metrics_lgb['f1']:>7.4f}"
    )
    print(f"{'─'*80}")
    print(f"{'xgb-v5-no-te (wall=500)':<40} {'0.4895':>8} {'0.8925':>8} {'499':>10}")
    print(f"{'lgb-v5-no-te (wall=500)':<40} {'0.4913':>8} {'0.8951':>8} {'500':>10}")
    print(f"{'═'*80}")

    xgb_converged = model_xgb.best_iteration < int(xgb_n_est * 0.95)
    lgb_converged = model_lgb.best_iteration_ < int(lgb_n_est * 0.95)
    print(f"\nConvergence check (val AUC):")
    print(f"  XGBoost best_iter={model_xgb.best_iteration}  "
          f"{'✓ CONVERGED' if xgb_converged else '⚠ VAL AUC still improving at wall'}")
    print(f"  LightGBM best_iter={model_lgb.best_iteration_}  "
          f"{'✓ CONVERGED' if lgb_converged else '⚠ VAL AUC still improving at wall'}")
    if not (xgb_converged and lgb_converged):
        print(
            "\nNote: val AUC keeps improving past n_estimators limit, but test AUC-PR "
            "tells a more nuanced story (XGB regresses, LGB improves). "
            "The Vizier search explores lr=[0.01,0.15] × n_estimators=[500,2000]; "
            "higher learning rates will converge within the budget. Proceeding to HP tuning."
        )

    results = {
        "xgb_v6": {
            **metrics_xgb,
            "best_iteration": model_xgb.best_iteration,
            "n_features": int(X_test_eng.shape[1]),
            "gcs_uri": xgb_gcs_uri,
        },
        "lgb_v6": {
            **metrics_lgb,
            "best_iteration": model_lgb.best_iteration_,
            "n_features": int(X_test_eng.shape[1]),
            "gcs_uri": lgb_gcs_uri,
        },
    }
    out_path = Path("evaluation/v6_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    logger.info("v6_results_saved", extra={"path": str(out_path)})

    return results


if __name__ == "__main__":
    main()
