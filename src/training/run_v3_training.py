"""
v3 training run — reduced missing indicators.

Root cause of v1/v2 AUC-PR 0.13 (below 0.49 raw-feature baseline):
  The feature engineering pipeline at missing_threshold=0.05 created 253 binary
  _was_missing indicators for V-columns with >5% null.  Most V-columns are
  95–100% null, so these indicators are nearly always=1 (near-constant), yet they
  occupied 49% of XGBoost's colsample_bytree=0.8 budget per tree.  The diagnostic
  (feature_diagnostic.py) confirmed B→C AUC-PR delta = −0.364.

v3 fixes:
  - missing_threshold raised 0.05 → 0.80: indicators created only for 5–80% null
    V-columns; >80% null columns get median imputation only (no indicator)
  - XGBoost: learning_rate=0.1 (vs config 0.05) for faster convergence on cleaner
    feature set; everything else identical to v2
  - LightGBM: min_child_samples=100 (vs 20) to prevent fraud micro-leaves under
    is_unbalance=True; is_unbalance and metric="auc" unchanged from v2

Logs to Vertex AI Experiments as:
  xgb-v3-reduced-indicators
  lgb-v3-reduced-indicators
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.features.engineer import build_from_config
from src.training.lightgbm_trainer import train_lgb_v3_reduced_indicators
from src.training.metrics import compute_fraud_metrics
from src.training.xgboost_trainer import (
    SORT_COL,
    TARGET,
    load_engineered_features,
    log_to_vertex_experiments,
    save_model_to_gcs,
    stratified_val_carve,
    train_xgb_v3_reduced_indicators,
)
from src.utils.config import load_config
from src.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> dict:
    from dotenv import load_dotenv

    load_dotenv(dotenv_path=".env")

    env = os.getenv("ENV", "dev")
    config = load_config(env)

    # ── Load data ──────────────────────────────────────────────────────────────
    df = load_engineered_features(config)

    train_frac = 0.75
    val_frac = float(config["model"]["val_split"])
    random_state = int(config["model"]["random_state"])

    df_sorted = df.sort_values(SORT_COL).reset_index(drop=True)
    n = len(df_sorted)
    train_end = int(n * train_frac)
    train_pool = df_sorted.iloc[:train_end]
    test_df = df_sorted.iloc[train_end:].reset_index(drop=True)

    train_df, val_df = stratified_val_carve(train_pool, val_frac=val_frac, random_state=random_state)

    feature_cols = [c for c in df.columns if c not in [TARGET]]
    X_train = train_df[feature_cols].reset_index(drop=True)
    y_train = train_df[TARGET].astype(int).reset_index(drop=True)
    X_val = val_df[feature_cols].reset_index(drop=True)
    y_val = val_df[TARGET].astype(int).reset_index(drop=True)
    X_test = test_df[feature_cols].reset_index(drop=True)
    y_test = test_df[TARGET].astype(int).reset_index(drop=True)

    print(f"\nSplit summary (v3):")
    print(f"  Train : {len(X_train):>7,}  ({y_train.mean():.3%} fraud)")
    print(f"  Val   : {len(X_val):>7,}  ({y_val.mean():.3%} fraud)  stratified  ({int(y_val.sum())} fraud rows)")
    print(f"  Test  : {len(X_test):>7,}  ({y_test.mean():.3%} fraud)  temporal")

    xgb_cfg = config["model"]["xgboost"]
    lgb_cfg = config["model"]["lightgbm"]

    # ── XGBoost v3: reduced indicators + lr=0.1 ───────────────────────────────
    engineer_xgb = build_from_config(config)
    model_xgb, params_xgb = train_xgb_v3_reduced_indicators(
        X_train, y_train, X_val, y_val, config, engineer_xgb
    )

    n_indicators_xgb = sum(1 for c in engineer_xgb._high_null_v_cols if c)
    print(f"\nXGBoost v3 engineer: {len(engineer_xgb._v_cols_present)} V-cols present, "
          f"{n_indicators_xgb} indicators (was 253 at threshold=0.05)")

    X_test_eng_xgb = engineer_xgb.transform(X_test).select_dtypes(include=[np.number])
    assert X_test_eng_xgb.isnull().sum().sum() == 0, "NaN in XGBoost v3 test features"
    y_proba_xgb = model_xgb.predict_proba(X_test_eng_xgb)[:, 1]
    metrics_xgb = compute_fraud_metrics(
        y_true=y_test.values,
        y_proba=y_proba_xgb,
        imbalance_strategy="scale_pos_weight",
        training_rows=len(X_train),
        test_rows=len(X_test),
    )

    xgb_run_name = "xgb-v3-reduced-indicators"
    log_to_vertex_experiments(
        config,
        xgb_run_name,
        {
            "model_type": "xgboost",
            "n_estimators": int(xgb_cfg["n_estimators"]),
            "max_depth": int(xgb_cfg["max_depth"]),
            "learning_rate": 0.1,
            "early_stopping_rounds": int(xgb_cfg["early_stopping_rounds"]),
            "early_stopping_metric": "auc",
            "val_strategy": "stratified",
            "missing_threshold": 0.80,
            "n_indicators": n_indicators_xgb,
            "n_features": X_test_eng_xgb.shape[1],
            "best_iteration": model_xgb.best_iteration,
            "train_frac": train_frac,
            "val_frac": val_frac,
            **params_xgb,
        },
        metrics_xgb,
    )
    xgb_gcs_uri = save_model_to_gcs(model_xgb, xgb_run_name, config)

    # ── LightGBM v3: reduced indicators + min_child_samples=100 ─────────────
    engineer_lgb = build_from_config(config)
    model_lgb, params_lgb = train_lgb_v3_reduced_indicators(
        X_train, y_train, X_val, y_val, config, engineer_lgb
    )

    n_indicators_lgb = len(engineer_lgb._high_null_v_cols)
    print(f"LightGBM v3 engineer: {len(engineer_lgb._v_cols_present)} V-cols present, "
          f"{n_indicators_lgb} indicators")

    X_test_eng_lgb = engineer_lgb.transform(X_test).select_dtypes(include=[np.number])
    assert X_test_eng_lgb.isnull().sum().sum() == 0, "NaN in LightGBM v3 test features"
    y_proba_lgb = model_lgb.predict_proba(X_test_eng_lgb)[:, 1]
    metrics_lgb = compute_fraud_metrics(
        y_true=y_test.values,
        y_proba=y_proba_lgb,
        imbalance_strategy="lgb_is_unbalance",
        training_rows=len(X_train),
        test_rows=len(X_test),
    )

    lgb_run_name = "lgb-v3-reduced-indicators"
    log_to_vertex_experiments(
        config,
        lgb_run_name,
        {
            "model_type": "lightgbm",
            "num_leaves": int(lgb_cfg["num_leaves"]),
            "learning_rate": float(lgb_cfg["learning_rate"]),
            "n_estimators": int(lgb_cfg["n_estimators"]),
            "min_child_samples": 100,
            "early_stopping_rounds": int(lgb_cfg["early_stopping_rounds"]),
            "early_stopping_metric": "auc",
            "val_strategy": "stratified",
            "missing_threshold": 0.80,
            "n_indicators": n_indicators_lgb,
            "n_features": X_test_eng_lgb.shape[1],
            "best_iteration": model_lgb.best_iteration_,
            "train_frac": train_frac,
            "val_frac": val_frac,
            **params_lgb,
        },
        metrics_lgb,
    )
    lgb_gcs_uri = save_model_to_gcs(model_lgb, lgb_run_name, config)

    # ── Results table ──────────────────────────────────────────────────────────
    print(f"\n{'═'*78}")
    print("V3 TRAINING RESULTS  (missing_threshold=0.80 — reduced indicators)")
    print(f"{'═'*78}")
    print(f"{'Model':<40} {'AUC-PR':>8} {'AUC-ROC':>8} {'best_iter':>10} {'F1':>7}")
    print(f"{'─'*78}")

    rows = [
        ("xgb-v3-reduced-indicators",  metrics_xgb, model_xgb.best_iteration),
        ("lgb-v3-reduced-indicators",  metrics_lgb, model_lgb.best_iteration_),
    ]
    for name, m, bi in rows:
        print(f"{name:<40} {m['auc_pr']:>8.4f} {m['auc_roc']:>8.4f} {bi:>10} {m['f1']:>7.4f}")

    print(f"{'─'*78}")
    print(f"{'Run A (raw features, XGB)' :<40} {'0.4853':>8} {'0.8894':>8} {'299':>10}")
    print(f"{'Run B (raw+time, XGB)'     :<40} {'0.4871':>8} {'0.8907':>8} {'299':>10}")
    print(f"{'xgb-v2-class-weight-fixed' :<40} {'0.1310':>8} {'0.6909':>8} {'339':>10}")
    print(f"{'lgb-v2-isunbalance-fixed'  :<40} {'0.0491':>8} {'0.5929':>8} {'17':>10}")
    print(f"{'Logistic Regression baseline':<40} {'0.2172':>8} {'0.8600':>8} {'N/A':>10}")
    print(f"{'═'*78}\n")

    results = {
        "xgb_v3": {
            **metrics_xgb,
            "best_iteration": model_xgb.best_iteration,
            "n_indicators": n_indicators_xgb,
            "n_features": int(X_test_eng_xgb.shape[1]),
            "gcs_uri": xgb_gcs_uri,
        },
        "lgb_v3": {
            **metrics_lgb,
            "best_iteration": model_lgb.best_iteration_,
            "n_indicators": n_indicators_lgb,
            "n_features": int(X_test_eng_lgb.shape[1]),
            "gcs_uri": lgb_gcs_uri,
        },
    }
    out_path = Path("evaluation/v3_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    logger.info("v3_results_saved", extra={"path": str(out_path)})

    return results


if __name__ == "__main__":
    main()
