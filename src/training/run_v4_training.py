"""
v4 training run — V-column missing indicators removed entirely.

Root cause progression:
  v1 (SMOTE + aucpr): AUC-PR=0.149  — SMOTE on 253 binary indicators invalid
  v2 (scale_pos_weight + auc): AUC-PR=0.131 — indicators still crowding colsample
  v3 (threshold 0.80): AUC-PR=0.201 — only 47 of 253 indicators removed
  Run A (raw features, no indicators): AUC-PR=0.485 — confirmed baseline without indicators

v4 fix: remove _was_missing indicators from FraudFeatureEngineer entirely.
  - V-columns get median imputation only (as in Run A)
  - Time, velocity, D-norm, target encoding remain
  - Expected feature count: ~392 (379 raw + 9 time/velocity + 3 D-norm + 4 te - 4 cat = 391)
  - Target: AUC-PR > 0.50 (Run A = 0.4853 + engineered features should add signal)

Logs to Vertex AI Experiments as:
  xgb-v4-no-indicators
  lgb-v4-no-indicators
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.features.engineer import build_from_config
from src.training.lightgbm_trainer import train_lgb_v4_no_indicators
from src.training.metrics import compute_fraud_metrics
from src.training.xgboost_trainer import (
    SORT_COL,
    TARGET,
    load_engineered_features,
    log_to_vertex_experiments,
    save_model_to_gcs,
    stratified_val_carve,
    train_xgb_v4_no_indicators,
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

    print(f"\nSplit summary (v4):")
    print(f"  Train : {len(X_train):>7,}  ({y_train.mean():.3%} fraud)")
    print(f"  Val   : {len(X_val):>7,}  ({y_val.mean():.3%} fraud)  stratified  ({int(y_val.sum())} fraud rows)")
    print(f"  Test  : {len(X_test):>7,}  ({y_test.mean():.3%} fraud)  temporal")

    xgb_cfg = config["model"]["xgboost"]
    lgb_cfg = config["model"]["lightgbm"]

    # ── Fit engineer once and report feature counts ────────────────────────────
    engineer_probe = build_from_config(config)
    X_probe_eng = engineer_probe.fit_transform(X_train, y_train).select_dtypes(include=[np.number])
    n_features = X_probe_eng.shape[1]
    print(f"\nFeature engineering (v4, no indicators):")
    print(f"  V-cols present   : {len(engineer_probe._v_cols_present)}")
    print(f"  Indicators created: 0  (removed — see engineer.py comment)")
    print(f"  Total numeric features: {n_features}")
    print(f"  NaN in engineered train: {X_probe_eng.isnull().sum().sum()}")

    # Save updated transformer to GCS (reflects indicator removal)
    transformer_uri = engineer_probe.save_to_gcs(config)
    print(f"  Transformer saved: {transformer_uri}")
    del engineer_probe, X_probe_eng  # free memory before training

    # ── XGBoost v4: no indicators + logloss early stopping ────────────────────
    engineer_xgb = build_from_config(config)
    model_xgb, params_xgb = train_xgb_v4_no_indicators(
        X_train, y_train, X_val, y_val, config, engineer_xgb
    )

    X_test_eng_xgb = engineer_xgb.transform(X_test).select_dtypes(include=[np.number])
    assert X_test_eng_xgb.isnull().sum().sum() == 0, "NaN in XGBoost v4 test features"
    y_proba_xgb = model_xgb.predict_proba(X_test_eng_xgb)[:, 1]
    metrics_xgb = compute_fraud_metrics(
        y_true=y_test.values,
        y_proba=y_proba_xgb,
        imbalance_strategy="scale_pos_weight",
        training_rows=len(X_train),
        test_rows=len(X_test),
    )

    xgb_run_name = "xgb-v4-no-indicators"
    log_to_vertex_experiments(
        config,
        xgb_run_name,
        {
            "model_type": "xgboost",
            "n_estimators": int(xgb_cfg["n_estimators"]),
            "max_depth": int(xgb_cfg["max_depth"]),
            "learning_rate": float(xgb_cfg["learning_rate"]),
            "early_stopping_rounds": int(xgb_cfg["early_stopping_rounds"]),
            "early_stopping_metric": "auc",  # logloss tested, failed (best_iter=0 with scale_pos_weight=27)
            "val_strategy": "stratified",
            "n_indicators": 0,
            "n_features": int(X_test_eng_xgb.shape[1]),
            "best_iteration": model_xgb.best_iteration,
            "train_frac": train_frac,
            "val_frac": val_frac,
            **params_xgb,
        },
        metrics_xgb,
    )
    xgb_gcs_uri = save_model_to_gcs(model_xgb, xgb_run_name, config)

    # ── LightGBM v4: no indicators + auc early stopping ──────────────────────
    engineer_lgb = build_from_config(config)
    model_lgb, params_lgb = train_lgb_v4_no_indicators(
        X_train, y_train, X_val, y_val, config, engineer_lgb
    )

    X_test_eng_lgb = engineer_lgb.transform(X_test).select_dtypes(include=[np.number])
    assert X_test_eng_lgb.isnull().sum().sum() == 0, "NaN in LightGBM v4 test features"
    y_proba_lgb = model_lgb.predict_proba(X_test_eng_lgb)[:, 1]
    metrics_lgb = compute_fraud_metrics(
        y_true=y_test.values,
        y_proba=y_proba_lgb,
        imbalance_strategy="lgb_is_unbalance",
        training_rows=len(X_train),
        test_rows=len(X_test),
    )

    lgb_run_name = "lgb-v4-no-indicators"
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
            "n_indicators": 0,
            "n_features": int(X_test_eng_lgb.shape[1]),
            "best_iteration": model_lgb.best_iteration_,
            "train_frac": train_frac,
            "val_frac": val_frac,
            **params_lgb,
        },
        metrics_lgb,
    )
    lgb_gcs_uri = save_model_to_gcs(model_lgb, lgb_run_name, config)

    # ── Results table ──────────────────────────────────────────────────────────
    print(f"\n{'═'*80}")
    print("V4 RESULTS — no missing indicators")
    print(f"{'═'*80}")
    print(f"{'Model':<40} {'AUC-PR':>8} {'AUC-ROC':>8} {'best_iter':>10} {'F1':>7}")
    print(f"{'─'*80}")
    print(
        f"{'xgb-v4-no-indicators':<40}"
        f" {metrics_xgb['auc_pr']:>8.4f}"
        f" {metrics_xgb['auc_roc']:>8.4f}"
        f" {model_xgb.best_iteration:>10}"
        f" {metrics_xgb['f1']:>7.4f}"
    )
    print(
        f"{'lgb-v4-no-indicators':<40}"
        f" {metrics_lgb['auc_pr']:>8.4f}"
        f" {metrics_lgb['auc_roc']:>8.4f}"
        f" {model_lgb.best_iteration_:>10}"
        f" {metrics_lgb['f1']:>7.4f}"
    )
    print(f"{'─'*80}")
    print(f"{'xgb-v3-reduced-indicators' :<40} {'0.2009':>8} {'0.6981':>8} {'31':>10}")
    print(f"{'lgb-v3-reduced-indicators' :<40} {'0.0554':>8} {'0.6581':>8} {'25':>10}")
    print(f"{'xgb-v2-class-weight-fixed' :<40} {'0.1310':>8} {'0.6909':>8} {'339':>10}")
    print(f"{'lgb-v2-isunbalance-fixed'  :<40} {'0.0491':>8} {'0.5929':>8} {'17':>10}")
    print(f"{'Run A (raw features, XGB)' :<40} {'0.4853':>8} {'0.8894':>8} {'299':>10}")
    print(f"{'Run B (raw+time, XGB)'     :<40} {'0.4871':>8} {'0.8907':>8} {'299':>10}")
    print(f"{'Logistic Regression baseline':<40} {'0.2172':>8} {'0.8600':>8} {'N/A':>10}")
    print(f"{'═'*80}\n")

    results = {
        "xgb_v4": {
            **metrics_xgb,
            "best_iteration": model_xgb.best_iteration,
            "n_features": int(X_test_eng_xgb.shape[1]),
            "gcs_uri": xgb_gcs_uri,
        },
        "lgb_v4": {
            **metrics_lgb,
            "best_iteration": model_lgb.best_iteration_,
            "n_features": int(X_test_eng_lgb.shape[1]),
            "gcs_uri": lgb_gcs_uri,
        },
    }
    out_path = Path("evaluation/v4_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    logger.info("v4_results_saved", extra={"path": str(out_path)})

    return results


if __name__ == "__main__":
    main()
