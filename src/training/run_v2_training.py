"""
v2 training run — Phase 2B fixes.

Root cause of v1 AUC-PR 0.149 (below 0.217 logistic regression baseline):
  1. SMOTE on 654 mixed binary+continuous features produced invalid synthetic
     samples — 253 binary _was_missing indicators interpolated to fractional
     values, destroying the missingness signal.
  2. eval_metric='aucpr' triggered premature early stopping (best_iteration=13)
     because AUC-PR is too noisy at 3.5% fraud rate during early rounds.
  3. Temporal val carve may land in a low-fraud time window, further degrading
     the early stopping signal.

v2 fixes:
  - SMOTE replaced with scale_pos_weight (XGBoost) / is_unbalance (LightGBM)
  - eval_metric / metric changed to logloss / binary_logloss (smooth signal)
  - early_stopping_rounds: 50 → 100
  - Validation: stratified carve from training pool (guarantees 3.5% fraud)

Logs to Vertex AI Experiments as:
  xgb-v2-class-weight-fixed
  lgb-v2-isunbalance-fixed
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.features.engineer import build_from_config
from src.training.lightgbm_trainer import train_lgb_v2_isunbalance_fixed
from src.training.metrics import compute_fraud_metrics
from src.training.xgboost_trainer import (
    SORT_COL,
    TARGET,
    load_engineered_features,
    log_to_vertex_experiments,
    save_model_to_gcs,
    stratified_val_carve,
    train_v2_class_weight_fixed,
)
from src.utils.config import load_config
from src.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> dict:
    from dotenv import load_dotenv

    load_dotenv(dotenv_path=".env")

    env = os.getenv("ENV", "dev")
    config = load_config(env)

    # ── Load data once ─────────────────────────────────────────────────────────
    df = load_engineered_features(config)

    train_frac = 0.75
    val_frac = float(config["model"]["val_split"])
    random_state = int(config["model"]["random_state"])

    # Temporal test boundary: last 25% — never touched during training or val tuning
    df_sorted = df.sort_values(SORT_COL).reset_index(drop=True)
    n = len(df_sorted)
    train_end = int(n * train_frac)
    train_pool = df_sorted.iloc[:train_end]
    test_df = df_sorted.iloc[train_end:].reset_index(drop=True)

    # Stratified val carve from training pool — preserves full fraud rate in val
    train_df, val_df = stratified_val_carve(train_pool, val_frac=val_frac, random_state=random_state)

    feature_cols = [c for c in df.columns if c not in [TARGET]]
    X_train = train_df[feature_cols].reset_index(drop=True)
    y_train = train_df[TARGET].astype(int).reset_index(drop=True)
    X_val = val_df[feature_cols].reset_index(drop=True)
    y_val = val_df[TARGET].astype(int).reset_index(drop=True)
    X_test = test_df[feature_cols].reset_index(drop=True)
    y_test = test_df[TARGET].astype(int).reset_index(drop=True)

    print(f"\nSplit summary (v2):")
    print(f"  Train : {len(X_train):>7,}  ({y_train.mean():.3%} fraud)")
    print(f"  Val   : {len(X_val):>7,}  ({y_val.mean():.3%} fraud)  stratified  ({int(y_val.sum())} fraud rows)")
    print(f"  Test  : {len(X_test):>7,}  ({y_test.mean():.3%} fraud)  temporal")

    xgb_cfg = config["model"]["xgboost"]
    lgb_cfg = config["model"]["lightgbm"]

    # ── XGBoost v2: scale_pos_weight + logloss early stopping ─────────────────
    engineer_xgb = build_from_config(config)
    model_xgb, params_xgb = train_v2_class_weight_fixed(
        X_train, y_train, X_val, y_val, config, engineer_xgb
    )

    X_test_eng_xgb = engineer_xgb.transform(X_test).select_dtypes(include=[np.number])
    assert X_test_eng_xgb.isnull().sum().sum() == 0, "NaN in XGBoost v2 test features"
    y_proba_xgb = model_xgb.predict_proba(X_test_eng_xgb)[:, 1]
    metrics_xgb = compute_fraud_metrics(
        y_true=y_test.values,
        y_proba=y_proba_xgb,
        imbalance_strategy="scale_pos_weight",
        training_rows=len(X_train),
        test_rows=len(X_test),
    )

    xgb_run_name = "xgb-v2-class-weight-fixed"
    log_to_vertex_experiments(
        config,
        xgb_run_name,
        {
            "model_type": "xgboost",
            "n_estimators": int(xgb_cfg["n_estimators"]),
            "max_depth": int(xgb_cfg["max_depth"]),
            "learning_rate": float(xgb_cfg["learning_rate"]),
            "early_stopping_rounds": int(xgb_cfg["early_stopping_rounds"]),
            "early_stopping_metric": "auc",
            "val_strategy": "stratified",
            "best_iteration": model_xgb.best_iteration,
            "train_frac": train_frac,
            "val_frac": val_frac,
            **params_xgb,
        },
        metrics_xgb,
    )
    xgb_gcs_uri = save_model_to_gcs(model_xgb, xgb_run_name, config)

    # ── LightGBM v2: is_unbalance + binary_logloss early stopping ─────────────
    engineer_lgb = build_from_config(config)
    model_lgb, params_lgb = train_lgb_v2_isunbalance_fixed(
        X_train, y_train, X_val, y_val, config, engineer_lgb
    )

    X_test_eng_lgb = engineer_lgb.transform(X_test).select_dtypes(include=[np.number])
    assert X_test_eng_lgb.isnull().sum().sum() == 0, "NaN in LightGBM v2 test features"
    y_proba_lgb = model_lgb.predict_proba(X_test_eng_lgb)[:, 1]
    metrics_lgb = compute_fraud_metrics(
        y_true=y_test.values,
        y_proba=y_proba_lgb,
        imbalance_strategy="lgb_is_unbalance",
        training_rows=len(X_train),
        test_rows=len(X_test),
    )

    lgb_run_name = "lgb-v2-isunbalance-fixed"
    log_to_vertex_experiments(
        config,
        lgb_run_name,
        {
            "model_type": "lightgbm",
            "num_leaves": int(lgb_cfg["num_leaves"]),
            "learning_rate": float(lgb_cfg["learning_rate"]),
            "n_estimators": int(lgb_cfg["n_estimators"]),
            "early_stopping_rounds": int(lgb_cfg["early_stopping_rounds"]),
            "early_stopping_metric": "binary_logloss",
            "val_strategy": "stratified",
            "best_iteration": model_lgb.best_iteration_,
            "train_frac": train_frac,
            "val_frac": val_frac,
            **params_lgb,
        },
        metrics_lgb,
    )
    lgb_gcs_uri = save_model_to_gcs(model_lgb, lgb_run_name, config)

    # ── Results ────────────────────────────────────────────────────────────────
    print(f"\n{'═'*74}")
    print("V2 TRAINING RESULTS")
    print(f"{'═'*74}")
    print(f"{'Model':<36} {'AUC-PR':>8} {'AUC-ROC':>8} {'best_iter':>10} {'F1':>7}")
    print(f"{'─'*74}")
    print(
        f"{'xgb-v2-class-weight-fixed':<36}"
        f" {metrics_xgb['auc_pr']:>8.4f}"
        f" {metrics_xgb['auc_roc']:>8.4f}"
        f" {model_xgb.best_iteration:>10}"
        f" {metrics_xgb['f1']:>7.4f}"
    )
    print(
        f"{'lgb-v2-isunbalance-fixed':<36}"
        f" {metrics_lgb['auc_pr']:>8.4f}"
        f" {metrics_lgb['auc_roc']:>8.4f}"
        f" {model_lgb.best_iteration_:>10}"
        f" {metrics_lgb['f1']:>7.4f}"
    )
    print(f"{'─'*74}")
    print(f"{'Logistic Regression baseline':<36} {'0.2172':>8} {'0.8600':>8} {'N/A':>10} {'':>7}")
    print(f"{'═'*74}\n")

    results = {
        "xgb_v2": {
            **metrics_xgb,
            "best_iteration": model_xgb.best_iteration,
            "gcs_uri": xgb_gcs_uri,
        },
        "lgb_v2": {
            **metrics_lgb,
            "best_iteration": model_lgb.best_iteration_,
            "gcs_uri": lgb_gcs_uri,
        },
    }
    out_path = Path("evaluation/v2_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    logger.info("v2_results_saved", extra={"path": str(out_path)})

    return results


if __name__ == "__main__":
    main()
