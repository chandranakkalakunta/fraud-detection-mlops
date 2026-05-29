"""
v5 training run — target encoding stripped.

v4 finding (xgb best_iter=15, AUC-PR=0.174 < Run A 0.485):
  LOO target encoding (card4_te, card6_te, P_emaildomain_te, R_emaildomain_te)
  causes early stopping collapse.  The model saturates on TE features within
  ~15 trees (card4_te SHAP=0.934 in Phase 2B — model relies on it immediately).
  LOO values in training are label-aware (exact per-row leave-one-out); val uses
  smoothed category means.  This train/val distribution mismatch means val AUC
  peaks early, then drops as the model pushes more weight onto miscalibrated TE
  features → early stopping fires at tree 15.

v5 fix: target_encode_cols=[] — categorical columns omitted entirely.
  The numerical columns (V1-V339, C1-C14, D1-D15, etc.) and engineered features
  (time, velocity, D-norm) are unchanged.  If v5 recovers to ~0.485 (Run A),
  target encoding is the culprit and must be replaced with a leakage-safe
  alternative (e.g. held-out fold encoding, or dropped permanently).

Logs to Vertex AI Experiments as:
  xgb-v5-no-te
  lgb-v5-no-te
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.features.engineer import FraudFeatureEngineer
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


def _build_no_te_engineer(config: dict) -> FraudFeatureEngineer:
    """Engineer with no target encoding — isolates TE as the performance variable."""
    return FraudFeatureEngineer(
        target_encode_cols=[],
        d_norm_cols=config["features"]["d_normalize_cols"],
        d_group_col=config["features"]["d_normalize_group_col"],
    )


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

    print(f"\nSplit summary (v5):")
    print(f"  Train : {len(X_train):>7,}  ({y_train.mean():.3%} fraud)")
    print(f"  Val   : {len(X_val):>7,}  ({y_val.mean():.3%} fraud)  stratified")
    print(f"  Test  : {len(X_test):>7,}  ({y_test.mean():.3%} fraud)  temporal")

    xgb_cfg = config["model"]["xgboost"]
    lgb_cfg = config["model"]["lightgbm"]

    # Report feature counts
    probe = _build_no_te_engineer(config)
    X_probe = probe.fit_transform(X_train, y_train).select_dtypes(include=[np.number])
    n_features = X_probe.shape[1]
    print(f"\nFeature engineering (v5, no TE, no indicators):")
    print(f"  Total numeric features : {n_features}")
    print(f"  NaN in engineered train: {X_probe.isnull().sum().sum()}")
    del probe, X_probe

    # ── XGBoost v5 ────────────────────────────────────────────────────────────
    engineer_xgb = _build_no_te_engineer(config)
    model_xgb, params_xgb = train_xgb_v4_no_indicators(
        X_train, y_train, X_val, y_val, config, engineer_xgb
    )

    X_test_eng_xgb = engineer_xgb.transform(X_test).select_dtypes(include=[np.number])
    assert X_test_eng_xgb.isnull().sum().sum() == 0, "NaN in XGBoost v5 test features"
    y_proba_xgb = model_xgb.predict_proba(X_test_eng_xgb)[:, 1]
    metrics_xgb = compute_fraud_metrics(
        y_true=y_test.values,
        y_proba=y_proba_xgb,
        imbalance_strategy="scale_pos_weight",
        training_rows=len(X_train),
        test_rows=len(X_test),
    )

    xgb_run_name = "xgb-v5-no-te"
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
            "n_indicators": 0,
            "target_encoding": False,
            "n_features": int(X_test_eng_xgb.shape[1]),
            "best_iteration": model_xgb.best_iteration,
            "train_frac": train_frac,
            "val_frac": val_frac,
            **params_xgb,
        },
        metrics_xgb,
    )
    xgb_gcs_uri = save_model_to_gcs(model_xgb, xgb_run_name, config)

    # ── LightGBM v5 ───────────────────────────────────────────────────────────
    engineer_lgb = _build_no_te_engineer(config)
    model_lgb, params_lgb = train_lgb_v4_no_indicators(
        X_train, y_train, X_val, y_val, config, engineer_lgb
    )

    X_test_eng_lgb = engineer_lgb.transform(X_test).select_dtypes(include=[np.number])
    assert X_test_eng_lgb.isnull().sum().sum() == 0, "NaN in LightGBM v5 test features"
    y_proba_lgb = model_lgb.predict_proba(X_test_eng_lgb)[:, 1]
    metrics_lgb = compute_fraud_metrics(
        y_true=y_test.values,
        y_proba=y_proba_lgb,
        imbalance_strategy="lgb_is_unbalance",
        training_rows=len(X_train),
        test_rows=len(X_test),
    )

    lgb_run_name = "lgb-v5-no-te"
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
            "target_encoding": False,
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
    print("V5 RESULTS — no indicators, no target encoding")
    print(f"{'═'*80}")
    print(f"{'Model':<40} {'AUC-PR':>8} {'AUC-ROC':>8} {'best_iter':>10} {'F1':>7}")
    print(f"{'─'*80}")
    print(
        f"{'xgb-v5-no-te':<40}"
        f" {metrics_xgb['auc_pr']:>8.4f}"
        f" {metrics_xgb['auc_roc']:>8.4f}"
        f" {model_xgb.best_iteration:>10}"
        f" {metrics_xgb['f1']:>7.4f}"
    )
    print(
        f"{'lgb-v5-no-te':<40}"
        f" {metrics_lgb['auc_pr']:>8.4f}"
        f" {metrics_lgb['auc_roc']:>8.4f}"
        f" {model_lgb.best_iteration_:>10}"
        f" {metrics_lgb['f1']:>7.4f}"
    )
    print(f"{'─'*80}")
    print(f"{'xgb-v4-no-indicators'          :<40} {'0.1743':>8} {'0.7140':>8} {'15':>10}")
    print(f"{'lgb-v4-no-indicators'          :<40} {'0.0678':>8} {'0.7191':>8} {'47':>10}")
    print(f"{'xgb-v3-reduced-indicators'     :<40} {'0.2009':>8} {'0.6981':>8} {'31':>10}")
    print(f"{'Run A (raw features, XGB)'     :<40} {'0.4853':>8} {'0.8894':>8} {'299':>10}")
    print(f"{'Run B (raw+time, XGB)'         :<40} {'0.4871':>8} {'0.8907':>8} {'299':>10}")
    print(f"{'Logistic Regression baseline'  :<40} {'0.2172':>8} {'0.8600':>8} {'N/A':>10}")
    print(f"{'═'*80}\n")

    results = {
        "xgb_v5": {
            **metrics_xgb,
            "best_iteration": model_xgb.best_iteration,
            "n_features": int(X_test_eng_xgb.shape[1]),
            "gcs_uri": xgb_gcs_uri,
        },
        "lgb_v5": {
            **metrics_lgb,
            "best_iteration": model_lgb.best_iteration_,
            "n_features": int(X_test_eng_lgb.shape[1]),
            "gcs_uri": lgb_gcs_uri,
        },
    }
    out_path = Path("evaluation/v5_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    logger.info("v5_results_saved", extra={"path": str(out_path)})

    return results


if __name__ == "__main__":
    main()
