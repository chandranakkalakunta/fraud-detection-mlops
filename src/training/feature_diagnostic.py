"""
Feature engineering diagnostic — Phase 2B.

Trains XGBoost with fixed hyperparameters on three feature sets to isolate
whether feature engineering pipeline steps improve on raw features.

Feature sets:
  Run A: Raw numeric columns from BigQuery, simple median imputation (~392 cols)
  Run B: Run A + 9 time/velocity/D-norm features (explicit additions only)
  Run C: Full 654-column FraudFeatureEngineer pipeline

Identical hyperparameters across all 3 runs:
  n_estimators=300, max_depth=6, learning_rate=0.1, scale_pos_weight=27
  eval_metric="auc", early_stopping_rounds=100, stratified val

IMPORTANT — eval_metric="auc" not logloss:
  scale_pos_weight=27 biases model predictions strongly toward 1.0 for all
  samples.  Unweighted logloss on the 96.5% non-fraud val set immediately
  worsens after tree 1 → early stopping fires at best_iter=0, leaving an
  untrained model.  AUC (ranking metric) is immune to this calibration
  effect and allows the full training budget to be used.

Also includes: card4_te LOO encoding correctness and leakage check.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.training.metrics import compute_fraud_metrics
from src.training.xgboost_trainer import (
    SORT_COL,
    TARGET,
    load_engineered_features,
    log_to_vertex_experiments,
    stratified_val_carve,
)
from src.utils.config import load_config
from src.utils.logging import get_logger

logger = get_logger(__name__)

_D_NORM_COLS  = ["D1", "D4", "D10"]
_D_GROUP_COL  = "card1"


# ── Feature set preparation ───────────────────────────────────────────────────

def _raw_numeric_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return numeric column names excluding the target label."""
    return [c for c in df.select_dtypes(include=[np.number]).columns if c != TARGET]


def prepare_run_a(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run A: raw numeric columns with training-set median imputation.

    Excludes string columns automatically (card4, card6, ProductCD, M1-M9,
    P/R_emaildomain, DeviceType, DeviceInfo, string id_ columns).
    Captures ~392 numeric columns from the BigQuery raw join table.
    """
    feature_cols = _raw_numeric_feature_cols(train_df)

    X_train = train_df[feature_cols].copy()
    X_val   = val_df[feature_cols].copy()
    X_test  = test_df[feature_cols].copy()

    # Impute with training medians only — no look-ahead
    medians = X_train.median().fillna(0.0)
    X_train = X_train.fillna(medians).reset_index(drop=True)
    X_val   = X_val.fillna(medians).reset_index(drop=True)
    X_test  = X_test.fillna(medians).reset_index(drop=True)

    logger.info("run_a_prepared", extra={"n_features": X_train.shape[1]})
    return X_train, X_val, X_test


def _add_9_features(X: pd.DataFrame, d_group_medians: dict) -> pd.DataFrame:
    """
    Add 9 time/velocity/D-norm features to an already-imputed numeric DataFrame.

    d_group_medians: {col: pd.Series(index=card1)} fitted on training data.
    All source values come from X (already median-imputed by prepare_run_a).
    """
    X = X.copy()

    # 3 time features
    dt = X[SORT_COL]
    X["time_of_day"]  = (dt % 86400).astype(np.int32)
    X["day_of_week"]  = ((dt // 86400) % 7).astype(np.int32)
    X["hour_of_day"]  = ((dt % 86400) // 3600).astype(np.int32)

    # 3 amount / velocity features
    amt = X["TransactionAmt"]
    X["log_transaction_amt"] = np.log1p(amt)
    X["amt_to_d1_ratio"]     = amt / (X["D1"] + 1)  # D1 already imputed
    X["is_round_amt"]        = (amt % 1 == 0).astype(np.int8)

    # 3 D-norm features (per-card-median subtraction, fitted on training only)
    for col in _D_NORM_COLS:
        if col in X.columns and _D_GROUP_COL in X.columns and col in d_group_medians:
            card_med = X[_D_GROUP_COL].map(d_group_medians[col])
            X[f"{col}_card_norm"] = (X[col] - card_med).fillna(0.0)

    return X


def prepare_run_b(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run B: Run A + 9 explicit time/velocity/D-norm features.

    No missing indicators, no target encoding. Isolates whether these
    lower-risk engineered features add signal over raw columns.
    """
    X_train, X_val, X_test = prepare_run_a(train_df, val_df, test_df)

    # Fit D-norm group medians on training raw data only
    d_group_medians: dict[str, pd.Series] = {}
    for col in _D_NORM_COLS:
        if col in train_df.columns and _D_GROUP_COL in train_df.columns:
            d_group_medians[col] = (
                train_df.groupby(_D_GROUP_COL, sort=False)[col].median()
            )

    X_train = _add_9_features(X_train, d_group_medians)
    X_val   = _add_9_features(X_val,   d_group_medians)
    X_test  = _add_9_features(X_test,  d_group_medians)

    logger.info("run_b_prepared", extra={"n_features": X_train.shape[1]})
    return X_train, X_val, X_test


def prepare_run_c(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, object]:
    """
    Run C: full FraudFeatureEngineer pipeline (654 features).

    Returns the fitted engineer for the leakage check.
    """
    from src.features.engineer import build_from_config

    feature_cols = [c for c in train_df.columns if c != TARGET]
    y_train = train_df[TARGET].astype(int).reset_index(drop=True)

    engineer = build_from_config(config)

    X_train_eng = engineer.fit_transform(
        train_df[feature_cols].reset_index(drop=True), y_train
    ).select_dtypes(include=[np.number])

    X_val_eng = engineer.transform(
        val_df[feature_cols].reset_index(drop=True)
    ).select_dtypes(include=[np.number])

    X_test_eng = engineer.transform(
        test_df[feature_cols].reset_index(drop=True)
    ).select_dtypes(include=[np.number])

    logger.info("run_c_prepared", extra={"n_features": X_train_eng.shape[1]})
    return (
        X_train_eng.reset_index(drop=True),
        X_val_eng.reset_index(drop=True),
        X_test_eng.reset_index(drop=True),
        engineer,
    )


# ── Training ──────────────────────────────────────────────────────────────────

def train_diagnostic_run(
    run_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    config: dict,
) -> dict:
    """
    Train XGBoost with fixed diagnostic hyperparameters and log to Vertex AI.

    eval_metric="auc" (see module docstring for why logloss is not used here).
    """
    assert X_train.isnull().sum().sum() == 0, f"NaN in {run_name} train features"
    assert X_val.isnull().sum().sum()   == 0, f"NaN in {run_name} val features"
    assert X_test.isnull().sum().sum()  == 0, f"NaN in {run_name} test features"

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=27,
        early_stopping_rounds=100,
        eval_metric="auc",  # see module docstring — logloss incompatible with scale_pos_weight
        tree_method="hist",
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = compute_fraud_metrics(
        y_true=y_test.values,
        y_proba=y_proba,
        imbalance_strategy="scale_pos_weight",
        training_rows=len(X_train),
        test_rows=len(X_test),
    )

    logger.info(
        "diagnostic_run_complete",
        extra={
            "run_name": run_name,
            "n_features": X_train.shape[1],
            "best_iteration": model.best_iteration,
            "auc_pr": round(metrics["auc_pr"], 4),
            "auc_roc": round(metrics["auc_roc"], 4),
        },
    )

    log_to_vertex_experiments(
        config,
        run_name,
        {
            "model_type": "xgboost_diagnostic",
            "feature_set": run_name,
            "n_features": X_train.shape[1],
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.1,
            "scale_pos_weight": 27,
            "early_stopping_rounds": 100,
            "early_stopping_metric": "auc",
            "val_strategy": "stratified",
            "best_iteration": model.best_iteration,
        },
        metrics,
    )

    return {"model": model, "metrics": metrics, "n_features": X_train.shape[1]}


# ── card4_te leakage check ────────────────────────────────────────────────────

def check_target_encoding_leakage(
    X_train_eng: pd.DataFrame,
    y_train: pd.Series,
    X_test_eng: pd.DataFrame,
    engineer,
) -> None:
    """
    Verify LOO target encoding correctness and absence of leakage.

    Correct LOO: card4_te in training ≈ per-category fraud rate. Each row
    excludes its own label so values cluster around the category mean with
    per-row variation of ±1/category_count — essentially invisible for large
    categories (visa/mastercard have ~100K rows each).

    No leakage: test card4_te values should not exceed the training range
    because unseen categories fall back to global_mean (smoothed).
    """
    if "card4_te" not in X_train_eng.columns or "card4_te" not in X_test_eng.columns:
        print("\ncard4_te not present in feature set — leakage check skipped")
        return

    print(f"\n{'─'*64}")
    print("card4_te LOO encoding — correctness and leakage check")
    print(f"{'─'*64}")

    # ── Category stats from fitted engineer ───────────────────────────────────
    if "card4" in engineer._cat_stats:
        stats       = engineer._cat_stats["card4"]
        global_mean = engineer._global_mean
        smoothing   = engineer.smoothing

        cat_fraud_rate = stats["sum"] / stats["count"]
        print(f"\n  Fitted category stats (training data, LOO-excluded):")
        print(f"  {'card4':<22} {'count':>10} {'train_fraud_rate':>17} {'test_te_smoothed':>17}")
        print(f"  {'─'*22} {'─'*10} {'─'*17} {'─'*17}")

        for cat in sorted(stats.index, key=lambda c: -float(cat_fraud_rate.get(c, 0))):
            count   = int(stats.loc[cat, "count"])
            rate    = float(cat_fraud_rate[cat])
            w       = count / (count + smoothing)
            te_test = w * rate + (1.0 - w) * global_mean
            print(f"  {str(cat):<22} {count:>10,} {rate:>17.6f} {te_test:>17.6f}")

        print(f"  {'unseen (global_mean)':<22} {'—':>10} {global_mean:>17.6f} {global_mean:>17.6f}")

    # ── Top-5 card4_te values in training vs actual fraud rate ────────────────
    check = X_train_eng[["card4_te"]].copy()
    check["isFraud"] = y_train.values

    summary = (
        check
        .assign(te_key=lambda d: d["card4_te"].round(6))
        .groupby("te_key")
        .agg(actual_fraud_rate=("isFraud", "mean"), rows=("isFraud", "count"))
        .sort_values("te_key", ascending=False)
        .head(5)
        .reset_index()
    )

    print(f"\n  Top 5 card4_te values in training (by value) vs actual isFraud mean:")
    print(f"  {'card4_te':>12}  {'actual_fraud_rate':>18}  {'rows':>10}  {'gap (te − actual)':>18}")
    for _, row in summary.iterrows():
        gap = float(row["te_key"]) - float(row["actual_fraud_rate"])
        print(
            f"  {row['te_key']:>12.6f}"
            f"  {row['actual_fraud_rate']:>18.6f}"
            f"  {int(row['rows']):>10,}"
            f"  {gap:>+18.6f}"
        )
    print(
        "\n  Expected: card4_te ≈ actual_fraud_rate (LOO accuracy)."
        "\n  Tiny gap (< 1/category_count) is correct — each row excludes its own label."
    )

    # ── Range check: test vs training ─────────────────────────────────────────
    tr_min = float(X_train_eng["card4_te"].min())
    tr_max = float(X_train_eng["card4_te"].max())
    te_min = float(X_test_eng["card4_te"].min())
    te_max = float(X_test_eng["card4_te"].max())

    print(f"\n  card4_te range — training : [{tr_min:.6f}, {tr_max:.6f}]")
    print(f"  card4_te range — test     : [{te_min:.6f}, {te_max:.6f}]")

    # Test-time uses smoothed means; unseen → global_mean.
    # Values slightly outside training range are OK due to smoothing compression.
    # Only flag if test range exceeds training range by > smoothing effect.
    eps = 1e-4
    if te_min < tr_min - eps or te_max > tr_max + eps:
        print("  WARNING: test values outside training range — potential leakage")
    else:
        print("  OK: test values within training range — no leakage signal")

    print(f"{'─'*64}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> dict:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=".env")

    env = os.getenv("ENV", "dev")
    config = load_config(env)

    # ── Load data ─────────────────────────────────────────────────────────────
    df = load_engineered_features(config)

    train_frac   = 0.75
    val_frac     = float(config["model"]["val_split"])
    random_state = int(config["model"]["random_state"])

    df_sorted  = df.sort_values(SORT_COL).reset_index(drop=True)
    n          = len(df_sorted)
    train_end  = int(n * train_frac)
    train_pool = df_sorted.iloc[:train_end]
    test_df    = df_sorted.iloc[train_end:].reset_index(drop=True)

    train_df, val_df = stratified_val_carve(
        train_pool, val_frac=val_frac, random_state=random_state
    )

    y_train = train_df[TARGET].astype(int).reset_index(drop=True)
    y_val   = val_df[TARGET].astype(int).reset_index(drop=True)
    y_test  = test_df[TARGET].astype(int).reset_index(drop=True)

    print(f"\nDiagnostic split:")
    print(f"  Train : {len(train_df):>7,}  ({y_train.mean():.3%} fraud)")
    print(f"  Val   : {len(val_df):>7,}  ({y_val.mean():.3%} fraud)  stratified ({int(y_val.sum())} fraud rows)")
    print(f"  Test  : {len(test_df):>7,}  ({y_test.mean():.3%} fraud)  temporal")
    print(f"\nHyperparams: n_estimators=300, max_depth=6, lr=0.1, scale_pos_weight=27")
    print(f"             eval_metric=auc (not logloss — see module docstring)")
    print(f"             early_stopping_rounds=100, val=stratified")

    results: dict = {}

    # ── Run A: raw numerics ───────────────────────────────────────────────────
    print("\n" + "─"*64)
    print("[Run A] Raw numeric features — simple median imputation")
    X_train_a, X_val_a, X_test_a = prepare_run_a(train_df, val_df, test_df)
    print(f"  {X_train_a.shape[1]} features")
    r_a = train_diagnostic_run(
        "diag-run-a-raw-features",
        X_train_a, y_train, X_val_a, y_val, X_test_a, y_test, config,
    )
    results["run_a"] = r_a
    print(
        f"  best_iter={r_a['model'].best_iteration}"
        f"  AUC-PR={r_a['metrics']['auc_pr']:.4f}"
        f"  AUC-ROC={r_a['metrics']['auc_roc']:.4f}"
    )

    # ── Run B: raw + 9 features ───────────────────────────────────────────────
    print("\n" + "─"*64)
    print("[Run B] Raw + 9 time/velocity/D-norm features")
    X_train_b, X_val_b, X_test_b = prepare_run_b(train_df, val_df, test_df)
    print(f"  {X_train_b.shape[1]} features  (+{X_train_b.shape[1] - X_train_a.shape[1]} new)")
    r_b = train_diagnostic_run(
        "diag-run-b-time-velocity",
        X_train_b, y_train, X_val_b, y_val, X_test_b, y_test, config,
    )
    results["run_b"] = r_b
    print(
        f"  best_iter={r_b['model'].best_iteration}"
        f"  AUC-PR={r_b['metrics']['auc_pr']:.4f}"
        f"  AUC-ROC={r_b['metrics']['auc_roc']:.4f}"
    )

    # ── Run C: full engineer ──────────────────────────────────────────────────
    print("\n" + "─"*64)
    print("[Run C] Full FraudFeatureEngineer pipeline")
    X_train_c, X_val_c, X_test_c, engineer_c = prepare_run_c(
        train_df, val_df, test_df, config
    )
    print(f"  {X_train_c.shape[1]} features  (+{X_train_c.shape[1] - X_train_b.shape[1]} vs Run B)")
    r_c = train_diagnostic_run(
        "diag-run-c-full-pipeline",
        X_train_c, y_train, X_val_c, y_val, X_test_c, y_test, config,
    )
    results["run_c"] = r_c
    print(
        f"  best_iter={r_c['model'].best_iteration}"
        f"  AUC-PR={r_c['metrics']['auc_pr']:.4f}"
        f"  AUC-ROC={r_c['metrics']['auc_roc']:.4f}"
    )

    # ── card4_te leakage check ─────────────────────────────────────────────────
    check_target_encoding_leakage(X_train_c, y_train, X_test_c, engineer_c)

    # ── Comparison table ──────────────────────────────────────────────────────
    print(f"\n{'═'*80}")
    print("FEATURE DIAGNOSTIC — COMPARISON TABLE")
    print(f"{'═'*80}")
    print(f"  {'Run':<38} {'Features':>10} {'best_iter':>10} {'AUC-PR':>8} {'AUC-ROC':>8}")
    print(f"{'─'*80}")
    rows = [
        ("A: raw numeric (~392 BigQuery cols)",    r_a, X_train_a.shape[1]),
        ("B: raw + 9 time/velocity/D-norm",        r_b, X_train_b.shape[1]),
        ("C: full FraudFeatureEngineer (654)",      r_c, X_train_c.shape[1]),
    ]
    for label, r, nf in rows:
        print(
            f"  {label:<38} {nf:>10} {r['model'].best_iteration:>10}"
            f" {r['metrics']['auc_pr']:>8.4f} {r['metrics']['auc_roc']:>8.4f}"
        )
    print(f"{'─'*80}")
    print(f"  {'Logistic Regression baseline':<38} {'N/A':>10} {'N/A':>10} {'0.2172':>8} {'0.8600':>8}")
    print(f"{'═'*80}")

    # ── Interpretation hint ────────────────────────────────────────────────────
    a_pr = r_a["metrics"]["auc_pr"]
    b_pr = r_b["metrics"]["auc_pr"]
    c_pr = r_c["metrics"]["auc_pr"]
    print(f"\nDiagnosis:")
    print(f"  A→B delta : {b_pr - a_pr:+.4f}  (time/velocity/D-norm contribution)")
    print(f"  B→C delta : {c_pr - b_pr:+.4f}  (missing indicators + target encoding contribution)")
    print(f"  A→LR      : {0.2172 - a_pr:+.4f}  (gap between raw-feature XGBoost and LR baseline)")

    # Save
    out = {
        "run_a": {**r_a["metrics"], "best_iteration": r_a["model"].best_iteration, "n_features": r_a["n_features"]},
        "run_b": {**r_b["metrics"], "best_iteration": r_b["model"].best_iteration, "n_features": r_b["n_features"]},
        "run_c": {**r_c["metrics"], "best_iteration": r_c["model"].best_iteration, "n_features": r_c["n_features"]},
    }
    out_path = Path("evaluation/feature_diagnostic_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    logger.info("diagnostic_results_saved", extra={"path": str(out_path)})

    return results


if __name__ == "__main__":
    main()
