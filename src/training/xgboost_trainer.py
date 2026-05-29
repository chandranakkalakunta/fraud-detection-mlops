"""
XGBoost fraud detection trainer — Phase 2B.

Trains two variants on 654 engineered features with a time-based 75/25 split:
  Variant A: XGBoost + SMOTE
  Variant B: XGBoost + scale_pos_weight (class-weight rebalancing)

Validation set is carved from the training portion (last val_split %) —
it is never the held-out test set, which would cause information leakage.

All runs are logged to Vertex AI Experiments with identical metric names so
the comparison table is apples-to-apples.  The champion variant (higher auc_pr)
is saved to GCS.
"""

import io
import json
import os
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.features.imbalance import ClassWeightHandler, SMOTEHandler
from src.training.metrics import compute_fraud_metrics
from src.utils.config import load_config
from src.utils.logging import get_logger

logger = get_logger(__name__)

TARGET = "isFraud"
SORT_COL = "TransactionDT"


def load_engineered_features(config: dict) -> pd.DataFrame:
    """Load engineered features from BigQuery transactions_joined table."""
    from google.cloud import bigquery

    project = config["gcp"]["project_id"]
    dataset = config["bigquery"]["dataset"]
    table = config["bigquery"]["tables"]["transactions_joined"]

    # Load all columns — feature engineer will be applied to produce 654 features
    query = f"""
        SELECT *
        FROM `{project}.{dataset}.{table}`
        WHERE {TARGET} IS NOT NULL
        ORDER BY {SORT_COL}
    """

    logger.info("loading_features_from_bq", extra={"table": f"{dataset}.{table}"})
    client = bigquery.Client(project=project)
    df = client.query(query).to_dataframe(progress_bar_type=None)
    logger.info(
        "features_loaded",
        extra={"rows": len(df), "cols": len(df.columns), "fraud_rate": round(float(df[TARGET].mean()), 6)},
    )
    return df


def engineer_features(df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.Series]:
    """Apply feature engineering using the fitted transformer from GCS."""
    from src.features.engineer import build_from_config

    y = df[TARGET].astype(int)
    drop_cols = [TARGET, SORT_COL] if SORT_COL in df.columns else [TARGET]
    X = df.drop(columns=drop_cols)

    sort_idx = df[SORT_COL].argsort().values if SORT_COL in df.columns else np.arange(len(df))
    X = X.iloc[sort_idx].reset_index(drop=True)
    y = y.iloc[sort_idx].reset_index(drop=True)

    return X, y


def time_based_split(
    df: pd.DataFrame,
    train_frac: float = 0.75,
    val_frac: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Sort by TransactionDT and split into train / validation / test.

    Never shuffles — temporal ordering must be preserved.

    val_frac is the fraction of the TRAINING portion held out for validation.
    Example: train_frac=0.75, val_frac=0.20 →
      60% of total data for actual training
      15% of total data for validation (early stopping)
      25% of total data for final test evaluation
    """
    assert 0 < train_frac < 1
    assert 0 < val_frac < 1

    df_sorted = df.sort_values(SORT_COL).reset_index(drop=True)
    n = len(df_sorted)
    train_end = int(n * train_frac)
    val_start = int(train_end * (1 - val_frac))

    train_df = df_sorted.iloc[:val_start].reset_index(drop=True)
    val_df = df_sorted.iloc[val_start:train_end].reset_index(drop=True)
    test_df = df_sorted.iloc[train_end:].reset_index(drop=True)

    logger.info(
        "time_based_split",
        extra={
            "train_rows": len(train_df),
            "val_rows": len(val_df),
            "test_rows": len(test_df),
            "train_fraud_rate": round(float(train_df[TARGET].mean()), 6),
            "val_fraud_rate": round(float(val_df[TARGET].mean()), 6),
            "test_fraud_rate": round(float(test_df[TARGET].mean()), 6),
        },
    )
    return train_df, val_df, test_df


def _prepare_features(df: pd.DataFrame, engineer) -> tuple[pd.DataFrame, pd.Series]:
    """Extract X, y from a split DataFrame and apply feature engineering."""
    y = df[TARGET].astype(int).reset_index(drop=True)
    X = df.drop(columns=[TARGET] + ([SORT_COL] if SORT_COL in df.columns else []))
    X = X.reset_index(drop=True)
    return X, y


def save_model_to_gcs(model: Any, model_name: str, config: dict) -> str:
    """
    Serialise a fitted model to GCS.

    Saves as models/{model_name}/model.joblib so the directory URI
    models/{model_name}/ can be passed to Vertex AI Model Registry as artifact_uri.
    Returns the directory gs:// URI (without the filename).
    """
    from google.cloud import storage as gcs

    bucket_name = config["storage"]["artifacts_bucket"]
    blob_path = f"models/{model_name}/model.joblib"

    buf = io.BytesIO()
    joblib.dump(model, buf)
    buf.seek(0)

    client = gcs.Client(project=config["gcp"]["project_id"])
    client.bucket(bucket_name).blob(blob_path).upload_from_file(
        buf, content_type="application/octet-stream"
    )

    # Return the directory URI — required by Vertex AI Model Registry
    dir_uri = f"gs://{bucket_name}/models/{model_name}/"
    logger.info("model_saved_to_gcs", extra={"uri": dir_uri, "model": model_name})
    return dir_uri


def log_to_vertex_experiments(
    config: dict,
    run_name: str,
    params: dict,
    metrics: dict,
) -> None:
    """Log a training run to Vertex AI Experiments."""
    import google.cloud.aiplatform as aip

    project = config["gcp"]["project_id"]
    location = config["gcp"]["region"]
    experiment_name = config["vertex"]["experiment_name"]

    # Initialise WITHOUT experiment to avoid IPython/Tensorboard widget side-effects.
    # ExperimentRun.create() accepts experiment as a direct argument.
    aip.init(project=project, location=location)

    from google.api_core.exceptions import AlreadyExists

    try:
        run = aip.ExperimentRun.create(
            run_name=run_name,
            experiment=experiment_name,
            project=project,
            location=location,
        )
    except AlreadyExists:
        # Idempotent re-run: get the existing run and set it back to RUNNING
        run = aip.ExperimentRun(
            run_name=run_name,
            experiment=experiment_name,
            project=project,
            location=location,
        )
        run.update_state(aip.gapic.Execution.State.RUNNING)

    run.log_params(params)
    run.log_metrics(metrics)
    run.end_run(state=aip.gapic.Execution.State.COMPLETE)

    logger.info(
        "vertex_experiment_logged",
        extra={"run_name": run_name, "auc_pr": metrics.get("auc_pr")},
    )


def stratified_val_carve(
    train_pool: pd.DataFrame,
    val_frac: float = 0.20,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Stratified holdout carve from the training pool for early-stopping validation.

    The test set boundary is always temporal (time_based_split).  This function
    affects only the train/val split *within* the training window.

    A random stratified split (rather than temporal tail) is used because:
    - Early stopping only needs a stable, representative signal — not ordering.
    - At 3.5% fraud rate a temporal carve may land on a low-fraud time window,
      leaving the stopping callback with almost no positive examples.
    - sklearn stratify= guarantees the dataset fraud rate is preserved in val,
      giving early stopping a reliable logloss surface to optimise against.
    """
    from sklearn.model_selection import train_test_split

    train_df, val_df = train_test_split(
        train_pool,
        test_size=val_frac,
        stratify=train_pool[TARGET],
        random_state=random_state,
        shuffle=True,
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    logger.info(
        "stratified_val_carve",
        extra={
            "train_rows": len(train_df),
            "val_rows": len(val_df),
            "val_fraud_count": int(val_df[TARGET].sum()),
            "val_fraud_rate": round(float(val_df[TARGET].mean()), 6),
        },
    )
    return train_df, val_df


def train_v2_class_weight_fixed(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    config: dict,
    engineer,
) -> tuple[xgb.XGBClassifier, dict]:
    """
    v2 XGBoost: scale_pos_weight + logloss early stopping + stratified val.

    SMOTE removed: the 654-feature matrix contains 253 binary _was_missing
    indicators (from high-null V-columns).  k-NN synthetic sampling in a
    mixed binary+continuous space produces geometrically invalid points —
    interpolating between binary values yields non-integer fractions, erasing
    the missingness signal encoded in those indicators.  scale_pos_weight is
    the correct cost-sensitive alternative for gradient-boosted trees.

    NOTE (future): if SMOTE is reconsidered, apply StandardScaler to continuous
    features only (TransactionAmt, Cx, Dx, Vx columns) before fitting; never
    scale or interpolate the binary _was_missing indicators.

    eval_metric="auc" (ROC-AUC) is used for early stopping signal only.
    - aucpr: too noisy at 3.5% fraud rate; caused best_iteration=13 in v1 SMOTE runs
    - logloss: incompatible with scale_pos_weight — the weighted training loss biases
      predictions toward high fraud probability, which inflates unweighted logloss on
      the 96.5% non-fraud val set immediately, causing best_iteration=0
    - auc: ranking metric; threshold-free; immune to scale_pos_weight's calibration
      effect; moves smoothly upward as the model learns to separate classes
    AUC-PR is still computed from predict_proba() and logged as the primary metric.
    """
    xgb_cfg = config["model"]["xgboost"]

    cw = ClassWeightHandler()
    cw.fit(y_train)

    X_train_eng = engineer.fit_transform(X_train, y_train).select_dtypes(include=[np.number])
    X_val_eng = engineer.transform(X_val).select_dtypes(include=[np.number])

    assert X_train_eng.isnull().sum().sum() == 0, "NaN in train features (v2)"
    assert X_val_eng.isnull().sum().sum() == 0, "NaN in val features (v2)"

    model = xgb.XGBClassifier(
        n_estimators=int(xgb_cfg["n_estimators"]),
        max_depth=int(xgb_cfg["max_depth"]),
        learning_rate=float(xgb_cfg["learning_rate"]),
        tree_method=xgb_cfg["tree_method"],
        early_stopping_rounds=int(xgb_cfg["early_stopping_rounds"]),
        eval_metric="auc",  # ranking metric — immune to scale_pos_weight calibration effect; AUC-PR logged separately
        scale_pos_weight=cw.scale_pos_weight,
        subsample=float(xgb_cfg["subsample"]),
        colsample_bytree=float(xgb_cfg["colsample_bytree"]),
        min_child_weight=int(xgb_cfg["min_child_weight"]),
        random_state=int(config["model"]["random_state"]),
        n_jobs=-1,
        verbosity=0,
    )

    model.fit(
        X_train_eng, y_train,
        eval_set=[(X_val_eng, y_val)],
        verbose=False,
    )

    logger.info(
        "xgb_v2_trained",
        extra={
            "best_iteration": model.best_iteration,
            "scale_pos_weight": round(cw.scale_pos_weight, 4),
            "n_train_rows": len(X_train_eng),
        },
    )
    return model, cw.get_params()


def train_xgb_v4_no_indicators(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    config: dict,
    engineer,
) -> tuple[xgb.XGBClassifier, dict]:
    """
    v4 XGBoost: scale_pos_weight, no V-column indicators, eval_metric='logloss'.

    V-column _was_missing indicators were removed from the engineer entirely —
    see src/features/engineer.py module comment for the diagnostic findings.

    eval_metric='logloss': with a clean feature set (no 200+ near-constant
    indicators) the model's early predictions are better calibrated, so logloss
    on the validation set provides a stable gradient signal.  The stratified
    val carve guarantees 3.5% fraud in val so logloss doesn't collapse
    immediately as it did in v1 (where near-constant indicators biased predictions
    toward 1.0 for all rows, inflating unweighted logloss).
    """
    xgb_cfg = config["model"]["xgboost"]

    cw = ClassWeightHandler()
    cw.fit(y_train)

    X_train_eng = engineer.fit_transform(X_train, y_train).select_dtypes(include=[np.number])
    X_val_eng = engineer.transform(X_val).select_dtypes(include=[np.number])

    assert X_train_eng.isnull().sum().sum() == 0, "NaN in train features (xgb v4)"
    assert X_val_eng.isnull().sum().sum() == 0, "NaN in val features (xgb v4)"

    model = xgb.XGBClassifier(
        n_estimators=int(xgb_cfg["n_estimators"]),
        max_depth=int(xgb_cfg["max_depth"]),
        learning_rate=float(xgb_cfg["learning_rate"]),
        tree_method=xgb_cfg["tree_method"],
        early_stopping_rounds=int(xgb_cfg["early_stopping_rounds"]),
        # eval_metric="auc" not "logloss": scale_pos_weight=27 biases round-1 predictions
    # upward, inflating UNWEIGHTED val logloss on 96.5% non-fraud rows → best_iter=0.
    # Confirmed empirically in v4 run (AUC-ROC=0.4984 at best_iter=0 with logloss).
    # AUC is a ranking metric, immune to this calibration bias.
        eval_metric="auc",
        scale_pos_weight=cw.scale_pos_weight,
        subsample=float(xgb_cfg["subsample"]),
        colsample_bytree=float(xgb_cfg["colsample_bytree"]),
        min_child_weight=int(xgb_cfg["min_child_weight"]),
        random_state=int(config["model"]["random_state"]),
        n_jobs=-1,
        verbosity=0,
    )

    model.fit(
        X_train_eng, y_train,
        eval_set=[(X_val_eng, y_val)],
        verbose=False,
    )

    logger.info(
        "xgb_v4_trained",
        extra={
            "best_iteration": model.best_iteration,
            "n_features": X_train_eng.shape[1],
            "scale_pos_weight": round(cw.scale_pos_weight, 4),
        },
    )
    return model, cw.get_params()


def train_xgb_v3_reduced_indicators(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    config: dict,
    engineer,
) -> tuple[xgb.XGBClassifier, dict]:
    """
    v3 XGBoost: scale_pos_weight + reduced missing indicators + lr=0.1.

    The engineer is built with missing_threshold=0.80, which creates indicators
    only for V-columns with 5–80% null (genuine missingness).  V-columns above
    80% null (near-constant, always-1) get median imputation only — eliminating
    the 253 near-constant indicators that occupied 49% of colsample_bytree=0.8
    budget in v1/v2 and caused B→C AUC-PR delta of −0.364.

    lr=0.1 (vs config 0.05): with fewer noisy features the loss surface is
    smoother; a larger step converges faster without overshooting.
    eval_metric="auc": same reasoning as v2 — immune to scale_pos_weight bias.
    """
    xgb_cfg = config["model"]["xgboost"]

    cw = ClassWeightHandler()
    cw.fit(y_train)

    X_train_eng = engineer.fit_transform(X_train, y_train).select_dtypes(include=[np.number])
    X_val_eng = engineer.transform(X_val).select_dtypes(include=[np.number])

    assert X_train_eng.isnull().sum().sum() == 0, "NaN in train features (xgb v3)"
    assert X_val_eng.isnull().sum().sum() == 0, "NaN in val features (xgb v3)"

    model = xgb.XGBClassifier(
        n_estimators=int(xgb_cfg["n_estimators"]),
        max_depth=int(xgb_cfg["max_depth"]),
        learning_rate=0.1,  # v3: faster convergence with cleaner feature set
        tree_method=xgb_cfg["tree_method"],
        early_stopping_rounds=int(xgb_cfg["early_stopping_rounds"]),
        eval_metric="auc",
        scale_pos_weight=cw.scale_pos_weight,
        subsample=float(xgb_cfg["subsample"]),
        colsample_bytree=float(xgb_cfg["colsample_bytree"]),
        min_child_weight=int(xgb_cfg["min_child_weight"]),
        random_state=int(config["model"]["random_state"]),
        n_jobs=-1,
        verbosity=0,
    )

    model.fit(
        X_train_eng, y_train,
        eval_set=[(X_val_eng, y_val)],
        verbose=False,
    )

    logger.info(
        "xgb_v3_trained",
        extra={
            "best_iteration": model.best_iteration,
            "n_features": X_train_eng.shape[1],
            "scale_pos_weight": round(cw.scale_pos_weight, 4),
        },
    )
    params = cw.get_params()
    params["learning_rate_override"] = 0.1
    return model, params


def train_variant_a_smote(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    config: dict,
    engineer,
) -> tuple[xgb.XGBClassifier, dict]:
    """Variant A: XGBoost + SMOTE."""
    xgb_cfg = config["model"]["xgboost"]
    smote_cfg = config["model"]["smote"]

    smote = SMOTEHandler(
        sampling_strategy=float(smote_cfg["sampling_strategy"]),
        random_state=int(smote_cfg["random_state"]),
        k_neighbors=int(smote_cfg["k_neighbors"]),
    )

    # Apply feature engineering; keep only numeric columns (drops M1-M9, ProductCD etc.)
    X_train_eng = engineer.fit_transform(X_train, y_train).select_dtypes(include=[np.number])
    X_val_eng = engineer.transform(X_val).select_dtypes(include=[np.number])

    assert X_train_eng.isnull().sum().sum() == 0, "NaN in train features (variant A)"
    assert X_val_eng.isnull().sum().sum() == 0, "NaN in val features (variant A)"

    # SMOTE on training data only
    X_train_res, y_train_res = smote.fit_resample(X_train_eng, y_train, train_only=True)

    model = xgb.XGBClassifier(
        n_estimators=int(xgb_cfg["n_estimators"]),
        max_depth=int(xgb_cfg["max_depth"]),
        learning_rate=float(xgb_cfg["learning_rate"]),
        tree_method=xgb_cfg["tree_method"],
        early_stopping_rounds=int(xgb_cfg["early_stopping_rounds"]),
        eval_metric=xgb_cfg["eval_metric"],
        subsample=float(xgb_cfg["subsample"]),
        colsample_bytree=float(xgb_cfg["colsample_bytree"]),
        min_child_weight=int(xgb_cfg["min_child_weight"]),
        random_state=int(config["model"]["random_state"]),
        n_jobs=-1,
        verbosity=0,
    )

    model.fit(
        X_train_res, y_train_res,
        eval_set=[(X_val_eng, y_val)],
        verbose=False,
    )

    logger.info(
        "xgb_variant_a_trained",
        extra={"best_iteration": model.best_iteration, "n_smote_rows": len(X_train_res)},
    )
    return model, smote.get_params()


def train_variant_b_class_weight(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    config: dict,
    engineer,
) -> tuple[xgb.XGBClassifier, dict]:
    """Variant B: XGBoost + scale_pos_weight."""
    xgb_cfg = config["model"]["xgboost"]

    cw = ClassWeightHandler()
    cw.fit(y_train)

    # Apply feature engineering; keep only numeric columns (drops M1-M9, ProductCD etc.)
    X_train_eng = engineer.fit_transform(X_train, y_train).select_dtypes(include=[np.number])
    X_val_eng = engineer.transform(X_val).select_dtypes(include=[np.number])

    assert X_train_eng.isnull().sum().sum() == 0, "NaN in train features (variant B)"
    assert X_val_eng.isnull().sum().sum() == 0, "NaN in val features (variant B)"

    model = xgb.XGBClassifier(
        n_estimators=int(xgb_cfg["n_estimators"]),
        max_depth=int(xgb_cfg["max_depth"]),
        learning_rate=float(xgb_cfg["learning_rate"]),
        tree_method=xgb_cfg["tree_method"],
        early_stopping_rounds=int(xgb_cfg["early_stopping_rounds"]),
        eval_metric=xgb_cfg["eval_metric"],
        scale_pos_weight=cw.scale_pos_weight,
        subsample=float(xgb_cfg["subsample"]),
        colsample_bytree=float(xgb_cfg["colsample_bytree"]),
        min_child_weight=int(xgb_cfg["min_child_weight"]),
        random_state=int(config["model"]["random_state"]),
        n_jobs=-1,
        verbosity=0,
    )

    model.fit(
        X_train_eng, y_train,
        eval_set=[(X_val_eng, y_val)],
        verbose=False,
    )

    logger.info(
        "xgb_variant_b_trained",
        extra={
            "best_iteration": model.best_iteration,
            "scale_pos_weight": round(cw.scale_pos_weight, 4),
        },
    )
    return model, cw.get_params()


def main() -> None:
    from dotenv import load_dotenv
    load_dotenv()

    env = os.getenv("ENV", "dev")
    config = load_config(env)

    # ── Load data ─────────────────────────────────────────────────────────────
    df = load_engineered_features(config)

    train_frac = 0.75
    val_frac = float(config["model"]["val_split"])

    train_df, val_df, test_df = time_based_split(df, train_frac=train_frac, val_frac=val_frac)

    # Feature engineering is applied per-variant to avoid leakage between variants
    # Keep TransactionDT in the feature matrix — the engineer needs it to create
    # time_of_day, day_of_week, hour_of_day features.  After engineering it remains
    # as a pass-through column (consistent with Phase 2A 654-feature output).
    feature_cols = [c for c in df.columns if c not in [TARGET]]
    X_train = train_df[feature_cols].reset_index(drop=True)
    y_train = train_df[TARGET].astype(int).reset_index(drop=True)
    X_val = val_df[feature_cols].reset_index(drop=True)
    y_val = val_df[TARGET].astype(int).reset_index(drop=True)
    X_test = test_df[feature_cols].reset_index(drop=True)
    y_test = test_df[TARGET].astype(int).reset_index(drop=True)

    # ── Variant A: SMOTE ──────────────────────────────────────────────────────
    from src.features.engineer import build_from_config

    engineer_a = build_from_config(config)
    model_a, params_a = train_variant_a_smote(X_train, y_train, X_val, y_val, config, engineer_a)

    X_test_eng_a = engineer_a.transform(X_test).select_dtypes(include=[np.number])
    assert X_test_eng_a.isnull().sum().sum() == 0, "NaN in test features (variant A)"
    y_proba_a = model_a.predict_proba(X_test_eng_a)[:, 1]
    metrics_a = compute_fraud_metrics(
        y_true=y_test.values,
        y_proba=y_proba_a,
        imbalance_strategy="smote",
        training_rows=len(X_train),
        test_rows=len(X_test),
    )

    logger.info("variant_a_evaluated", extra={"auc_pr": metrics_a["auc_pr"]})
    print(f"\nVariant A (SMOTE)  — AUC-PR: {metrics_a['auc_pr']:.4f}  AUC-ROC: {metrics_a['auc_roc']:.4f}")

    # ── Variant B: class weight ────────────────────────────────────────────────
    engineer_b = build_from_config(config)
    model_b, params_b = train_variant_b_class_weight(X_train, y_train, X_val, y_val, config, engineer_b)

    X_test_eng_b = engineer_b.transform(X_test).select_dtypes(include=[np.number])
    assert X_test_eng_b.isnull().sum().sum() == 0, "NaN in test features (variant B)"
    y_proba_b = model_b.predict_proba(X_test_eng_b)[:, 1]
    metrics_b = compute_fraud_metrics(
        y_true=y_test.values,
        y_proba=y_proba_b,
        imbalance_strategy="class_weight",
        training_rows=len(X_train),
        test_rows=len(X_test),
    )

    logger.info("variant_b_evaluated", extra={"auc_pr": metrics_b["auc_pr"]})
    print(f"Variant B (weight) — AUC-PR: {metrics_b['auc_pr']:.4f}  AUC-ROC: {metrics_b['auc_roc']:.4f}")

    # ── Log to Vertex AI Experiments ──────────────────────────────────────────
    xgb_cfg = config["model"]["xgboost"]
    shared_params = {
        "model_type": "xgboost",
        "n_estimators": int(xgb_cfg["n_estimators"]),
        "max_depth": int(xgb_cfg["max_depth"]),
        "learning_rate": float(xgb_cfg["learning_rate"]),
        "tree_method": xgb_cfg["tree_method"],
        "early_stopping_rounds": int(xgb_cfg["early_stopping_rounds"]),
        "train_frac": train_frac,
        "val_frac": val_frac,
    }

    log_to_vertex_experiments(config, "xgb-variant-a-smote",        {**shared_params, **params_a}, metrics_a)
    log_to_vertex_experiments(config, "xgb-variant-b-class-weight",  {**shared_params, **params_b}, metrics_b)

    # ── Select champion (higher auc_pr) ───────────────────────────────────────
    if metrics_a["auc_pr"] >= metrics_b["auc_pr"]:
        champion_model = model_a
        champion_engineer = engineer_a
        champion_metrics = metrics_a
        champion_name = "xgb-variant-a-smote"
    else:
        champion_model = model_b
        champion_engineer = engineer_b
        champion_metrics = metrics_b
        champion_name = "xgb-variant-b-class-weight"

    print(f"\nXGBoost champion: {champion_name}  (AUC-PR: {champion_metrics['auc_pr']:.4f})")

    # ── Save champion to GCS ──────────────────────────────────────────────────
    gcs_uri = save_model_to_gcs(champion_model, champion_name, config)

    # Write evaluation JSON
    output = {
        "champion": champion_name,
        "variant_a": {**metrics_a},
        "variant_b": {**metrics_b},
        "champion_gcs_uri": gcs_uri,
    }
    out_path = Path(config["model"]["xgboost"]["evaluation_output"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    logger.info("xgboost_results_saved", extra={"path": str(out_path)})

    print(f"Champion saved to: {gcs_uri}")
    return champion_name, champion_model, champion_metrics, gcs_uri


if __name__ == "__main__":
    main()
