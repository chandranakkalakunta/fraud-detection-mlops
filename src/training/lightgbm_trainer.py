"""
LightGBM fraud detection trainer — Phase 2B.

Trains two variants on 654 engineered features with a time-based 75/25 split:
  Variant C: LightGBM + SMOTE
  Variant D: LightGBM + is_unbalance=True

Same split and validation carve-out as XGBoost trainer — all 4 variants are
directly comparable in the Vertex AI Experiments table.
"""

import io
import json
import os
import sys
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.features.imbalance import ClassWeightHandler, SMOTEHandler
from src.training.metrics import compute_fraud_metrics
from src.training.xgboost_trainer import (
    TARGET,
    SORT_COL,
    load_engineered_features,
    log_to_vertex_experiments,
    save_model_to_gcs,
    stratified_val_carve,
    time_based_split,
)
from src.utils.config import load_config
from src.utils.logging import get_logger

logger = get_logger(__name__)


def train_variant_c_smote(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    config: dict,
    engineer,
) -> tuple[lgb.LGBMClassifier, dict]:
    """Variant C: LightGBM + SMOTE."""
    lgb_cfg = config["model"]["lightgbm"]
    smote_cfg = config["model"]["smote"]

    smote = SMOTEHandler(
        sampling_strategy=float(smote_cfg["sampling_strategy"]),
        random_state=int(smote_cfg["random_state"]),
        k_neighbors=int(smote_cfg["k_neighbors"]),
    )

    X_train_eng = engineer.fit_transform(X_train, y_train).select_dtypes(include=[np.number])
    X_val_eng = engineer.transform(X_val).select_dtypes(include=[np.number])

    assert X_train_eng.isnull().sum().sum() == 0, "NaN in train features (variant C)"
    assert X_val_eng.isnull().sum().sum() == 0, "NaN in val features (variant C)"

    X_train_res, y_train_res = smote.fit_resample(X_train_eng, y_train, train_only=True)

    callbacks = [lgb.early_stopping(int(lgb_cfg["early_stopping_rounds"]), verbose=False)]

    model = lgb.LGBMClassifier(
        num_leaves=int(lgb_cfg["num_leaves"]),
        learning_rate=float(lgb_cfg["learning_rate"]),
        n_estimators=int(lgb_cfg["n_estimators"]),
        min_child_samples=int(lgb_cfg["min_child_samples"]),
        feature_fraction=float(lgb_cfg["feature_fraction"]),
        bagging_fraction=float(lgb_cfg["bagging_fraction"]),
        bagging_freq=int(lgb_cfg["bagging_freq"]),
        metric=lgb_cfg["metric"],
        random_state=int(config["model"]["random_state"]),
        n_jobs=-1,
        verbose=-1,
    )

    model.fit(
        X_train_res, y_train_res,
        eval_set=[(X_val_eng, y_val)],
        callbacks=callbacks,
    )

    logger.info(
        "lgb_variant_c_trained",
        extra={"best_iteration": model.best_iteration_, "n_smote_rows": len(X_train_res)},
    )
    return model, smote.get_params()


def train_variant_d_is_unbalance(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    config: dict,
    engineer,
) -> tuple[lgb.LGBMClassifier, dict]:
    """Variant D: LightGBM + is_unbalance=True."""
    lgb_cfg = config["model"]["lightgbm"]

    # Compute class weight for logging purposes (not passed to model — is_unbalance handles it)
    cw = ClassWeightHandler()
    cw.fit(y_train)

    X_train_eng = engineer.fit_transform(X_train, y_train).select_dtypes(include=[np.number])
    X_val_eng = engineer.transform(X_val).select_dtypes(include=[np.number])

    assert X_train_eng.isnull().sum().sum() == 0, "NaN in train features (variant D)"
    assert X_val_eng.isnull().sum().sum() == 0, "NaN in val features (variant D)"

    callbacks = [lgb.early_stopping(int(lgb_cfg["early_stopping_rounds"]), verbose=False)]

    model = lgb.LGBMClassifier(
        num_leaves=int(lgb_cfg["num_leaves"]),
        learning_rate=float(lgb_cfg["learning_rate"]),
        n_estimators=int(lgb_cfg["n_estimators"]),
        min_child_samples=int(lgb_cfg["min_child_samples"]),
        feature_fraction=float(lgb_cfg["feature_fraction"]),
        bagging_fraction=float(lgb_cfg["bagging_fraction"]),
        bagging_freq=int(lgb_cfg["bagging_freq"]),
        is_unbalance=True,
        metric=lgb_cfg["metric"],
        random_state=int(config["model"]["random_state"]),
        n_jobs=-1,
        verbose=-1,
    )

    model.fit(
        X_train_eng, y_train,
        eval_set=[(X_val_eng, y_val)],
        callbacks=callbacks,
    )

    logger.info(
        "lgb_variant_d_trained",
        extra={"best_iteration": model.best_iteration_},
    )
    params = cw.get_params()
    params["lgb_is_unbalance"] = True
    params["imbalance_strategy"] = "lgb_is_unbalance"
    return model, params


def train_lgb_v2_isunbalance_fixed(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    config: dict,
    engineer,
) -> tuple[lgb.LGBMClassifier, dict]:
    """
    v2 LightGBM: is_unbalance + binary_logloss early stopping + stratified val.

    SMOTE removed — same reasoning as XGBoost v2: 253 binary _was_missing
    indicators in the feature matrix make k-NN synthetic sampling invalid.
    is_unbalance=True is LightGBM-native cost-sensitive weighting equivalent
    to scale_pos_weight = n_neg / n_pos.

    metric="auc" (ROC-AUC) for early stopping signal only.
    - average_precision: noisy at 3.5% fraud rate; caused premature stopping in v1
    - binary_logloss: incompatible with is_unbalance — weighted training biases
      probabilities upward, inflating unweighted logloss on val immediately (best_iter=1)
    - auc: ranking metric; immune to is_unbalance calibration effect; stable signal
    AUC-PR is computed from predict_proba() and logged as the primary metric.
    """
    lgb_cfg = config["model"]["lightgbm"]

    cw = ClassWeightHandler()
    cw.fit(y_train)

    X_train_eng = engineer.fit_transform(X_train, y_train).select_dtypes(include=[np.number])
    X_val_eng = engineer.transform(X_val).select_dtypes(include=[np.number])

    assert X_train_eng.isnull().sum().sum() == 0, "NaN in train features (lgb v2)"
    assert X_val_eng.isnull().sum().sum() == 0, "NaN in val features (lgb v2)"

    callbacks = [lgb.early_stopping(int(lgb_cfg["early_stopping_rounds"]), verbose=False)]

    model = lgb.LGBMClassifier(
        num_leaves=int(lgb_cfg["num_leaves"]),
        learning_rate=float(lgb_cfg["learning_rate"]),
        n_estimators=int(lgb_cfg["n_estimators"]),
        min_child_samples=int(lgb_cfg["min_child_samples"]),
        feature_fraction=float(lgb_cfg["feature_fraction"]),
        bagging_fraction=float(lgb_cfg["bagging_fraction"]),
        bagging_freq=int(lgb_cfg["bagging_freq"]),
        is_unbalance=True,
        metric="auc",  # ranking metric — immune to is_unbalance calibration effect; AUC-PR logged separately
        random_state=int(config["model"]["random_state"]),
        n_jobs=-1,
        verbose=-1,
    )

    model.fit(
        X_train_eng, y_train,
        eval_set=[(X_val_eng, y_val)],
        callbacks=callbacks,
    )

    logger.info("lgb_v2_trained", extra={"best_iteration": model.best_iteration_})
    params = cw.get_params()
    params["lgb_is_unbalance"] = True
    params["imbalance_strategy"] = "lgb_is_unbalance"
    return model, params


def train_lgb_v4_no_indicators(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    config: dict,
    engineer,
) -> tuple[lgb.LGBMClassifier, dict]:
    """
    v4 LightGBM: is_unbalance + no V-column indicators + min_child_samples=100.

    metric="auc": binary_logloss is incompatible with is_unbalance=True (upweighted
    training probabilities inflate unweighted val logloss immediately → best_iter=1).
    AUC is a ranking metric, immune to this calibration bias.
    AUC-PR is computed post-training from predict_proba() as the primary metric.
    """
    lgb_cfg = config["model"]["lightgbm"]

    cw = ClassWeightHandler()
    cw.fit(y_train)

    X_train_eng = engineer.fit_transform(X_train, y_train).select_dtypes(include=[np.number])
    X_val_eng = engineer.transform(X_val).select_dtypes(include=[np.number])

    assert X_train_eng.isnull().sum().sum() == 0, "NaN in train features (lgb v4)"
    assert X_val_eng.isnull().sum().sum() == 0, "NaN in val features (lgb v4)"

    callbacks = [lgb.early_stopping(int(lgb_cfg["early_stopping_rounds"]), verbose=False)]

    model = lgb.LGBMClassifier(
        num_leaves=int(lgb_cfg["num_leaves"]),
        learning_rate=float(lgb_cfg["learning_rate"]),
        n_estimators=int(lgb_cfg["n_estimators"]),
        min_child_samples=100,
        feature_fraction=float(lgb_cfg["feature_fraction"]),
        bagging_fraction=float(lgb_cfg["bagging_fraction"]),
        bagging_freq=int(lgb_cfg["bagging_freq"]),
        is_unbalance=True,
        metric="auc",
        random_state=int(config["model"]["random_state"]),
        n_jobs=-1,
        verbose=-1,
    )

    model.fit(
        X_train_eng, y_train,
        eval_set=[(X_val_eng, y_val)],
        callbacks=callbacks,
    )

    logger.info(
        "lgb_v4_trained",
        extra={
            "best_iteration": model.best_iteration_,
            "n_features": X_train_eng.shape[1],
        },
    )
    params = cw.get_params()
    params["lgb_is_unbalance"] = True
    params["imbalance_strategy"] = "lgb_is_unbalance"
    params["min_child_samples"] = 100
    return model, params


def train_lgb_v3_reduced_indicators(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    config: dict,
    engineer,
) -> tuple[lgb.LGBMClassifier, dict]:
    """
    v3 LightGBM: is_unbalance + reduced missing indicators + min_child_samples=100.

    min_child_samples raised 20 → 100: with is_unbalance=True, each fraud row is
    upweighted by ~28x (n_neg/n_pos).  With min_child_samples=20, the leaf
    membership threshold is effectively 1 real fraud row (20/28 ≈ 0.7), allowing
    fraud-only micro-leaves that overfit.  100 requires ~4 real fraud rows per leaf,
    preventing this collapse and giving the AUC curve time to stabilise.

    metric="auc": same as lgb v2 — immune to is_unbalance calibration bias.
    """
    lgb_cfg = config["model"]["lightgbm"]

    cw = ClassWeightHandler()
    cw.fit(y_train)

    X_train_eng = engineer.fit_transform(X_train, y_train).select_dtypes(include=[np.number])
    X_val_eng = engineer.transform(X_val).select_dtypes(include=[np.number])

    assert X_train_eng.isnull().sum().sum() == 0, "NaN in train features (lgb v3)"
    assert X_val_eng.isnull().sum().sum() == 0, "NaN in val features (lgb v3)"

    callbacks = [lgb.early_stopping(int(lgb_cfg["early_stopping_rounds"]), verbose=False)]

    model = lgb.LGBMClassifier(
        num_leaves=int(lgb_cfg["num_leaves"]),
        learning_rate=float(lgb_cfg["learning_rate"]),
        n_estimators=int(lgb_cfg["n_estimators"]),
        min_child_samples=100,  # v3: prevents fraud micro-leaves under is_unbalance=True
        feature_fraction=float(lgb_cfg["feature_fraction"]),
        bagging_fraction=float(lgb_cfg["bagging_fraction"]),
        bagging_freq=int(lgb_cfg["bagging_freq"]),
        is_unbalance=True,
        metric="auc",
        random_state=int(config["model"]["random_state"]),
        n_jobs=-1,
        verbose=-1,
    )

    model.fit(
        X_train_eng, y_train,
        eval_set=[(X_val_eng, y_val)],
        callbacks=callbacks,
    )

    logger.info(
        "lgb_v3_trained",
        extra={
            "best_iteration": model.best_iteration_,
            "n_features": X_train_eng.shape[1],
        },
    )
    params = cw.get_params()
    params["lgb_is_unbalance"] = True
    params["imbalance_strategy"] = "lgb_is_unbalance"
    params["min_child_samples_override"] = 100
    return model, params


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

    feature_cols = [c for c in df.columns if c not in [TARGET]]
    X_train = train_df[feature_cols].reset_index(drop=True)
    y_train = train_df[TARGET].astype(int).reset_index(drop=True)
    X_val = val_df[feature_cols].reset_index(drop=True)
    y_val = val_df[TARGET].astype(int).reset_index(drop=True)
    X_test = test_df[feature_cols].reset_index(drop=True)
    y_test = test_df[TARGET].astype(int).reset_index(drop=True)

    # ── Variant C: SMOTE ──────────────────────────────────────────────────────
    from src.features.engineer import build_from_config

    engineer_c = build_from_config(config)
    model_c, params_c = train_variant_c_smote(X_train, y_train, X_val, y_val, config, engineer_c)

    X_test_eng_c = engineer_c.transform(X_test).select_dtypes(include=[np.number])
    assert X_test_eng_c.isnull().sum().sum() == 0, "NaN in test features (variant C)"
    y_proba_c = model_c.predict_proba(X_test_eng_c)[:, 1]
    metrics_c = compute_fraud_metrics(
        y_true=y_test.values,
        y_proba=y_proba_c,
        imbalance_strategy="smote",
        training_rows=len(X_train),
        test_rows=len(X_test),
    )

    logger.info("variant_c_evaluated", extra={"auc_pr": metrics_c["auc_pr"]})
    print(f"\nVariant C (LGB+SMOTE)     — AUC-PR: {metrics_c['auc_pr']:.4f}  AUC-ROC: {metrics_c['auc_roc']:.4f}")

    # ── Variant D: is_unbalance ────────────────────────────────────────────────
    engineer_d = build_from_config(config)
    model_d, params_d = train_variant_d_is_unbalance(X_train, y_train, X_val, y_val, config, engineer_d)

    X_test_eng_d = engineer_d.transform(X_test).select_dtypes(include=[np.number])
    assert X_test_eng_d.isnull().sum().sum() == 0, "NaN in test features (variant D)"
    y_proba_d = model_d.predict_proba(X_test_eng_d)[:, 1]
    metrics_d = compute_fraud_metrics(
        y_true=y_test.values,
        y_proba=y_proba_d,
        imbalance_strategy="lgb_is_unbalance",
        training_rows=len(X_train),
        test_rows=len(X_test),
    )

    logger.info("variant_d_evaluated", extra={"auc_pr": metrics_d["auc_pr"]})
    print(f"Variant D (LGB+unbalance) — AUC-PR: {metrics_d['auc_pr']:.4f}  AUC-ROC: {metrics_d['auc_roc']:.4f}")

    # ── Log to Vertex AI Experiments ──────────────────────────────────────────
    lgb_cfg = config["model"]["lightgbm"]
    shared_params = {
        "model_type": "lightgbm",
        "num_leaves": int(lgb_cfg["num_leaves"]),
        "learning_rate": float(lgb_cfg["learning_rate"]),
        "n_estimators": int(lgb_cfg["n_estimators"]),
        "early_stopping_rounds": int(lgb_cfg["early_stopping_rounds"]),
        "train_frac": train_frac,
        "val_frac": val_frac,
    }

    log_to_vertex_experiments(config, "lgb-variant-c-smote",       {**shared_params, **params_c}, metrics_c)
    log_to_vertex_experiments(config, "lgb-variant-d-is-unbalance", {**shared_params, **params_d}, metrics_d)

    # ── Select champion (higher auc_pr) ───────────────────────────────────────
    if metrics_c["auc_pr"] >= metrics_d["auc_pr"]:
        champion_model = model_c
        champion_engineer = engineer_c
        champion_metrics = metrics_c
        champion_name = "lgb-variant-c-smote"
    else:
        champion_model = model_d
        champion_engineer = engineer_d
        champion_metrics = metrics_d
        champion_name = "lgb-variant-d-is-unbalance"

    print(f"\nLightGBM champion: {champion_name}  (AUC-PR: {champion_metrics['auc_pr']:.4f})")

    # ── Save champion to GCS ──────────────────────────────────────────────────
    gcs_uri = save_model_to_gcs(champion_model, champion_name, config)

    output = {
        "champion": champion_name,
        "variant_c": {**metrics_c},
        "variant_d": {**metrics_d},
        "champion_gcs_uri": gcs_uri,
    }
    out_path = Path(config["model"]["lightgbm"]["evaluation_output"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    logger.info("lightgbm_results_saved", extra={"path": str(out_path)})

    print(f"Champion saved to: {gcs_uri}")
    return champion_name, champion_model, champion_metrics, gcs_uri


if __name__ == "__main__":
    main()
