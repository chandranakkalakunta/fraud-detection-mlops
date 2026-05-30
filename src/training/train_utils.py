"""
Shared training utilities — v6, Vizier HP search, v7 champion pipeline.

Extracted from run_v5_training.py pattern to avoid duplication across
convergence baseline, HP tuning, and final champion runs.
"""

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.features.engineer import FraudFeatureEngineer
from src.features.imbalance import ClassWeightHandler
from src.training.xgboost_trainer import SORT_COL, TARGET, stratified_val_carve
from src.utils.logging import get_logger

logger = get_logger(__name__)


def build_no_te_engineer(config: dict) -> FraudFeatureEngineer:
    """FraudFeatureEngineer with target_encode_cols=[] (v5+ standard)."""
    return FraudFeatureEngineer(
        target_encode_cols=[],
        d_norm_cols=config["features"]["d_normalize_cols"],
        d_group_col=config["features"]["d_normalize_group_col"],
    )


def prepare_data_splits(
    df: pd.DataFrame,
    config: dict,
    train_frac: float = 0.75,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Time-based test split + stratified val carve from the training pool.

    Test set = temporal last (1 - train_frac) fraction — never shuffled.
    Val set = stratified carve from training pool, guaranteeing 3.5% fraud rate.

    Returns X_train, y_train, X_val, y_val, X_test, y_test.
    SORT_COL is kept in X so the engineer can create time-of-day features.
    """
    val_frac = float(config["model"]["val_split"])
    random_state = int(config["model"]["random_state"])

    df_sorted = df.sort_values(SORT_COL).reset_index(drop=True)
    n = len(df_sorted)
    train_end = int(n * train_frac)
    train_pool = df_sorted.iloc[:train_end]
    test_df = df_sorted.iloc[train_end:].reset_index(drop=True)

    train_df, val_df = stratified_val_carve(train_pool, val_frac=val_frac, random_state=random_state)

    feature_cols = [c for c in df.columns if c != TARGET]

    X_train = train_df[feature_cols].reset_index(drop=True)
    y_train = train_df[TARGET].astype(int).reset_index(drop=True)
    X_val = val_df[feature_cols].reset_index(drop=True)
    y_val = val_df[TARGET].astype(int).reset_index(drop=True)
    X_test = test_df[feature_cols].reset_index(drop=True)
    y_test = test_df[TARGET].astype(int).reset_index(drop=True)

    logger.info(
        "data_splits_prepared",
        extra={
            "train_rows": len(X_train),
            "val_rows": len(X_val),
            "test_rows": len(X_test),
            "train_fraud_rate": round(float(y_train.mean()), 4),
            "val_fraud_rate": round(float(y_val.mean()), 4),
            "test_fraud_rate": round(float(y_test.mean()), 4),
        },
    )
    return X_train, y_train, X_val, y_val, X_test, y_test


def apply_engineer(
    engineer: FraudFeatureEngineer,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Fit engineer on training data and transform all three splits.

    Returns numeric-only DataFrames (select_dtypes strips M/ProductCD columns).
    Asserts zero NaN in all outputs — training should fail loudly on data issues.
    """
    X_train_eng = engineer.fit_transform(X_train, y_train).select_dtypes(include=[np.number])
    X_val_eng = engineer.transform(X_val).select_dtypes(include=[np.number])
    X_test_eng = engineer.transform(X_test).select_dtypes(include=[np.number])

    assert X_train_eng.isnull().sum().sum() == 0, "NaN in engineered train features"
    assert X_val_eng.isnull().sum().sum() == 0, "NaN in engineered val features"
    assert X_test_eng.isnull().sum().sum() == 0, "NaN in engineered test features"

    logger.info(
        "engineer_applied",
        extra={"n_features": X_train_eng.shape[1], "train_rows": len(X_train_eng)},
    )
    return X_train_eng, X_val_eng, X_test_eng


def train_xgb_from_params(
    X_train_eng: pd.DataFrame,
    y_train: pd.Series,
    X_val_eng: pd.DataFrame,
    y_val: pd.Series,
    params: dict,
    config: dict,
) -> xgb.XGBClassifier:
    """
    Train XGBoost with an explicit params dict.

    Recognised params keys (all fall back to config if absent):
      n_estimators, max_depth, learning_rate, min_child_weight,
      subsample, colsample_bytree, scale_pos_weight, early_stopping_rounds.

    eval_metric is always "auc" — logloss + scale_pos_weight is incompatible
    (best_iter=0; confirmed empirically in v4).
    """
    xgb_cfg = config["model"]["xgboost"]
    cw = ClassWeightHandler()
    cw.fit(y_train)

    model = xgb.XGBClassifier(
        n_estimators=int(params.get("n_estimators", xgb_cfg["n_estimators"])),
        max_depth=int(params.get("max_depth", xgb_cfg["max_depth"])),
        learning_rate=float(params.get("learning_rate", xgb_cfg["learning_rate"])),
        tree_method=xgb_cfg["tree_method"],
        early_stopping_rounds=int(params.get("early_stopping_rounds", xgb_cfg["early_stopping_rounds"])),
        eval_metric="auc",
        scale_pos_weight=float(params.get("scale_pos_weight", cw.scale_pos_weight)),
        subsample=float(params.get("subsample", xgb_cfg["subsample"])),
        colsample_bytree=float(params.get("colsample_bytree", xgb_cfg["colsample_bytree"])),
        min_child_weight=int(params.get("min_child_weight", xgb_cfg["min_child_weight"])),
        random_state=int(config["model"]["random_state"]),
        n_jobs=-1,
        verbosity=0,
    )

    model.fit(X_train_eng, y_train, eval_set=[(X_val_eng, y_val)], verbose=False)

    logger.info(
        "xgb_trained",
        extra={
            "best_iteration": model.best_iteration,
            "n_features": X_train_eng.shape[1],
            "n_estimators_requested": int(params.get("n_estimators", xgb_cfg["n_estimators"])),
        },
    )
    return model


def train_lgb_from_params(
    X_train_eng: pd.DataFrame,
    y_train: pd.Series,
    X_val_eng: pd.DataFrame,
    y_val: pd.Series,
    params: dict,
    config: dict,
) -> lgb.LGBMClassifier:
    """
    Train LightGBM with an explicit params dict.

    Recognised params keys (fall back to config if absent):
      n_estimators, num_leaves, learning_rate, min_child_samples,
      subsample (→ bagging_fraction), colsample_bytree (→ feature_fraction),
      reg_alpha, reg_lambda, early_stopping_rounds.

    metric is always "auc" — binary_logloss + is_unbalance is incompatible
    (best_iter=1; same root cause as XGBoost logloss issue).
    """
    lgb_cfg = config["model"]["lightgbm"]
    early_stopping_rounds = int(params.get("early_stopping_rounds", lgb_cfg["early_stopping_rounds"]))
    callbacks = [lgb.early_stopping(early_stopping_rounds, verbose=False)]

    model = lgb.LGBMClassifier(
        num_leaves=int(params.get("num_leaves", lgb_cfg["num_leaves"])),
        learning_rate=float(params.get("learning_rate", lgb_cfg["learning_rate"])),
        n_estimators=int(params.get("n_estimators", lgb_cfg["n_estimators"])),
        min_child_samples=int(params.get("min_child_samples", 100)),
        feature_fraction=float(params.get("colsample_bytree", lgb_cfg["feature_fraction"])),
        bagging_fraction=float(params.get("subsample", lgb_cfg["bagging_fraction"])),
        bagging_freq=int(lgb_cfg["bagging_freq"]),
        reg_alpha=float(params.get("reg_alpha", 0.0)),
        reg_lambda=float(params.get("reg_lambda", 0.0)),
        is_unbalance=True,
        metric="auc",
        random_state=int(config["model"]["random_state"]),
        n_jobs=-1,
        verbose=-1,
    )

    model.fit(X_train_eng, y_train, eval_set=[(X_val_eng, y_val)], callbacks=callbacks)

    logger.info(
        "lgb_trained",
        extra={
            "best_iteration": model.best_iteration_,
            "n_features": X_train_eng.shape[1],
            "n_estimators_requested": int(params.get("n_estimators", lgb_cfg["n_estimators"])),
        },
    )
    return model
