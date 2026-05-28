"""
Baseline model: Logistic Regression on IEEE-CIS Fraud Detection dataset.

Primary metric: Average Precision (AUC-PR) — NOT accuracy.
Rationale: the dataset has ~3.5% fraud rate (severe class imbalance).
Accuracy would be misleading — a model predicting all-legit gets ~96.5%.
AUC-PR penalises false negatives and rewards high precision at every recall
level, which is exactly what a fraud detection system requires.

This module establishes the baseline every subsequent model must beat.
Results are written to evaluation/baseline_results.json.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from google.cloud import bigquery
from src.utils.config import load_config
from src.utils.logging import get_logger

logger = get_logger(__name__)

# ─── Feature groups ───────────────────────────────────────────────────────────

NUMERIC_FEATURES = (
    ["TransactionAmt", "card1", "card2", "card3", "card5",
     "addr1", "addr2", "dist1", "dist2"]
    + [f"C{i}" for i in range(1, 15)]
    + [f"D{i}" for i in range(1, 16)]
)

CATEGORICAL_FEATURES = [
    "ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain",
    "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9",
]

TARGET = "isFraud"


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_data_from_bigquery(config: dict) -> tuple[Any, Any]:
    """Load feature matrix and target from BigQuery joined table."""
    project = config["gcp"]["project_id"]
    dataset = config["bigquery"]["dataset"]
    table = config["bigquery"]["tables"]["transactions_joined"]

    feature_cols = ", ".join(NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET])
    query = f"""
        SELECT {feature_cols}
        FROM `{project}.{dataset}.{table}`
        WHERE {TARGET} IS NOT NULL
    """

    logger.info("loading_data_from_bq", extra={"table": f"{dataset}.{table}"})
    client = bigquery.Client(project=project)
    df = client.query(query).to_dataframe(progress_bar_type=None)

    logger.info(
        "data_loaded",
        extra={
            "rows": len(df),
            "fraud_count": int(df[TARGET].sum()),
            "fraud_rate_pct": round(df[TARGET].mean() * 100, 4),
        },
    )

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET].astype(int)
    return X, y


# ─── Pipeline ────────────────────────────────────────────────────────────────

def build_pipeline(config: dict) -> Pipeline:
    """Build preprocessing + Logistic Regression pipeline."""
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_transformer, NUMERIC_FEATURES),
        ("categorical", categorical_transformer, CATEGORICAL_FEATURES),
    ])

    model = LogisticRegression(
        max_iter=int(config["model"]["max_iter"]),
        random_state=int(config["model"]["random_state"]),
        class_weight="balanced",   # addresses class imbalance
        solver="saga",             # scales to large datasets
        n_jobs=-1,
    )

    return Pipeline([("preprocessor", preprocessor), ("classifier", model)])


# ─── Evaluation ───────────────────────────────────────────────────────────────

def _find_optimal_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Return threshold that maximises F1 on the precision-recall curve."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    # Avoid division by zero
    f1_scores = np.where(
        (precision + recall) > 0,
        2 * (precision * recall) / (precision + recall),
        0,
    )
    return float(thresholds[np.argmax(f1_scores[:-1])])


def evaluate(
    pipeline: Pipeline,
    X_train: Any,
    X_test: Any,
    y_train: np.ndarray,
    y_test: np.ndarray,
    cv_folds: int,
    random_state: int,
) -> dict:
    """Fit pipeline and compute all evaluation metrics."""
    logger.info("fitting_baseline_pipeline")
    pipeline.fit(X_train, y_train)

    y_proba = pipeline.predict_proba(X_test)[:, 1]
    threshold = _find_optimal_threshold(y_test.values, y_proba)
    y_pred = (y_proba >= threshold).astype(int)

    auc_pr = average_precision_score(y_test, y_proba)
    auc_roc = roc_auc_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    # Cross-validated AUC-PR on training set for variance estimate
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    cv_auc_pr = cross_val_score(
        pipeline, X_train, y_train,
        cv=cv, scoring="average_precision", n_jobs=-1,
    )

    report = classification_report(y_test, y_pred, output_dict=True)

    metrics = {
        "model": "LogisticRegression",
        "primary_metric": "average_precision",
        "primary_metric_value": round(auc_pr, 6),
        "auc_roc": round(auc_roc, 6),
        "f1_at_optimal_threshold": round(f1, 6),
        "precision_at_optimal_threshold": round(precision, 6),
        "recall_at_optimal_threshold": round(recall, 6),
        "optimal_threshold": round(threshold, 6),
        "cv_auc_pr_mean": round(float(cv_auc_pr.mean()), 6),
        "cv_auc_pr_std": round(float(cv_auc_pr.std()), 6),
        "cv_folds": cv_folds,
        "classification_report": report,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "features_numeric": NUMERIC_FEATURES,
        "features_categorical": CATEGORICAL_FEATURES,
        "note": (
            "AUC-PR is the primary metric because the dataset has ~3.5% fraud rate. "
            "AUC-PR is insensitive to the large number of true negatives and directly "
            "measures performance on the minority (fraud) class. Every subsequent model "
            "must exceed this AUC-PR value."
        ),
    }

    logger.info(
        "baseline_evaluation_complete",
        extra={
            "auc_pr": auc_pr,
            "auc_roc": auc_roc,
            "f1": f1,
            "threshold": threshold,
            "cv_auc_pr_mean": metrics["cv_auc_pr_mean"],
        },
    )
    return metrics


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    from sklearn.model_selection import train_test_split

    env = os.getenv("ENV", "dev")
    config = load_config(env)

    random_state = int(config["model"]["random_state"])
    test_size = float(config["model"]["test_size"])
    cv_folds = int(config["model"]["cv_folds"])

    X, y = load_data_from_bigquery(config)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    logger.info(
        "train_test_split",
        extra={
            "train": len(X_train),
            "test": len(X_test),
            "train_fraud_rate": round(float(y_train.mean()), 4),
            "test_fraud_rate": round(float(y_test.mean()), 4),
        },
    )

    pipeline = build_pipeline(config)
    metrics = evaluate(pipeline, X_train, X_test, y_train, y_test, cv_folds, random_state)

    output_path = Path(config["model"]["evaluation_output"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2))
    logger.info("results_saved", extra={"path": str(output_path)})

    print(f"\nBaseline AUC-PR : {metrics['primary_metric_value']:.4f}")
    print(f"AUC-ROC         : {metrics['auc_roc']:.4f}")
    print(f"F1 (threshold={metrics['optimal_threshold']:.3f}) : {metrics['f1_at_optimal_threshold']:.4f}")
    print(f"CV AUC-PR       : {metrics['cv_auc_pr_mean']:.4f} ± {metrics['cv_auc_pr_std']:.4f}")
    print(f"\nResults written to {output_path}")


if __name__ == "__main__":
    main()
