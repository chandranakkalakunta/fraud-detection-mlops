"""
Shared metric computation for fraud detection model evaluation.

All trainers use this module to ensure identical metric names across
Vertex AI Experiments runs — apples-to-apples comparison is only possible
when the same computation produces each metric value.
"""

from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import (
    average_precision_score,
    fbeta_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def find_optimal_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Return the decision threshold that maximises F1 on the precision-recall curve."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    f1_scores = np.where(
        (precision + recall) > 0,
        2 * precision * recall / (precision + recall),
        0.0,
    )
    best_idx = int(np.argmax(f1_scores[:-1]))
    return float(thresholds[best_idx])


def compute_fraud_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    imbalance_strategy: str,
    training_rows: int,
    test_rows: int,
    threshold: Optional[float] = None,
) -> dict:
    """
    Compute the full evaluation metric set used across all Vertex AI Experiment runs.

    Metric names are canonical — must not be changed between runs so that
    Experiments comparison tables remain apples-to-apples.

    Returns a flat dict suitable for direct logging via run.log_metrics().
    """
    if threshold is None:
        threshold = find_optimal_threshold(y_true, y_proba)

    y_pred = (y_proba >= threshold).astype(int)

    auc_pr = float(average_precision_score(y_true, y_proba))
    auc_roc = float(roc_auc_score(y_true, y_proba))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    f2 = float(fbeta_score(y_true, y_pred, beta=2, zero_division=0))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))

    # KS statistic: max separation between fraud and legitimate score distributions
    ks_stat = float(
        ks_2samp(
            y_proba[y_true == 0],
            y_proba[y_true == 1],
        ).statistic
    )

    return {
        "auc_pr":            round(auc_pr, 6),
        "auc_roc":           round(auc_roc, 6),
        "f1":                round(f1, 6),
        "f2":                round(f2, 6),
        "precision":         round(precision, 6),
        "recall":            round(recall, 6),
        "ks_statistic":      round(ks_stat, 6),
        "threshold":         round(threshold, 6),
        "training_rows":     training_rows,
        "test_rows":         test_rows,
        "imbalance_strategy": imbalance_strategy,
    }
