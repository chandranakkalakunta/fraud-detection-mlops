"""
Imbalance handling strategies for fraud detection training.

Two strategies implemented:
  - SMOTEHandler: over-samples the minority (fraud) class in training data only.
  - ClassWeightHandler: computes cost-sensitive weights instead of resampling.

Both implement get_params() returning a dict suitable for Vertex AI Experiments logging.
"""

from typing import Optional

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE

from src.utils.logging import get_logger

logger = get_logger(__name__)


class SMOTEHandler:
    """
    Apply SMOTE to training data only — never to validation or test sets.

    The assertion in fit_resample() enforces this at the call site: callers
    must explicitly confirm they are passing training data by using the
    train_only=True flag.
    """

    def __init__(
        self,
        sampling_strategy: float = 0.1,
        random_state: int = 42,
        k_neighbors: int = 5,
    ):
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        self.k_neighbors = k_neighbors

    def fit_resample(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        *,
        train_only: bool,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Apply SMOTE and return resampled (X, y).

        train_only must be True — callers confirm this is training data,
        not a validation or test set.
        """
        assert train_only is True, (
            "SMOTE must only be applied to training data. "
            "Pass train_only=True explicitly at the call site."
        )

        smote = SMOTE(
            sampling_strategy=self.sampling_strategy,
            random_state=self.random_state,
            k_neighbors=self.k_neighbors,
        )
        X_arr, y_arr = smote.fit_resample(X_train.values, y_train.values)

        X_resampled = pd.DataFrame(X_arr, columns=X_train.columns)
        y_resampled = pd.Series(y_arr, name=y_train.name)

        logger.info(
            "smote_applied",
            extra={
                "original_rows": len(X_train),
                "resampled_rows": len(X_resampled),
                "original_fraud_rate": round(float(y_train.mean()), 6),
                "resampled_fraud_rate": round(float(y_resampled.mean()), 6),
                "sampling_strategy": self.sampling_strategy,
            },
        )
        return X_resampled, y_resampled

    def get_params(self) -> dict:
        return {
            "imbalance_strategy": "smote",
            "smote_sampling_strategy": self.sampling_strategy,
            "smote_k_neighbors": self.k_neighbors,
            "smote_random_state": self.random_state,
        }


class ClassWeightHandler:
    """
    Compute class weights from training labels for cost-sensitive learning.

    Provides:
      scale_pos_weight — for XGBoost (ratio negative / positive count)
      class_weight     — dict {0: 1.0, 1: scale_pos_weight} for sklearn / LightGBM
    """

    def __init__(self) -> None:
        self._scale_pos_weight: Optional[float] = None
        self._class_weight: Optional[dict] = None
        self._n_positive: Optional[int] = None
        self._n_negative: Optional[int] = None
        self._fraud_rate: Optional[float] = None

    def fit(self, y_train: pd.Series) -> "ClassWeightHandler":
        """Compute weights from training labels only."""
        n_negative = int((y_train == 0).sum())
        n_positive = int((y_train == 1).sum())

        if n_positive == 0:
            raise ValueError("Training set contains no positive (fraud) examples.")

        self._n_negative = n_negative
        self._n_positive = n_positive
        self._scale_pos_weight = n_negative / n_positive
        self._class_weight = {0: 1.0, 1: self._scale_pos_weight}
        self._fraud_rate = float(y_train.mean())

        logger.info(
            "class_weights_computed",
            extra={
                "n_negative": n_negative,
                "n_positive": n_positive,
                "scale_pos_weight": round(self._scale_pos_weight, 4),
                "fraud_rate": round(self._fraud_rate, 6),
            },
        )
        return self

    @property
    def scale_pos_weight(self) -> float:
        if self._scale_pos_weight is None:
            raise RuntimeError("Call fit() before accessing scale_pos_weight.")
        return self._scale_pos_weight

    @property
    def class_weight(self) -> dict:
        if self._class_weight is None:
            raise RuntimeError("Call fit() before accessing class_weight.")
        return self._class_weight

    def get_params(self) -> dict:
        return {
            "imbalance_strategy": "class_weight",
            "scale_pos_weight": round(self._scale_pos_weight, 4) if self._scale_pos_weight is not None else None,
            "n_positive": self._n_positive,
            "n_negative": self._n_negative,
            "fraud_rate": round(self._fraud_rate, 6) if self._fraud_rate is not None else None,
        }
