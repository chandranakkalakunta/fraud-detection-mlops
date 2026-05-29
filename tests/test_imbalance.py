"""
Tests for src/features/imbalance.py.

Covers:
  - SMOTE never touches test/validation data (assertion enforced)
  - SMOTE increases minority class representation
  - ClassWeightHandler computes correct ratio for realistic fraud rate
  - ClassWeightHandler raises before fit is called
"""

import numpy as np
import pandas as pd
import pytest

from src.features.imbalance import ClassWeightHandler, SMOTEHandler


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def imbalanced_train():
    """500 rows, ~3.5% fraud — matches real dataset rate."""
    rng = np.random.default_rng(42)
    n = 500
    y = pd.Series(rng.choice([0, 1], n, p=[0.965, 0.035]), name="isFraud")
    # 10 numeric features — SMOTE works in feature space
    X = pd.DataFrame(
        rng.standard_normal((n, 10)),
        columns=[f"f{i}" for i in range(10)],
    )
    return X, y


@pytest.fixture
def exact_imbalanced():
    """Exact 3.5% fraud rate for deterministic weight assertions."""
    n_total = 1000
    n_fraud = 35   # 3.5%
    n_legit = 965  # 96.5%
    y = pd.Series([1] * n_fraud + [0] * n_legit, name="isFraud")
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.standard_normal((n_total, 5)), columns=[f"f{i}" for i in range(5)])
    return X, y


# ─── SMOTEHandler ─────────────────────────────────────────────────────────────

class TestSMOTEHandler:
    def test_smote_requires_train_only_flag(self, imbalanced_train):
        """Calling fit_resample without train_only=True must raise AssertionError."""
        X, y = imbalanced_train
        handler = SMOTEHandler()
        with pytest.raises(AssertionError, match="training data"):
            handler.fit_resample(X, y, train_only=False)

    def test_smote_never_touches_test_set(self, imbalanced_train):
        """Passing test data without train_only=True raises — SMOTE cannot run on test."""
        X, y = imbalanced_train
        handler = SMOTEHandler()
        X_test = X.iloc[:50].reset_index(drop=True)
        y_test = y.iloc[:50].reset_index(drop=True)
        with pytest.raises(AssertionError):
            handler.fit_resample(X_test, y_test, train_only=False)

    def test_smote_increases_minority_class(self, imbalanced_train):
        """After SMOTE, fraud rate must be higher than in the original training data."""
        X, y = imbalanced_train
        handler = SMOTEHandler(sampling_strategy=0.1, random_state=42)
        X_res, y_res = handler.fit_resample(X, y, train_only=True)
        assert float(y_res.mean()) > float(y.mean())

    def test_smote_output_has_no_nan(self, imbalanced_train):
        """SMOTE output must contain no NaN values."""
        X, y = imbalanced_train
        handler = SMOTEHandler(random_state=42)
        X_res, y_res = handler.fit_resample(X, y, train_only=True)
        assert X_res.isnull().sum().sum() == 0
        assert y_res.isnull().sum() == 0

    def test_smote_preserves_column_names(self, imbalanced_train):
        """Column names must be preserved exactly after resampling."""
        X, y = imbalanced_train
        handler = SMOTEHandler(random_state=42)
        X_res, _ = handler.fit_resample(X, y, train_only=True)
        assert list(X_res.columns) == list(X.columns)

    def test_smote_output_larger_than_input(self, imbalanced_train):
        """Resampled dataset must be larger than the original training set."""
        X, y = imbalanced_train
        handler = SMOTEHandler(sampling_strategy=0.1, random_state=42)
        X_res, _ = handler.fit_resample(X, y, train_only=True)
        assert len(X_res) > len(X)

    def test_get_params_returns_strategy_key(self, imbalanced_train):
        """get_params must include imbalance_strategy='smote' for Experiments logging."""
        handler = SMOTEHandler(sampling_strategy=0.2, k_neighbors=3)
        params = handler.get_params()
        assert params["imbalance_strategy"] == "smote"
        assert params["smote_sampling_strategy"] == 0.2
        assert params["smote_k_neighbors"] == 3


# ─── ClassWeightHandler ───────────────────────────────────────────────────────

class TestClassWeightHandler:
    def test_scale_pos_weight_correct_for_3p5_fraud_rate(self, exact_imbalanced):
        """
        With 965 legit and 35 fraud, scale_pos_weight = 965/35 ≈ 27.57.
        This is the exact ratio XGBoost needs to compensate for imbalance.
        """
        _, y = exact_imbalanced
        handler = ClassWeightHandler()
        handler.fit(y)
        expected = 965 / 35
        assert handler.scale_pos_weight == pytest.approx(expected, rel=1e-4)

    def test_class_weight_dict_structure(self, exact_imbalanced):
        """class_weight must be {0: 1.0, 1: scale_pos_weight}."""
        _, y = exact_imbalanced
        handler = ClassWeightHandler()
        handler.fit(y)
        cw = handler.class_weight
        assert cw[0] == pytest.approx(1.0)
        assert cw[1] == pytest.approx(handler.scale_pos_weight)

    def test_raises_before_fit(self):
        """Accessing scale_pos_weight or class_weight before fit must raise RuntimeError."""
        handler = ClassWeightHandler()
        with pytest.raises(RuntimeError, match="fit()"):
            _ = handler.scale_pos_weight
        with pytest.raises(RuntimeError, match="fit()"):
            _ = handler.class_weight

    def test_raises_with_no_positive_examples(self):
        """Fit must raise ValueError if there are no fraud examples."""
        y_all_legit = pd.Series([0, 0, 0, 0, 0])
        handler = ClassWeightHandler()
        with pytest.raises(ValueError, match="no positive"):
            handler.fit(y_all_legit)

    def test_get_params_includes_strategy_key(self, exact_imbalanced):
        """get_params must include imbalance_strategy='class_weight'."""
        _, y = exact_imbalanced
        handler = ClassWeightHandler()
        handler.fit(y)
        params = handler.get_params()
        assert params["imbalance_strategy"] == "class_weight"
        assert "scale_pos_weight" in params
        assert "fraud_rate" in params

    def test_weights_match_before_and_after_get_params(self, exact_imbalanced):
        """scale_pos_weight in get_params() must equal the property value."""
        _, y = exact_imbalanced
        handler = ClassWeightHandler()
        handler.fit(y)
        params = handler.get_params()
        assert params["scale_pos_weight"] == pytest.approx(handler.scale_pos_weight, rel=1e-3)
