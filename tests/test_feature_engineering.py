"""
Tests for src/features/engineer.py.

Coverage:
  - No-leakage: D-column medians and target encoding use only training data
  - Missing indicators: correct columns created, correct values
  - Target encoding in [0, 1] range
  - Output feature count matches expected
"""

import numpy as np
import pandas as pd
import pytest

from src.features.engineer import FraudFeatureEngineer


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def small_df() -> pd.DataFrame:
    """
    20-row DataFrame with all relevant feature types.
    V1: 50% nulls (> 5% threshold → indicator created)
    V2:  0% nulls (≤ 5% threshold → no indicator)
    """
    rng = np.random.default_rng(42)
    n = 20

    v1 = rng.uniform(0, 1, n)
    v1[[0, 2, 4, 6, 8, 10, 12, 14, 16, 18]] = np.nan  # 50% nulls

    return pd.DataFrame({
        "TransactionDT": rng.integers(86400, 86400 * 500, n),
        "TransactionAmt": rng.uniform(1.0, 1000.0, n),
        "card1": rng.choice([100, 200, 300], n),
        "card4": rng.choice(["visa", "mastercard", "amex"], n),
        "card6": rng.choice(["debit", "credit"], n),
        "P_emaildomain": rng.choice(["gmail.com", "yahoo.com", "hotmail.com"], n),
        "R_emaildomain": rng.choice(["gmail.com", "yahoo.com"], n),
        "D1": rng.uniform(0.0, 300.0, n),
        "D4": rng.uniform(0.0, 300.0, n),
        "D10": rng.uniform(0.0, 300.0, n),
        "V1": v1,
        "V2": rng.uniform(0.0, 1.0, n),
    })


@pytest.fixture
def small_y(small_df: pd.DataFrame) -> pd.Series:
    rng = np.random.default_rng(0)
    return pd.Series(
        rng.choice([0, 1], len(small_df), p=[0.8, 0.2]), name="isFraud"
    )


@pytest.fixture
def engineer() -> FraudFeatureEngineer:
    return FraudFeatureEngineer(
        missing_threshold=0.05,
        d_norm_cols=["D1", "D4", "D10"],
        d_group_col="card1",
        target_encode_cols=["card4", "card6", "P_emaildomain", "R_emaildomain"],
    )


# ─── No-leakage tests ─────────────────────────────────────────────────────────

class TestNoLeakage:
    def test_d_col_medians_fit_from_train_only(
        self, small_df: pd.DataFrame, small_y: pd.Series, engineer: FraudFeatureEngineer
    ):
        """Stored D-column medians must match per-group medians of the training half only."""
        train_X = small_df.iloc[:10].reset_index(drop=True)
        train_y = small_y.iloc[:10].reset_index(drop=True)

        engineer.fit(train_X, train_y)

        expected = train_X.groupby("card1")["D1"].median()
        pd.testing.assert_series_equal(
            engineer._d_group_medians["D1"].sort_index(),
            expected.sort_index(),
            check_names=False,
        )

    def test_transform_test_split_does_not_use_test_labels(
        self, small_df: pd.DataFrame, small_y: pd.Series, engineer: FraudFeatureEngineer
    ):
        """Transforming test data (y=None) must rely solely on training stats."""
        train_X = small_df.iloc[:10].reset_index(drop=True)
        test_X = small_df.iloc[10:].reset_index(drop=True)
        train_y = small_y.iloc[:10].reset_index(drop=True)

        engineer.fit(train_X, train_y)
        # Passing no y means test data cannot influence the encoding
        result = engineer.transform(test_X)

        assert "card4_te" in result.columns
        # Values must be in valid [0, 1] range — any leakage would produce
        # unstable or out-of-range values
        assert result["card4_te"].between(0, 1).all()

    def test_loo_excludes_current_row_label(self):
        """
        Minimal 4-row dataset:
          card4  y
          visa   1   →  LOO = (sum_visa - 1) / (count_visa - 1) = (2-1)/(2-1) = 1.0
          visa   1   →  LOO = (2-1)/(2-1) = 1.0
          other  0   →  LOO = (0-0)/(2-1) = 0.0
          other  0   →  LOO = (0-0)/(2-1) = 0.0
        """
        X = pd.DataFrame({
            "TransactionDT": [86400, 86401, 86402, 86403],
            "TransactionAmt": [100.0, 200.0, 150.0, 120.0],
            "card1": [1, 1, 2, 2],
            "card4": ["visa", "visa", "other", "other"],
            "card6": ["debit", "debit", "credit", "credit"],
            "P_emaildomain": ["gmail.com"] * 4,
            "R_emaildomain": ["yahoo.com"] * 4,
            "D1": [10.0, 20.0, 15.0, 25.0],
            "D4": [5.0, 10.0, 8.0, 12.0],
            "D10": [3.0, 6.0, 4.0, 8.0],
        })
        y = pd.Series([1, 1, 0, 0])

        eng = FraudFeatureEngineer(
            missing_threshold=0.05,
            d_norm_cols=["D1", "D4", "D10"],
            d_group_col="card1",
            target_encode_cols=["card4", "card6", "P_emaildomain", "R_emaildomain"],
        )
        result = eng.fit_transform(X, y)

        assert result.loc[0, "card4_te"] == pytest.approx(1.0)
        assert result.loc[1, "card4_te"] == pytest.approx(1.0)
        assert result.loc[2, "card4_te"] == pytest.approx(0.0)
        assert result.loc[3, "card4_te"] == pytest.approx(0.0)

    def test_unseen_category_falls_back_to_global_mean(
        self, small_df: pd.DataFrame, small_y: pd.Series, engineer: FraudFeatureEngineer
    ):
        """A category not seen during fit must receive the global mean, not 0 or NaN."""
        train_X = small_df.iloc[:10].reset_index(drop=True)
        train_y = small_y.iloc[:10].reset_index(drop=True)
        engineer.fit(train_X, train_y)

        test_X = small_df.iloc[10:15].copy().reset_index(drop=True)
        test_X["card4"] = "__unseen_card__"

        result = engineer.transform(test_X)
        expected = engineer._global_mean

        np.testing.assert_allclose(result["card4_te"].values, expected, rtol=1e-6)


# ─── Missing indicator tests ──────────────────────────────────────────────────

class TestMissingIndicators:
    def test_indicator_created_for_high_null_column(
        self, small_df: pd.DataFrame, small_y: pd.Series, engineer: FraudFeatureEngineer
    ):
        result = engineer.fit_transform(small_df, small_y)
        # V1 has 50% nulls → above 5% threshold
        assert "V1_was_missing" in result.columns

    def test_no_indicator_for_low_null_column(
        self, small_df: pd.DataFrame, small_y: pd.Series, engineer: FraudFeatureEngineer
    ):
        result = engineer.fit_transform(small_df, small_y)
        # V2 has 0% nulls → below 5% threshold
        assert "V2_was_missing" not in result.columns

    def test_indicator_values_match_original_nulls(
        self, small_df: pd.DataFrame, small_y: pd.Series, engineer: FraudFeatureEngineer
    ):
        was_null = small_df["V1"].isnull().astype(int).values
        result = engineer.fit_transform(small_df, small_y)

        np.testing.assert_array_equal(result["V1_was_missing"].values, was_null)

    def test_v_columns_fully_imputed_after_transform(
        self, small_df: pd.DataFrame, small_y: pd.Series, engineer: FraudFeatureEngineer
    ):
        result = engineer.fit_transform(small_df, small_y)
        assert result["V1"].isnull().sum() == 0
        assert result["V2"].isnull().sum() == 0

    def test_imputation_uses_training_median_only(self):
        """V-column median is taken from training; test-set values don't shift it."""
        X_train = pd.DataFrame({
            "TransactionDT": [86400, 86401, 86402],
            "TransactionAmt": [100.0, 200.0, 150.0],
            "card1": [1, 1, 2],
            "card4": ["visa", "visa", "mc"],
            "card6": ["debit", "credit", "debit"],
            "P_emaildomain": ["gmail.com"] * 3,
            "R_emaildomain": ["yahoo.com"] * 3,
            "D1": [10.0, 20.0, 15.0],
            "D4": [5.0, 10.0, 8.0],
            "D10": [3.0, 6.0, 4.0],
            "V1": [np.nan, 2.0, np.nan],  # train median = 2.0
        })
        y_train = pd.Series([0, 1, 0])

        X_test = pd.DataFrame({
            "TransactionDT": [86404],
            "TransactionAmt": [300.0],
            "card1": [1],
            "card4": ["visa"],
            "card6": ["debit"],
            "P_emaildomain": ["gmail.com"],
            "R_emaildomain": ["yahoo.com"],
            "D1": [12.0],
            "D4": [6.0],
            "D10": [3.5],
            "V1": [np.nan],
        })

        eng = FraudFeatureEngineer(
            missing_threshold=0.05,
            d_norm_cols=["D1", "D4", "D10"],
            d_group_col="card1",
            target_encode_cols=["card4", "card6", "P_emaildomain", "R_emaildomain"],
        )
        eng.fit(X_train, y_train)

        # Training median of V1 is 2.0 (only non-null value)
        assert eng._v_medians["V1"] == pytest.approx(2.0)

        result = eng.transform(X_test)
        assert result["V1"].iloc[0] == pytest.approx(2.0)


# ─── Target encoding range tests ─────────────────────────────────────────────

class TestTargetEncodingRange:
    def test_training_encoded_values_in_0_1_range(
        self, small_df: pd.DataFrame, small_y: pd.Series, engineer: FraudFeatureEngineer
    ):
        result = engineer.fit_transform(small_df, small_y)
        for col in ["card4_te", "card6_te", "P_emaildomain_te", "R_emaildomain_te"]:
            assert result[col].between(0, 1).all(), f"{col} has values outside [0, 1]"

    def test_test_encoded_values_in_0_1_range(
        self, small_df: pd.DataFrame, small_y: pd.Series, engineer: FraudFeatureEngineer
    ):
        train_X = small_df.iloc[:10].reset_index(drop=True)
        test_X = small_df.iloc[10:].reset_index(drop=True)
        train_y = small_y.iloc[:10].reset_index(drop=True)

        engineer.fit(train_X, train_y)
        result = engineer.transform(test_X)

        for col in ["card4_te", "card6_te", "P_emaildomain_te", "R_emaildomain_te"]:
            assert result[col].between(0, 1).all(), f"{col} has values outside [0, 1]"

    def test_encoded_columns_not_nan(
        self, small_df: pd.DataFrame, small_y: pd.Series, engineer: FraudFeatureEngineer
    ):
        result = engineer.fit_transform(small_df, small_y)
        for col in ["card4_te", "card6_te", "P_emaildomain_te", "R_emaildomain_te"]:
            assert result[col].isnull().sum() == 0, f"{col} contains NaN"


# ─── Feature count tests ──────────────────────────────────────────────────────

class TestOutputFeatureCount:
    """
    Input  (small_df): 12 columns
      TransactionDT, TransactionAmt, card1,
      card4, card6, P_emaildomain, R_emaildomain,
      D1, D4, D10, V1, V2

    Additions:
      +3  time features  (time_of_day, day_of_week, hour_of_day)
      +3  velocity       (log_transaction_amt, amt_to_d1_ratio, is_round_amt)
      +3  D-norm         (D1_card_norm, D4_card_norm, D10_card_norm)
      +1  V1_was_missing (V1 has 50% nulls → above threshold)

    No net change:
       4 raw cat cols dropped, 4 _te cols added → net 0

    Expected total: 12 + 3 + 3 + 3 + 1 = 22
    """

    def test_total_output_columns(
        self, small_df: pd.DataFrame, small_y: pd.Series, engineer: FraudFeatureEngineer
    ):
        result = engineer.fit_transform(small_df, small_y)
        assert result.shape[1] == 22, (
            f"Expected 22 columns, got {result.shape[1]}: {sorted(result.columns)}"
        )

    def test_time_features_present(
        self, small_df: pd.DataFrame, small_y: pd.Series, engineer: FraudFeatureEngineer
    ):
        result = engineer.fit_transform(small_df, small_y)
        for col in ["time_of_day", "day_of_week", "hour_of_day"]:
            assert col in result.columns

    def test_velocity_features_present(
        self, small_df: pd.DataFrame, small_y: pd.Series, engineer: FraudFeatureEngineer
    ):
        result = engineer.fit_transform(small_df, small_y)
        for col in ["log_transaction_amt", "amt_to_d1_ratio", "is_round_amt"]:
            assert col in result.columns

    def test_d_norm_features_present(
        self, small_df: pd.DataFrame, small_y: pd.Series, engineer: FraudFeatureEngineer
    ):
        result = engineer.fit_transform(small_df, small_y)
        for col in ["D1_card_norm", "D4_card_norm", "D10_card_norm"]:
            assert col in result.columns

    def test_raw_categorical_columns_removed(
        self, small_df: pd.DataFrame, small_y: pd.Series, engineer: FraudFeatureEngineer
    ):
        result = engineer.fit_transform(small_df, small_y)
        for col in ["card4", "card6", "P_emaildomain", "R_emaildomain"]:
            assert col not in result.columns

    def test_target_encoded_columns_present(
        self, small_df: pd.DataFrame, small_y: pd.Series, engineer: FraudFeatureEngineer
    ):
        result = engineer.fit_transform(small_df, small_y)
        for col in ["card4_te", "card6_te", "P_emaildomain_te", "R_emaildomain_te"]:
            assert col in result.columns

    def test_row_count_unchanged(
        self, small_df: pd.DataFrame, small_y: pd.Series, engineer: FraudFeatureEngineer
    ):
        result = engineer.fit_transform(small_df, small_y)
        assert len(result) == len(small_df)

    def test_transform_not_fitted_raises(self, engineer: FraudFeatureEngineer):
        with pytest.raises(RuntimeError, match="fit()"):
            engineer.transform(pd.DataFrame({"x": [1, 2]}))
