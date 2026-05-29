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
    V1: 50% nulls (in 5–80% window → indicator created)
    V2:  0% nulls (below 5% lower bound → no indicator)
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
        missing_threshold=0.80,
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
            missing_threshold=0.80,
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


# ─── V-column imputation tests ────────────────────────────────────────────────
# _was_missing indicators were removed in v4 (see engineer.py module comment).
# Run A (no indicators, AUC-PR=0.4853) beat Run C (253 indicators, AUC-PR=0.1228)
# by 0.28 AUC-PR. All V-columns are now imputed with training medians only.

class TestMissingIndicators:
    def test_no_indicator_columns_created(
        self, small_df: pd.DataFrame, small_y: pd.Series, engineer: FraudFeatureEngineer
    ):
        result = engineer.fit_transform(small_df, small_y)
        was_missing_cols = [c for c in result.columns if c.endswith("_was_missing")]
        assert len(was_missing_cols) == 0, (
            f"Expected no indicator columns, got: {was_missing_cols}"
        )

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
            missing_threshold=0.80,
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

    No net change:
       4 raw cat cols dropped, 4 _te cols added → net 0

    No _was_missing indicator columns — removed in v4 (see engineer.py comment).

    Expected total: 12 + 3 + 3 + 3 = 21
    """

    def test_total_output_columns(
        self, small_df: pd.DataFrame, small_y: pd.Series, engineer: FraudFeatureEngineer
    ):
        result = engineer.fit_transform(small_df, small_y)
        assert result.shape[1] == 21, (
            f"Expected 21 columns, got {result.shape[1]}: {sorted(result.columns)}"
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


# ─── Pass-through imputation tests ───────────────────────────────────────────

class TestPassThroughImputation:
    """
    Verify median imputation for non-V numeric pass-through columns
    (C1–C14, D2–D9, D11–D15, addr1, addr2, dist1, dist2).
    """

    def _make_passthrough_df(self, n: int = 20) -> pd.DataFrame:
        rng = np.random.default_rng(7)
        df = pd.DataFrame({
            "TransactionDT": rng.integers(86400, 86400 * 500, n),
            "TransactionAmt": rng.uniform(1.0, 1000.0, n),
            "card1": rng.choice([100, 200, 300], n),
            "card4": rng.choice(["visa", "mc"], n),
            "card6": rng.choice(["debit", "credit"], n),
            "P_emaildomain": rng.choice(["gmail.com", "yahoo.com"], n),
            "R_emaildomain": rng.choice(["gmail.com", "yahoo.com"], n),
            "D1": rng.uniform(0.0, 100.0, n),
            "D4": rng.uniform(0.0, 100.0, n),
            "D10": rng.uniform(0.0, 100.0, n),
            "C1": rng.uniform(0.0, 10.0, n),
            "C2": rng.uniform(0.0, 10.0, n),
            "D3": rng.uniform(0.0, 50.0, n),
            "D12": rng.uniform(0.0, 50.0, n),
            "addr1": rng.uniform(100.0, 500.0, n),
            "dist1": rng.uniform(0.0, 200.0, n),
        })
        # Inject NaN into pass-through columns
        df.loc[[0, 5, 10, 15], "C1"] = np.nan
        df.loc[[1, 6, 11], "C2"] = np.nan
        df.loc[[2, 7, 12], "D3"] = np.nan
        df.loc[[3, 8, 13], "D12"] = np.nan
        df.loc[[4, 9, 14], "addr1"] = np.nan
        df.loc[[0, 10], "dist1"] = np.nan
        return df

    def test_medians_fit_from_train_only(self):
        """Stored medians must equal median of training slice — not influenced by test."""
        df = self._make_passthrough_df(30)
        y = pd.Series(np.zeros(30, dtype=int))
        train_X = df.iloc[:20].reset_index(drop=True)
        train_y = y.iloc[:20].reset_index(drop=True)

        eng = FraudFeatureEngineer(
            missing_threshold=0.80,
            d_norm_cols=["D1", "D4", "D10"],
            d_group_col="card1",
            target_encode_cols=["card4", "card6", "P_emaildomain", "R_emaildomain"],
        )
        eng.fit(train_X, train_y)

        # Median must match training slice only, ignoring the held-out test rows
        expected_c1 = train_X["C1"].median()
        assert eng._passthrough_medians["C1"] == pytest.approx(expected_c1)

        expected_addr1 = train_X["addr1"].median()
        assert eng._passthrough_medians["addr1"] == pytest.approx(expected_addr1)

    def test_zero_nan_in_passthrough_cols_after_transform(self):
        """After fit_transform, no pass-through column may contain NaN."""
        df = self._make_passthrough_df(20)
        y = pd.Series(np.zeros(20, dtype=int))

        eng = FraudFeatureEngineer(
            missing_threshold=0.80,
            d_norm_cols=["D1", "D4", "D10"],
            d_group_col="card1",
            target_encode_cols=["card4", "card6", "P_emaildomain", "R_emaildomain"],
        )
        result = eng.fit_transform(df, y)

        passthrough_output_cols = [c for c in ["C1", "C2", "D3", "D12", "addr1", "dist1"]
                                   if c in result.columns]
        for col in passthrough_output_cols:
            assert result[col].isnull().sum() == 0, f"{col} still has NaN after transform"

    def test_zero_nan_in_passthrough_cols_on_test_set(self):
        """Test set pass-through NaN imputed using training medians only."""
        df = self._make_passthrough_df(30)
        y = pd.Series(np.zeros(30, dtype=int))
        train_X = df.iloc[:20].reset_index(drop=True)
        train_y = y.iloc[:20].reset_index(drop=True)
        test_X = df.iloc[20:].reset_index(drop=True)

        eng = FraudFeatureEngineer(
            missing_threshold=0.80,
            d_norm_cols=["D1", "D4", "D10"],
            d_group_col="card1",
            target_encode_cols=["card4", "card6", "P_emaildomain", "R_emaildomain"],
        )
        eng.fit(train_X, train_y)
        result = eng.transform(test_X)

        for col in ["C1", "C2", "addr1"]:
            if col in result.columns:
                assert result[col].isnull().sum() == 0, f"{col} has NaN in test transform"

    def test_imputed_values_equal_training_median(self):
        """NaN values must be replaced with the training-set median, not test-set median."""
        rng = np.random.default_rng(99)
        n = 10

        def make_row(c1_val):
            return {
                "TransactionDT": int(rng.integers(86400, 86400 * 10)),
                "TransactionAmt": float(rng.uniform(1, 100)),
                "card1": 1,
                "card4": "visa",
                "card6": "debit",
                "P_emaildomain": "gmail.com",
                "R_emaildomain": "yahoo.com",
                "D1": 10.0, "D4": 5.0, "D10": 3.0,
                "C1": c1_val,
            }

        # Training: C1 values [1, 2, 3, 4, 5] → median = 3.0
        train_rows = [make_row(float(v)) for v in [1, 2, 3, 4, 5]]
        train_X = pd.DataFrame(train_rows)
        train_y = pd.Series([0, 0, 1, 0, 0])

        # Test: C1 is NaN → must be replaced with 3.0 (train median)
        test_X = pd.DataFrame([make_row(np.nan)])

        eng = FraudFeatureEngineer(
            missing_threshold=0.80,
            d_norm_cols=["D1", "D4", "D10"],
            d_group_col="card1",
            target_encode_cols=["card4", "card6", "P_emaildomain", "R_emaildomain"],
        )
        eng.fit(train_X, train_y)
        result = eng.transform(test_X)

        assert result["C1"].iloc[0] == pytest.approx(3.0)

    def test_d_norm_raw_cols_have_zero_nan(self):
        """D-norm columns (D1, D4, D10) are imputed with global training median in raw output."""
        rng = np.random.default_rng(55)
        n = 20
        df = pd.DataFrame({
            "TransactionDT": rng.integers(86400, 86400 * 500, n),
            "TransactionAmt": rng.uniform(1.0, 500.0, n),
            "card1": rng.choice([1, 2], n),
            "card4": rng.choice(["visa", "mc"], n),
            "card6": rng.choice(["debit", "credit"], n),
            "P_emaildomain": ["gmail.com"] * n,
            "R_emaildomain": ["yahoo.com"] * n,
            "D1": rng.uniform(0.0, 100.0, n),
            "D4": rng.uniform(0.0, 100.0, n),
            "D10": rng.uniform(0.0, 100.0, n),
        })
        # Inject NaN into D1, D4, D10
        df.loc[[0, 5, 10], "D1"] = np.nan
        df.loc[[1, 6], "D4"] = np.nan
        df.loc[[2, 7, 12], "D10"] = np.nan
        y = pd.Series(rng.choice([0, 1], n, p=[0.9, 0.1]))

        eng = FraudFeatureEngineer(
            missing_threshold=0.80,
            d_norm_cols=["D1", "D4", "D10"],
            d_group_col="card1",
            target_encode_cols=["card4", "card6", "P_emaildomain", "R_emaildomain"],
        )
        result = eng.fit_transform(df, y)

        for col in ["D1", "D4", "D10"]:
            assert result[col].isnull().sum() == 0, f"{col} raw column still has NaN"
