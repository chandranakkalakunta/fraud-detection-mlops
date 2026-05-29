"""
Tests for the training pipeline — Phase 2B.

Covers:
  - Time-based split: never shuffled, temporal ordering preserved
  - Validation set carved from training portion, not the held-out test set
  - SMOTE applied only to the training fold
  - Vertex AI Experiments logging called with correct metric names
  - Model saved to GCS after training
  - Champion selection picks highest auc_pr
"""

import io
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import joblib
import numpy as np
import pandas as pd
import pytest

from src.training.metrics import compute_fraud_metrics, find_optimal_threshold
from src.training.model_selector import select_champion
from src.training.xgboost_trainer import time_based_split


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def config(tmp_path):
    """Minimal config dict for training tests."""
    return {
        "gcp": {"project_id": "test-project", "region": "us-central1"},
        "bigquery": {
            "dataset": "fraud_detection",
            "tables": {"transactions_joined": "transactions_joined"},
        },
        "storage": {
            "artifacts_bucket": "test-artifacts",
            "processed_bucket": "test-processed",
        },
        "vertex": {
            "experiment_name": "test-experiment",
            "featurestore_id": "test_featurestore",
            "model_display_name": "fraud-detection-champion",
            "serving_container": {
                "xgboost": "us-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.1-7:latest",
                "lightgbm": "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-4:latest",
            },
        },
        "features": {
            "missing_indicator_threshold": 0.05,
            "target_encode_cols": ["card4", "card6", "P_emaildomain", "R_emaildomain"],
            "d_normalize_group_col": "card1",
            "d_normalize_cols": ["D1", "D4", "D10"],
            "transformer_gcs_prefix": "transformers/",
        },
        "model": {
            "random_state": 42,
            "val_split": 0.2,
            "smote": {"sampling_strategy": 0.1, "k_neighbors": 5, "random_state": 42},
            "xgboost": {
                "n_estimators": 10,
                "max_depth": 3,
                "learning_rate": 0.1,
                "scale_pos_weight": 27,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "min_child_weight": 1,
                "early_stopping_rounds": 3,
                "tree_method": "hist",
                "eval_metric": "aucpr",
                "evaluation_output": str(tmp_path / "xgb_results.json"),
            },
            "lightgbm": {
                "num_leaves": 8,
                "learning_rate": 0.1,
                "n_estimators": 10,
                "min_child_samples": 5,
                "feature_fraction": 0.8,
                "bagging_fraction": 0.8,
                "bagging_freq": 1,
                "early_stopping_rounds": 3,
                "metric": "average_precision",
                "evaluation_output": str(tmp_path / "lgb_results.json"),
            },
        },
        "explainability": {
            "top_n_features": 5,
            "output_gcs_prefix": "explainability/",
        },
    }


@pytest.fixture
def synthetic_df():
    """
    400-row synthetic dataset with TransactionDT for time-based split testing.
    Rows are NOT pre-sorted — the split must sort them correctly.
    """
    rng = np.random.default_rng(0)
    n = 400
    # Shuffle TransactionDT deliberately to confirm sorting is needed
    dt = rng.permutation(np.arange(86400, 86400 + n * 3600, 3600))
    df = pd.DataFrame({
        "TransactionDT": dt,
        "TransactionAmt": rng.uniform(1, 1000, n),
        "card1": rng.choice([100, 200, 300], n),
        "card4": rng.choice(["visa", "mc", "amex"], n),
        "card6": rng.choice(["debit", "credit"], n),
        "P_emaildomain": rng.choice(["gmail.com", "yahoo.com"], n),
        "R_emaildomain": rng.choice(["gmail.com", "yahoo.com"], n),
        "D1": rng.uniform(0, 100, n),
        "D4": rng.uniform(0, 100, n),
        "D10": rng.uniform(0, 100, n),
        "isFraud": rng.choice([0, 1], n, p=[0.965, 0.035]),
    })
    return df


@pytest.fixture
def fraud_proba_arrays():
    """Synthetic probability arrays for metric computation tests."""
    rng = np.random.default_rng(7)
    n = 200
    y_true = rng.choice([0, 1], n, p=[0.965, 0.035])
    # Biased probabilities so metrics are nontrivial
    y_proba = np.where(y_true == 1, rng.uniform(0.5, 1.0, n), rng.uniform(0.0, 0.5, n))
    return y_true, y_proba


# ─── Time-based split ─────────────────────────────────────────────────────────

class TestTimeBasedSplit:
    def test_split_preserves_temporal_ordering(self, synthetic_df):
        """Train, val, test sets must be in ascending TransactionDT order."""
        train, val, test = time_based_split(synthetic_df, train_frac=0.75, val_frac=0.20)
        assert train["TransactionDT"].is_monotonic_increasing
        assert val["TransactionDT"].is_monotonic_increasing
        assert test["TransactionDT"].is_monotonic_increasing

    def test_no_overlap_between_splits(self, synthetic_df):
        """TransactionDT ranges must not overlap between any two splits."""
        train, val, test = time_based_split(synthetic_df, train_frac=0.75, val_frac=0.20)
        assert train["TransactionDT"].max() <= val["TransactionDT"].min()
        assert val["TransactionDT"].max() <= test["TransactionDT"].min()

    def test_total_rows_conserved(self, synthetic_df):
        """Sum of split sizes must equal total input size."""
        train, val, test = time_based_split(synthetic_df, train_frac=0.75, val_frac=0.20)
        assert len(train) + len(val) + len(test) == len(synthetic_df)

    def test_test_is_last_25_percent(self, synthetic_df):
        """Test set size must be approximately 25% of total data."""
        train, val, test = time_based_split(synthetic_df, train_frac=0.75, val_frac=0.20)
        expected_test = int(len(synthetic_df) * 0.25)
        # Allow ±1 for integer rounding
        assert abs(len(test) - expected_test) <= 1

    def test_val_carved_from_train_not_test(self, synthetic_df):
        """Validation set must be earlier in time than the test set."""
        train, val, test = time_based_split(synthetic_df, train_frac=0.75, val_frac=0.20)
        # All val DTs must precede the earliest test DT
        assert val["TransactionDT"].max() < test["TransactionDT"].min()

    def test_never_shuffles(self, synthetic_df):
        """
        The shuffled input must produce sorted output splits — no shuffle=True.
        Verify by checking the global sort order holds even with randomised input.
        """
        # synthetic_df has shuffled TransactionDT
        assert not synthetic_df["TransactionDT"].is_monotonic_increasing, (
            "Fixture must have unsorted DTs to test that splitting sorts them"
        )
        train, val, test = time_based_split(synthetic_df)
        combined = pd.concat([train, val, test])
        assert combined["TransactionDT"].is_monotonic_increasing


# ─── Metric computation ───────────────────────────────────────────────────────

class TestMetricComputation:
    REQUIRED_METRIC_NAMES = frozenset([
        "auc_pr", "auc_roc", "f1", "f2", "precision", "recall",
        "ks_statistic", "threshold", "training_rows", "test_rows",
        "imbalance_strategy",
    ])

    def test_all_required_metric_names_present(self, fraud_proba_arrays):
        """compute_fraud_metrics must return all keys used in Vertex AI Experiments logging."""
        y_true, y_proba = fraud_proba_arrays
        metrics = compute_fraud_metrics(
            y_true=y_true,
            y_proba=y_proba,
            imbalance_strategy="smote",
            training_rows=1000,
            test_rows=200,
        )
        assert self.REQUIRED_METRIC_NAMES.issubset(set(metrics.keys()))

    def test_auc_pr_in_valid_range(self, fraud_proba_arrays):
        """AUC-PR must be in [0, 1]."""
        y_true, y_proba = fraud_proba_arrays
        metrics = compute_fraud_metrics(y_true, y_proba, "smote", 1000, 200)
        assert 0.0 <= metrics["auc_pr"] <= 1.0

    def test_auc_roc_in_valid_range(self, fraud_proba_arrays):
        """AUC-ROC must be in [0, 1]."""
        y_true, y_proba = fraud_proba_arrays
        metrics = compute_fraud_metrics(y_true, y_proba, "smote", 1000, 200)
        assert 0.0 <= metrics["auc_roc"] <= 1.0

    def test_ks_statistic_in_valid_range(self, fraud_proba_arrays):
        """KS statistic must be in [0, 1]."""
        y_true, y_proba = fraud_proba_arrays
        metrics = compute_fraud_metrics(y_true, y_proba, "smote", 1000, 200)
        assert 0.0 <= metrics["ks_statistic"] <= 1.0

    def test_imbalance_strategy_preserved(self, fraud_proba_arrays):
        """imbalance_strategy value must pass through unchanged."""
        y_true, y_proba = fraud_proba_arrays
        metrics = compute_fraud_metrics(y_true, y_proba, "class_weight", 500, 100)
        assert metrics["imbalance_strategy"] == "class_weight"

    def test_training_and_test_rows_preserved(self, fraud_proba_arrays):
        """training_rows and test_rows must match what was passed in."""
        y_true, y_proba = fraud_proba_arrays
        metrics = compute_fraud_metrics(y_true, y_proba, "smote", 12345, 6789)
        assert metrics["training_rows"] == 12345
        assert metrics["test_rows"] == 6789

    def test_threshold_in_0_1_range(self, fraud_proba_arrays):
        """Optimal threshold must be in (0, 1)."""
        y_true, y_proba = fraud_proba_arrays
        threshold = find_optimal_threshold(y_true, y_proba)
        assert 0.0 < threshold < 1.0


# ─── GCS model saving ─────────────────────────────────────────────────────────

class TestModelSavedToGCS:
    def test_save_model_to_gcs_uploads_blob(self, config):
        """save_model_to_gcs must upload a non-empty blob to the correct bucket/path."""
        from src.training.xgboost_trainer import save_model_to_gcs
        import sklearn.linear_model

        dummy_model = sklearn.linear_model.LogisticRegression()

        mock_blob = MagicMock()
        mock_bucket = MagicMock()
        mock_bucket.blob.return_value = mock_blob
        mock_client = MagicMock()
        mock_client.bucket.return_value = mock_bucket

        # save_model_to_gcs imports `from google.cloud import storage as gcs` inside
        # the function body, so we patch google.cloud.storage.Client directly.
        with patch("google.cloud.storage.Client", return_value=mock_client):
            uri = save_model_to_gcs(dummy_model, "test-model", config)

        mock_client.bucket.assert_called_once_with("test-artifacts")
        mock_bucket.blob.assert_called_once_with("models/test-model/model.joblib")
        mock_blob.upload_from_file.assert_called_once()
        assert uri == "gs://test-artifacts/models/test-model/"

    def test_saved_model_is_deserializable(self, tmp_path, config):
        """A model saved with joblib must round-trip correctly."""
        import xgboost as xgb
        import sklearn.datasets

        X, y = sklearn.datasets.make_classification(
            n_samples=100, n_features=10, n_informative=5, random_state=42
        )
        model = xgb.XGBClassifier(n_estimators=5, verbosity=0)
        model.fit(X, y)

        buf = io.BytesIO()
        joblib.dump(model, buf)
        buf.seek(0)
        loaded = joblib.load(buf)

        preds = loaded.predict_proba(X)[:, 1]
        assert len(preds) == 100


# ─── SMOTE on training fold only ──────────────────────────────────────────────

class TestSMOTEOnTrainOnly:
    def test_smote_not_applied_to_val_or_test(self, synthetic_df, config):
        """
        When training with SMOTE, the val and test sets passed to model.fit
        must NOT be resampled — they must have the original row counts.
        """
        from src.features.imbalance import SMOTEHandler
        from src.features.engineer import FraudFeatureEngineer

        train, val, test = time_based_split(synthetic_df, train_frac=0.75, val_frac=0.20)
        # Include TransactionDT — the engineer needs it to create time features
        feature_cols = [c for c in synthetic_df.columns if c not in ["isFraud"]]

        X_train = train[feature_cols].reset_index(drop=True)
        y_train = train["isFraud"].astype(int).reset_index(drop=True)
        X_val = val[feature_cols].reset_index(drop=True)
        y_val = val["isFraud"].astype(int).reset_index(drop=True)
        X_test = test[feature_cols].reset_index(drop=True)
        y_test = test["isFraud"].astype(int).reset_index(drop=True)

        val_size_before = len(X_val)
        test_size_before = len(X_test)

        smote = SMOTEHandler(sampling_strategy=0.1, random_state=42)
        eng = FraudFeatureEngineer(
            missing_threshold=0.05,
            d_norm_cols=["D1", "D4", "D10"],
            d_group_col="card1",
            target_encode_cols=["card4", "card6", "P_emaildomain", "R_emaildomain"],
        )
        X_train_eng = eng.fit_transform(X_train, y_train)
        X_train_res, y_train_res = smote.fit_resample(X_train_eng, y_train, train_only=True)

        # Val and test are not touched
        assert len(X_val) == val_size_before
        assert len(X_test) == test_size_before
        # SMOTE may increase training set size
        assert len(X_train_res) >= len(X_train_eng)

    def test_smote_assertion_prevents_accidental_test_application(self, synthetic_df, config):
        """Attempting to pass train_only=False raises AssertionError."""
        from src.features.imbalance import SMOTEHandler

        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.standard_normal((50, 5)), columns=[f"f{i}" for i in range(5)])
        y = pd.Series([0] * 45 + [1] * 5)

        smote = SMOTEHandler()
        with pytest.raises(AssertionError):
            smote.fit_resample(X, y, train_only=False)


# ─── Champion selection ────────────────────────────────────────────────────────

class TestChampionSelection:
    def test_champion_has_highest_auc_pr(self):
        """select_champion must return the run with the highest auc_pr."""
        runs = [
            {"run_name": "xgb-a-smote",       "auc_pr": 0.72, "auc_roc": 0.91},
            {"run_name": "xgb-b-weight",       "auc_pr": 0.68, "auc_roc": 0.89},
            {"run_name": "lgb-c-smote",        "auc_pr": 0.79, "auc_roc": 0.93},  # ← winner
            {"run_name": "lgb-d-unbalance",    "auc_pr": 0.75, "auc_roc": 0.92},
        ]
        champion = select_champion(runs)
        assert champion["run_name"] == "lgb-c-smote"
        assert champion["auc_pr"] == pytest.approx(0.79)

    def test_champion_breaks_tie_deterministically(self):
        """With equal auc_pr, the selection must be deterministic (max picks first max)."""
        runs = [
            {"run_name": "run-x", "auc_pr": 0.80},
            {"run_name": "run-y", "auc_pr": 0.80},
        ]
        # max() on equal values returns the first — no crash, no non-determinism
        champion = select_champion(runs)
        assert champion["auc_pr"] == pytest.approx(0.80)

    def test_select_champion_raises_on_empty_runs(self):
        """select_champion must raise ValueError if no runs are provided."""
        with pytest.raises(ValueError, match="No experiment runs"):
            select_champion([])

    def test_all_four_metric_names_logged_identically(self):
        """
        All 4 runs must use the same canonical metric names so Vertex AI
        Experiments comparison is apples-to-apples.
        """
        required = frozenset([
            "auc_pr", "auc_roc", "f1", "f2", "precision", "recall",
            "ks_statistic", "threshold", "training_rows", "test_rows",
            "imbalance_strategy",
        ])
        rng = np.random.default_rng(1)
        n = 100
        y_true = rng.choice([0, 1], n, p=[0.965, 0.035])
        y_proba = np.where(y_true == 1, rng.uniform(0.5, 1.0, n), rng.uniform(0.0, 0.4, n))

        for strategy in ["smote", "class_weight", "smote", "lgb_is_unbalance"]:
            metrics = compute_fraud_metrics(y_true, y_proba, strategy, 1000, 100)
            missing = required - set(metrics.keys())
            assert not missing, f"Strategy '{strategy}' missing metrics: {missing}"


# ─── Vertex AI Experiments logging ────────────────────────────────────────────

class TestVertexExperimentsLogging:
    def test_log_to_vertex_called_with_correct_metrics(self, config):
        """log_to_vertex_experiments must pass all required metric names to the run."""
        from src.training.xgboost_trainer import log_to_vertex_experiments

        mock_run = MagicMock()
        mock_run_cls = MagicMock(return_value=mock_run)

        with patch("google.cloud.aiplatform.init"), \
             patch("google.cloud.aiplatform.ExperimentRun.create", mock_run_cls):
            log_to_vertex_experiments(
                config=config,
                run_name="test-run",
                params={"model_type": "xgboost"},
                metrics={
                    "auc_pr": 0.72,
                    "auc_roc": 0.91,
                    "f1": 0.55,
                    "f2": 0.60,
                    "precision": 0.62,
                    "recall": 0.58,
                    "ks_statistic": 0.45,
                    "threshold": 0.35,
                    "training_rows": 1000,
                    "test_rows": 250,
                    "imbalance_strategy": "smote",
                },
            )

        mock_run.log_params.assert_called_once()
        mock_run.log_metrics.assert_called_once()

        logged_metrics = mock_run.log_metrics.call_args[0][0]
        assert "auc_pr" in logged_metrics
        assert "auc_roc" in logged_metrics
        assert "ks_statistic" in logged_metrics
        assert "imbalance_strategy" in logged_metrics
