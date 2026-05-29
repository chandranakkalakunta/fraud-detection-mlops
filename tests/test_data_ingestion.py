"""
Unit tests for data ingestion pipeline.

Tests schema validation, null thresholds, row counts, and join integrity
without hitting BigQuery — all GCP calls are mocked.
"""

from unittest.mock import MagicMock

import pytest

from src.ingestion.loader import (
    IDENTITY_SCHEMA,
    TRANSACTION_SCHEMA,
    gcs_uri,
    validate_join_integrity,
    validate_table,
)


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def base_config() -> dict:
    return {
        "gcp": {"project_id": "test-project"},
        "bigquery": {
            "dataset": "fraud_detection",
            "kms_key_name": "projects/test-project/locations/us/keyRings/r/cryptoKeys/k",
            "tables": {
                "transactions_raw": "transactions_raw",
                "identity_raw": "identity_raw",
                "transactions_joined": "transactions_joined",
            },
        },
        "storage": {
            "raw_bucket": "test-raw",
            "raw_data_prefix": "ieee-cis/",
            "audit_bucket": "test-audit",
        },
        "data_validation": {
            "max_null_pct_numeric": 0.95,
            "max_null_pct_categorical": 0.95,
            "min_rows_transactions": 500000,
            "min_rows_identity": 100000,
            "expected_fraud_rate_min": 0.03,
            "expected_fraud_rate_max": 0.05,
        },
    }


# ─── Schema Tests ────────────────────────────────────────────────────────────

class TestTransactionSchema:
    def test_transaction_id_required(self):
        txn_id = next(f for f in TRANSACTION_SCHEMA if f.name == "TransactionID")
        assert txn_id.mode == "REQUIRED"
        assert txn_id.field_type == "INTEGER"

    def test_is_fraud_nullable(self):
        is_fraud = next(f for f in TRANSACTION_SCHEMA if f.name == "isFraud")
        assert is_fraud.mode == "NULLABLE"
        assert is_fraud.field_type == "INTEGER"

    def test_v_features_present(self):
        v_fields = {f.name for f in TRANSACTION_SCHEMA if f.name.startswith("V")}
        assert len(v_fields) == 339
        assert "V1" in v_fields
        assert "V339" in v_fields

    def test_c_features_count(self):
        c_fields = [f for f in TRANSACTION_SCHEMA if f.name.startswith("C")]
        assert len(c_fields) == 14

    def test_d_features_count(self):
        d_fields = [f for f in TRANSACTION_SCHEMA if f.name.startswith("D")]
        assert len(d_fields) == 15

    def test_m_features_are_string(self):
        m_fields = [f for f in TRANSACTION_SCHEMA if f.name.startswith("M")]
        assert len(m_fields) == 9
        assert all(f.field_type == "STRING" for f in m_fields)

    def test_transaction_amt_is_float(self):
        amt = next(f for f in TRANSACTION_SCHEMA if f.name == "TransactionAmt")
        assert amt.field_type == "FLOAT"


class TestIdentitySchema:
    def test_transaction_id_required(self):
        txn_id = next(f for f in IDENTITY_SCHEMA if f.name == "TransactionID")
        assert txn_id.mode == "REQUIRED"

    def test_device_fields_present(self):
        names = {f.name for f in IDENTITY_SCHEMA}
        assert "DeviceType" in names
        assert "DeviceInfo" in names

    def test_id_string_fields(self):
        # All 38 id_ fields are STRING — mix of numeric and categorical values
        # in the raw data (id_12: "Found"/"NotFound", id_30: OS, etc.)
        str_fields = [
            f for f in IDENTITY_SCHEMA
            if f.name.startswith("id_") and f.field_type == "STRING"
        ]
        assert len(str_fields) == 38

    def test_id_total_count(self):
        id_fields = [f for f in IDENTITY_SCHEMA if f.name.startswith("id_")]
        assert len(id_fields) == 38


# ─── GCS URI Tests ────────────────────────────────────────────────────────────

class TestGcsUri:
    def test_basic_uri(self):
        uri = gcs_uri("my-bucket", "ieee-cis/", "train_transaction.csv")
        assert uri == "gs://my-bucket/ieee-cis/train_transaction.csv"

    def test_strips_trailing_slash(self):
        uri = gcs_uri("my-bucket", "ieee-cis///", "train_identity.csv")
        assert uri == "gs://my-bucket/ieee-cis/train_identity.csv"

    def test_no_prefix(self):
        uri = gcs_uri("my-bucket", "", "file.csv")
        assert uri == "gs://my-bucket/file.csv"


# ─── Row Count Validation Tests ──────────────────────────────────────────────

class TestValidateTable:
    def _make_bq_client(self, row_count: int, column_count: int = 50) -> MagicMock:
        client = MagicMock()

        def query_side_effect(q):
            job = MagicMock()
            mock_row = MagicMock()
            if "COUNT(*) as cnt" in q and "INFORMATION_SCHEMA" not in q:
                mock_row.__getitem__ = lambda self, key: row_count
                job.result.return_value = [mock_row]
            else:
                mock_row.__getitem__ = lambda self, key: column_count
                job.result.return_value = [mock_row]
            return job

        client.query.side_effect = query_side_effect
        return client

    def test_passes_when_row_count_sufficient(self, base_config):
        client = self._make_bq_client(row_count=590000, column_count=394)
        result = validate_table(client, "transactions_raw", 500000, "transactions_raw", base_config)
        assert result["row_count_pass"] is True
        assert result["row_count"] == 590000

    def test_fails_when_row_count_insufficient(self, base_config):
        client = self._make_bq_client(row_count=400000)
        result = validate_table(client, "transactions_raw", 500000, "transactions_raw", base_config)
        assert result["row_count_pass"] is False

    def test_identity_table_min_rows(self, base_config):
        client = self._make_bq_client(row_count=144233)
        result = validate_table(client, "identity_raw", 100000, "identity_raw", base_config)
        assert result["row_count_pass"] is True


# ─── Join Integrity Tests ─────────────────────────────────────────────────────

class TestValidateJoinIntegrity:
    def _make_client(self, stats: dict) -> MagicMock:
        client = MagicMock()

        def query_side_effect(q):
            mock_job = MagicMock()
            mock_row = MagicMock()
            mock_row.__getitem__ = lambda self, key: stats[key]
            mock_job.result.return_value = [mock_row]
            return mock_job

        client.query.side_effect = query_side_effect
        return client

    def test_passes_valid_join(self, base_config):
        stats = {
            "total_joined": 590540,
            "null_txn_ids": 0,
            "null_labels": 0,
            "fraud_rate_pct": 3.5,
            "unique_txn_ids": 590540,
        }
        client = self._make_client(stats)
        validate_join_integrity(client, base_config)  # must not raise

    def test_fails_on_null_transaction_ids(self, base_config):
        stats = {
            "total_joined": 590540,
            "null_txn_ids": 100,
            "null_labels": 0,
            "fraud_rate_pct": 3.5,
            "unique_txn_ids": 590440,
        }
        client = self._make_client(stats)
        with pytest.raises(RuntimeError, match="no_null_transaction_ids"):
            validate_join_integrity(client, base_config)

    def test_fails_on_fraud_rate_out_of_range(self, base_config):
        stats = {
            "total_joined": 590540,
            "null_txn_ids": 0,
            "null_labels": 0,
            "fraud_rate_pct": 15.0,
            "unique_txn_ids": 590540,
        }
        client = self._make_client(stats)
        with pytest.raises(RuntimeError, match="fraud_rate_in_range"):
            validate_join_integrity(client, base_config)

    def test_fails_on_duplicate_transaction_ids(self, base_config):
        stats = {
            "total_joined": 590540,
            "null_txn_ids": 0,
            "null_labels": 0,
            "fraud_rate_pct": 3.5,
            "unique_txn_ids": 580000,
        }
        client = self._make_client(stats)
        with pytest.raises(RuntimeError, match="no_duplicate_transaction_ids"):
            validate_join_integrity(client, base_config)

    def test_fails_on_null_labels(self, base_config):
        stats = {
            "total_joined": 590540,
            "null_txn_ids": 0,
            "null_labels": 50,
            "fraud_rate_pct": 3.5,
            "unique_txn_ids": 590540,
        }
        client = self._make_client(stats)
        with pytest.raises(RuntimeError, match="no_null_labels"):
            validate_join_integrity(client, base_config)


# ─── Null Threshold Constants ────────────────────────────────────────────────

class TestNullThresholds:
    """Config-driven thresholds must reflect IEEE-CIS dataset reality."""

    def test_numeric_threshold_accommodates_v_features(self, base_config):
        # V-features can be ~95% null — threshold must allow this
        assert base_config["data_validation"]["max_null_pct_numeric"] >= 0.9

    def test_categorical_threshold_accommodates_id_fields(self, base_config):
        # Identity fields only present for ~25% of transactions
        assert base_config["data_validation"]["max_null_pct_categorical"] >= 0.7
