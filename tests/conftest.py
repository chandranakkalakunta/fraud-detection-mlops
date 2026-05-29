"""Shared fixtures for all test modules."""

import os

import pytest


@pytest.fixture(autouse=True)
def set_required_env_vars(monkeypatch):
    """Provide minimal env vars so config loads without error in tests."""
    defaults = {
        "GCP_PROJECT_ID": "test-project",
        "GCP_REGION": "us-central1",
        "GCP_ZONE": "us-central1-a",
        "GCS_RAW_BUCKET": "test-raw",
        "GCS_PROCESSED_BUCKET": "test-processed",
        "GCS_ARTIFACTS_BUCKET": "test-artifacts",
        "GCS_AUDIT_BUCKET": "test-audit",
        "BQ_DATASET": "fraud_detection",
        "BQ_LOCATION": "US",
        "BQ_KMS_KEY_NAME": "projects/test-project/locations/us/keyRings/r/cryptoKeys/k",
        "VERTEX_STAGING_BUCKET": "gs://test-artifacts/staging",
        "VERTEX_PIPELINE_ROOT": "gs://test-artifacts/pipelines",
        "VERTEX_EXPERIMENT_NAME": "test-experiment",
        "VERTEX_FEATURESTORE_ID": "test_featurestore",
        "TRAINING_SA": "training-sa@test-project.iam.gserviceaccount.com",
        "SERVING_SA": "serving-sa@test-project.iam.gserviceaccount.com",
        "PIPELINE_SA": "pipeline-sa@test-project.iam.gserviceaccount.com",
        "MONITORING_SA": "monitoring-sa@test-project.iam.gserviceaccount.com",
        "KAGGLE_SECRET_NAME": "kaggle-credentials",
        "BQ_CMEK_SECRET_NAME": "bq-cmek-key-name",
        "MODEL_MAX_ITER": "100",
        "MODEL_RANDOM_STATE": "42",
        "MODEL_TEST_SIZE": "0.2",
        "MODEL_CV_FOLDS": "3",
        "AR_REPOSITORY": "test-repo",
        "AR_REGION": "us-central1",
        "ALERT_EMAIL": "test@example.com",
        "ENV": "dev",
        "FEATURE_MISSING_THRESHOLD": "0.05",
    }
    for key, value in defaults.items():
        if not os.getenv(key):
            monkeypatch.setenv(key, value)
