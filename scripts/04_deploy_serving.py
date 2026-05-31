"""
Phase 4 serving bootstrap — idempotent.

Creates:
  1. BigQuery prediction_logs table    (fraud_detection.prediction_logs)
  2. BigQuery drift_logs table         (fraud_detection.drift_logs)
  3. Baseline distribution JSON in GCS (monitoring/baseline_distributions.json)
  4. API key Secret in Secret Manager  (fraud-api-key)
  5. Vertex AI Endpoint deployment     (lgb-v8-no-txnid → fraud-detection-endpoint)

Run once after Phase 3.  Safe to re-run (idempotent).
Usage:
  set -a && source .env && set +a
  ENV=dev python scripts/04_deploy_serving.py
"""

import os
import secrets
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_config
from src.utils.logging import get_logger

logger = get_logger(__name__)


def create_bq_table(client, project: str, dataset: str, table_id: str, schema: list, partition_field: str) -> None:
    from google.cloud import bigquery

    table_ref = f"{project}.{dataset}.{table_id}"
    try:
        client.get_table(table_ref)
        logger.info("bq_table_exists", extra={"table": table_ref})
    except Exception:
        table = bigquery.Table(table_ref, schema=schema)
        table.time_partitioning = bigquery.TimePartitioning(
            type_=bigquery.TimePartitioningType.DAY,
            field=partition_field,
        )
        client.create_table(table)
        logger.info("bq_table_created", extra={"table": table_ref})


def step1_create_prediction_logs(config: dict) -> None:
    from google.cloud import bigquery

    project = config["gcp"]["project_id"]
    dataset = config["bigquery"]["dataset"]
    table_id = config["serving"]["prediction_log_table"]

    schema = [
        bigquery.SchemaField("prediction_id",  "STRING",    mode="REQUIRED"),
        bigquery.SchemaField("timestamp",       "TIMESTAMP", mode="REQUIRED"),
        bigquery.SchemaField("fraud_probability","FLOAT64",  mode="REQUIRED"),
        bigquery.SchemaField("prediction",      "INTEGER",   mode="REQUIRED"),
        bigquery.SchemaField("threshold_used",  "FLOAT64",   mode="REQUIRED"),
        bigquery.SchemaField("latency_ms",      "FLOAT64",   mode="REQUIRED"),
        bigquery.SchemaField("model_version",   "STRING",    mode="REQUIRED"),
    ]
    client = bigquery.Client(project=project)
    create_bq_table(client, project, dataset, table_id, schema, "timestamp")
    print(f"✓ prediction_logs: {project}.{dataset}.{table_id}")


def step2_create_drift_logs(config: dict) -> None:
    from google.cloud import bigquery

    project = config["gcp"]["project_id"]
    dataset = config["bigquery"]["dataset"]
    table_id = config["monitoring"]["drift_log_table"]

    schema = [
        bigquery.SchemaField("timestamp",      "TIMESTAMP", mode="REQUIRED"),
        bigquery.SchemaField("feature_name",   "STRING",    mode="REQUIRED"),
        bigquery.SchemaField("psi_score",      "FLOAT64",   mode="REQUIRED"),
        bigquery.SchemaField("alert_fired",    "BOOL",      mode="REQUIRED"),
        bigquery.SchemaField("baseline_mean",  "FLOAT64",   mode="NULLABLE"),
        bigquery.SchemaField("current_mean",   "FLOAT64",   mode="NULLABLE"),
        bigquery.SchemaField("window_days",    "INTEGER",   mode="NULLABLE"),
    ]
    client = bigquery.Client(project=project)
    create_bq_table(client, project, dataset, table_id, schema, "timestamp")
    print(f"✓ drift_logs: {project}.{dataset}.{table_id}")


def step3_save_baseline(config: dict) -> None:
    from src.monitoring.drift_monitor import compute_and_save_baseline

    print("Computing training baseline distributions (queries ~75% of transactions_joined)…")
    compute_and_save_baseline(config)
    bucket = config["storage"]["artifacts_bucket"]
    path = config["monitoring"]["baseline_gcs_path"]
    print(f"✓ baseline: gs://{bucket}/{path}")


def step4_create_api_key_secret(config: dict) -> None:
    from google.cloud import secretmanager
    from google.iam.v1 import policy_pb2

    project = config["gcp"]["project_id"]
    secret_id = config["serving"]["api_key_secret"]
    client = secretmanager.SecretManagerServiceClient()
    parent = f"projects/{project}"
    secret_name = f"{parent}/secrets/{secret_id}"
    serving_sa = f"serviceAccount:serving-sa@{project}.iam.gserviceaccount.com"

    created = False
    try:
        client.get_secret(name=secret_name)
        logger.info("secret_exists", extra={"secret": secret_id})
        print(f"✓ api-key secret already exists: {secret_id}")
    except Exception:
        client.create_secret(
            parent=parent,
            secret_id=secret_id,
            secret={"replication": {"user_managed": {"replicas": [{"location": config["gcp"]["region"]}]}}},
        )
        api_key = secrets.token_urlsafe(32)
        client.add_secret_version(
            parent=secret_name,
            payload={"data": api_key.encode()},
        )
        print(f"✓ api-key secret created: {secret_id}")
        print(f"  API key value (save this): {api_key}")
        logger.info("api_key_secret_created", extra={"secret": secret_id})
        created = True

    # Ensure serving SA can read the secret (idempotent).
    policy = client.get_iam_policy(request={"resource": secret_name})
    accessor_role = "roles/secretmanager.secretAccessor"
    for binding in policy.bindings:
        if binding.role == accessor_role and serving_sa in binding.members:
            print(f"✓ serving SA already has {accessor_role}")
            return
    policy.bindings.add(role=accessor_role, members=[serving_sa])
    client.set_iam_policy(request={"resource": secret_name, "policy": policy})
    print(f"✓ granted {accessor_role} to serving SA on {secret_id}")
    logger.info("secret_iam_granted", extra={"secret": secret_id, "member": serving_sa})


def step4b_grant_vertex_gcs_access(config: dict) -> None:
    """Grant Vertex AI default SA read access on the artifacts bucket."""
    import subprocess

    project = config["gcp"]["project_id"]
    # Vertex AI default SA pattern: service-PROJECT_NUMBER@gcp-sa-aiplatform.iam.gserviceaccount.com
    # Get project number
    result = subprocess.run(
        ["gcloud", "projects", "describe", project, "--format=value(projectNumber)"],
        capture_output=True, text=True, check=True
    )
    project_number = result.stdout.strip()
    vertex_sa = f"service-{project_number}@gcp-sa-aiplatform.iam.gserviceaccount.com"
    bucket = config["storage"]["artifacts_bucket"]

    for role in ["objectViewer", "legacyBucketReader"]:
        subprocess.run(
            ["gsutil", "iam", "ch", f"serviceAccount:{vertex_sa}:{role}", f"gs://{bucket}"],
            capture_output=True, check=False  # idempotent, ignore if already granted
        )
    print(f"✓ granted storage read to {vertex_sa} on gs://{bucket}")
    logger.info("vertex_sa_gcs_access_granted", extra={"sa": vertex_sa, "bucket": bucket})


def step5_deploy_endpoint(config: dict) -> str:
    from src.serving.vertex_endpoint import deploy_model

    print("Deploying model to Vertex AI Endpoint (this takes ~10–15 minutes)…")
    resource_name = deploy_model(config)
    print(f"✓ endpoint: {resource_name}")
    print(f"\nUpdate .env: VERTEX_ENDPOINT_RESOURCE_NAME={resource_name}")
    return resource_name


def main() -> None:
    env = os.getenv("ENV", "dev")
    config = load_config(env)
    project = config["gcp"]["project_id"]
    print(f"\n{'='*60}")
    print(f"Phase 4 Serving Bootstrap — {project}")
    print(f"{'='*60}\n")

    print("Step 1/5 — Create prediction_logs table")
    step1_create_prediction_logs(config)

    print("\nStep 2/5 — Create drift_logs table")
    step2_create_drift_logs(config)

    print("\nStep 3/5 — Compute and save training baseline distributions")
    step3_save_baseline(config)

    print("\nStep 4/5 — Create API key in Secret Manager")
    step4_create_api_key_secret(config)

    print("\nStep 4b/5 — Grant Vertex AI SA GCS read access")
    step4b_grant_vertex_gcs_access(config)

    print("\nStep 5/5 — Deploy model to Vertex AI Endpoint")
    endpoint_resource = step5_deploy_endpoint(config)

    print(f"\n{'='*60}")
    print("Phase 4 Serving Bootstrap COMPLETE")
    print(f"  Endpoint : {endpoint_resource}")
    print(f"  BQ logs  : {project}.{config['bigquery']['dataset']}.{config['serving']['prediction_log_table']}")
    print(f"  Drift log: {project}.{config['bigquery']['dataset']}.{config['monitoring']['drift_log_table']}")
    print(f"{'='*60}\n")
    print("Next: run scripts/05_setup_monitoring.py")


if __name__ == "__main__":
    main()
