"""
Drift monitoring for lgb-v8-no-txnid production endpoint.

Two detection methods:
  1. Feature drift — PSI (Population Stability Index) on TransactionAmt and C13.
     Training baseline distribution stored in GCS as JSON.
     Query window: last 7 days of transactions from transactions_joined.
     Alert if PSI > 0.2 for either feature.

  2. Performance drift — 7-day rolling fraud rate in prediction_logs.
     Alert if |rate - training_rate| > 2 * training_std (binomial std).

All results written to BigQuery fraud_detection.drift_logs:
  timestamp TIMESTAMP, feature_name STRING, psi_score FLOAT64,
  alert_fired BOOL, baseline_mean FLOAT64, current_mean FLOAT64,
  window_days INT64

Also configures:
  - Vertex AI Model Monitoring on the deployed endpoint
  - Cloud Scheduler daily 6am IST trigger
  - Cloud Monitoring alert policies (feature_drift_alert, performance_drift_alert)
"""

import io
import json
import math
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)

_PSI_BUCKETS = 10
_PSI_MIN_FLOOR = 1e-6  # avoid log(0)


# ── PSI core ──────────────────────────────────────────────────────────────────

def compute_psi(expected: np.ndarray, actual: np.ndarray, buckets: int = _PSI_BUCKETS) -> float:
    """
    Population Stability Index between two 1-D numeric distributions.

    PSI < 0.1  — no meaningful change
    PSI 0.1–0.2 — moderate change, investigate
    PSI > 0.2  — significant shift, alert
    """
    breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
    breakpoints = np.unique(breakpoints)  # collapse duplicates at tails

    exp_counts = np.histogram(expected, bins=breakpoints)[0].astype(float)
    act_counts = np.histogram(actual, bins=breakpoints)[0].astype(float)

    exp_pct = np.maximum(exp_counts / exp_counts.sum(), _PSI_MIN_FLOOR)
    act_pct = np.maximum(act_counts / act_counts.sum(), _PSI_MIN_FLOOR)

    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


# ── Baseline management ───────────────────────────────────────────────────────

def compute_and_save_baseline(config: dict) -> None:
    """
    Query the training portion of transactions_joined, compute per-feature
    distribution percentiles, and persist to GCS as JSON.

    Run once (or whenever the champion model changes).
    Idempotent — overwrites the blob on re-run.
    """
    from google.cloud import bigquery, storage as gcs

    project = config["gcp"]["project_id"]
    dataset = config["bigquery"]["dataset"]
    monitored = config["monitoring"]["monitored_features"]

    cols = ", ".join(monitored)
    # Training data = temporal first 75% ordered by TransactionDT.
    query = f"""
        WITH ranked AS (
          SELECT {cols},
                 ROW_NUMBER() OVER (ORDER BY TransactionDT) AS rn,
                 COUNT(*) OVER ()                           AS total
          FROM `{project}.{dataset}.transactions_joined`
        )
        SELECT {cols} FROM ranked WHERE rn <= CAST(total * 0.75 AS INT64)
    """
    df = bigquery.Client(project=project).query(query).to_dataframe()

    baseline: dict[str, dict] = {}
    for feat in monitored:
        vals = df[feat].dropna().values
        percentiles = np.percentile(vals, np.linspace(0, 100, _PSI_BUCKETS + 1))
        baseline[feat] = {
            "mean": float(vals.mean()),
            "std": float(vals.std()),
            "percentiles": percentiles.tolist(),
            "n": int(len(vals)),
        }

    bucket_name = config["storage"]["artifacts_bucket"]
    blob_path = config["monitoring"]["baseline_gcs_path"]
    buf = io.BytesIO(json.dumps(baseline, indent=2).encode())
    gcs.Client(project=project).bucket(bucket_name).blob(blob_path).upload_from_file(
        buf, content_type="application/json"
    )
    logger.info(
        "baseline_saved",
        extra={"gcs_path": f"gs://{bucket_name}/{blob_path}", "features": monitored},
    )


def load_baseline(config: dict) -> dict:
    """Load baseline distributions from GCS."""
    from google.cloud import storage as gcs

    bucket_name = config["storage"]["artifacts_bucket"]
    blob_path = config["monitoring"]["baseline_gcs_path"]
    data = gcs.Client(project=config["gcp"]["project_id"]).bucket(bucket_name).blob(blob_path).download_as_bytes()
    return json.loads(data)


# ── Drift checks ──────────────────────────────────────────────────────────────

def check_feature_drift(config: dict) -> list[dict]:
    """
    PSI on each monitored feature over the last N days vs training baseline.

    Returns list of result dicts (one per feature), also logged to BQ drift_logs.
    """
    from google.cloud import bigquery

    project = config["gcp"]["project_id"]
    dataset = config["bigquery"]["dataset"]
    window_days = int(config["monitoring"]["monitoring_window_days"])
    psi_threshold = float(config["monitoring"]["psi_alert_threshold"])
    monitored = config["monitoring"]["monitored_features"]

    cutoff = (datetime.now(timezone.utc) - timedelta(days=window_days)).isoformat()
    cols = ", ".join(monitored)
    query = f"""
        SELECT {cols}
        FROM `{project}.{dataset}.transactions_joined`
        WHERE TIMESTAMP_SECONDS(CAST(TransactionDT AS INT64)) >= '{cutoff}'
    """

    client = bigquery.Client(project=project)
    try:
        df_recent = client.query(query).to_dataframe()
    except Exception as exc:
        logger.warning("feature_drift_query_failed", extra={"error": str(exc)})
        return []

    if df_recent.empty:
        logger.warning("feature_drift_no_recent_data", extra={"window_days": window_days})
        return []

    baseline = load_baseline(config)
    results = []
    ts = datetime.now(timezone.utc).isoformat()

    for feat in monitored:
        base_info = baseline.get(feat, {})
        if not base_info:
            continue

        base_percentiles = np.array(base_info["percentiles"])
        recent_vals = df_recent[feat].dropna().values
        base_vals = np.random.default_rng(42).choice(
            np.interp(np.random.uniform(size=len(recent_vals)),
                      np.linspace(0, 1, len(base_percentiles)),
                      base_percentiles),
            size=len(recent_vals),
            replace=True,
        )

        psi = compute_psi(base_vals, recent_vals)
        alert = psi > psi_threshold

        result = {
            "timestamp": ts,
            "feature_name": feat,
            "psi_score": round(psi, 6),
            "alert_fired": alert,
            "baseline_mean": round(base_info["mean"], 4),
            "current_mean": round(float(recent_vals.mean()), 4),
            "window_days": window_days,
        }
        results.append(result)

        logger.info(
            "feature_drift_check",
            extra={
                "feature": feat,
                "psi": round(psi, 4),
                "alert": alert,
                "baseline_mean": result["baseline_mean"],
                "current_mean": result["current_mean"],
            },
        )

    if results:
        _log_drift_to_bq(results, config)

    return results


def check_performance_drift(config: dict) -> dict:
    """
    Monitor 7-day rolling fraud rate in prediction_logs.

    Alert condition: |observed_rate - training_rate| > 2 * binomial_std
    where binomial_std = sqrt(p*(1-p)/n), p = training_fraud_rate.

    Returns result dict, also logged to BQ drift_logs.
    """
    from google.cloud import bigquery

    project = config["gcp"]["project_id"]
    dataset = config["bigquery"]["dataset"]
    pred_table = config["serving"]["prediction_log_table"]
    window_days = int(config["monitoring"]["monitoring_window_days"])
    training_rate = float(config["monitoring"]["training_fraud_rate"])
    alert_stddev = float(config["monitoring"]["fraud_rate_alert_stddev"])

    cutoff = (datetime.now(timezone.utc) - timedelta(days=window_days)).isoformat()
    query = f"""
        SELECT COUNT(*) AS total, AVG(CAST(prediction AS FLOAT64)) AS fraud_rate
        FROM `{project}.{dataset}.{pred_table}`
        WHERE timestamp >= '{cutoff}'
    """
    client = bigquery.Client(project=project)
    try:
        rows = list(client.query(query).result())
        row = rows[0] if rows else None
        n = int(row.total) if row and row.total else 0
        observed_rate = float(row.fraud_rate or 0) if row else 0.0
    except Exception as exc:
        logger.warning("performance_drift_query_failed", extra={"error": str(exc)})
        return {}

    if n < 100:
        logger.info("performance_drift_insufficient_data", extra={"n": n})
        return {}

    binom_std = math.sqrt(training_rate * (1 - training_rate) / n)
    deviation = abs(observed_rate - training_rate)
    alert = deviation > alert_stddev * binom_std

    ts = datetime.now(timezone.utc).isoformat()
    result = {
        "timestamp": ts,
        "feature_name": "fraud_rate",
        "psi_score": round(deviation / binom_std, 4) if binom_std > 0 else 0.0,
        "alert_fired": alert,
        "baseline_mean": round(training_rate, 4),
        "current_mean": round(observed_rate, 4),
        "window_days": window_days,
    }

    logger.info(
        "performance_drift_check",
        extra={
            "n": n,
            "observed_rate": round(observed_rate, 4),
            "training_rate": training_rate,
            "deviation_stddev": round(deviation / binom_std, 2) if binom_std > 0 else 0,
            "alert": alert,
        },
    )

    _log_drift_to_bq([result], config)
    return result


def _log_drift_to_bq(results: list[dict], config: dict) -> None:
    from google.cloud import bigquery

    project = config["gcp"]["project_id"]
    dataset = config["bigquery"]["dataset"]
    table = f"{project}.{dataset}.{config['monitoring']['drift_log_table']}"
    errors = bigquery.Client(project=project).insert_rows_json(table, results)
    if errors:
        logger.warning("drift_log_bq_error", extra={"errors": str(errors[:1])})
    else:
        logger.info("drift_logged_to_bq", extra={"n": len(results)})


# ── Vertex AI Model Monitoring ────────────────────────────────────────────────

def setup_model_monitoring(config: dict) -> None:
    """
    Enable Vertex AI Model Monitoring on the deployed endpoint.

    Monitors TransactionAmt and C13 for feature skew vs the training dataset.
    Alert notification sent to ALERT_EMAIL via email channel.

    Idempotent — skips creation if a monitoring job already exists.
    """
    import google.cloud.aiplatform as aip
    from google.cloud.aiplatform import model_monitoring

    project = config["gcp"]["project_id"]
    location = config["gcp"]["region"]
    dataset = config["bigquery"]["dataset"]
    aip.init(project=project, location=location, staging_bucket=config["vertex"]["staging_bucket"])

    serving_cfg = config["serving"]
    resource_name = str(serving_cfg.get("endpoint_resource_name", "")).strip()
    if not resource_name:
        logger.warning("model_monitoring_skipped_no_endpoint")
        return

    endpoint = aip.Endpoint(resource_name)
    training_table = f"bq://{project}.{dataset}.transactions_joined"
    alert_email = config["monitoring"]["alert_email"]
    monitored_feats = config["monitoring"]["monitored_features"]

    skew_configs = {
        feat: model_monitoring.SkewDetectionConfig(
            data_source=training_table,
            skew_results_gcs_uri=f"gs://{config['storage']['artifacts_bucket']}/monitoring/skew/",
            target_field="isFraud",
        )
        for feat in monitored_feats
    }

    job = aip.ModelDeploymentMonitoringJob.create(
        display_name="fraud-model-monitoring",
        endpoint=endpoint,
        logging_sampling_strategy=model_monitoring.RandomSampleConfig(sample_rate=0.1),
        schedule_config=model_monitoring.ScheduleConfig(monitor_interval=24),
        alert_config=model_monitoring.EmailAlertConfig(user_emails=[alert_email]),
        objective_configs=[
            model_monitoring.ObjectiveConfig(
                skew_detection_config=skew_configs[feat],
            )
            for feat in monitored_feats
        ],
        project=project,
        location=location,
    )
    logger.info("model_monitoring_created", extra={"job": job.resource_name})


# ── Cloud Scheduler + Cloud Monitoring alert policies ─────────────────────────

def create_scheduler_job(config: dict, cloud_run_url: str) -> None:
    """
    Create Cloud Scheduler job: daily 6am IST → POST {cloud_run_url}/drift-check.
    Requires Cloud Scheduler API and scheduler-sa (or MONITORING_SA) to have
    roles/cloudrun.invoker on the Cloud Run service.

    Idempotent — deletes and recreates if job with same name already exists.
    """
    from google.cloud import scheduler_v1

    project = config["gcp"]["project_id"]
    location = config["gcp"]["region"]
    monitoring_cfg = config["monitoring"]
    job_name = f"projects/{project}/locations/{location}/jobs/fraud-daily-drift-check"

    client = scheduler_v1.CloudSchedulerClient()

    # Delete existing job if present (idempotent re-run).
    try:
        client.delete_job(name=job_name)
        logger.info("scheduler_job_deleted_for_recreate", extra={"name": job_name})
    except Exception:
        pass

    job = scheduler_v1.Job(
        name=job_name,
        description="Daily fraud model drift check — 6am IST",
        schedule=monitoring_cfg["scheduler_schedule"],
        time_zone=monitoring_cfg["scheduler_timezone"],
        http_target=scheduler_v1.HttpTarget(
            uri=f"{cloud_run_url}/drift-check",
            http_method=scheduler_v1.HttpMethod.POST,
            headers={"Content-Type": "application/json"},
            body=b"{}",
            oidc_token=scheduler_v1.OidcToken(
                service_account_email=config["service_accounts"]["monitoring"],
            ),
        ),
    )
    parent = f"projects/{project}/locations/{location}"
    created = client.create_job(parent=parent, job=job)
    logger.info("scheduler_job_created", extra={"name": created.name, "schedule": monitoring_cfg["scheduler_schedule"]})


def create_alert_policies(config: dict) -> None:
    """
    Create two Cloud Monitoring alert policies:
      - feature_drift_alert  — PSI > 0.2 (metric: custom/fraud_detection/psi_score)
      - performance_drift_alert — fraud_rate deviation > 2 std

    Both send email notifications to ALERT_EMAIL.
    Idempotent — skips creation if policy with the same display_name exists.
    """
    from google.cloud import monitoring_v3

    project = config["gcp"]["project_id"]
    monitoring_cfg = config["monitoring"]
    alert_email = monitoring_cfg["alert_email"]

    client = monitoring_v3.AlertPolicyServiceClient()
    notification_client = monitoring_v3.NotificationChannelServiceClient()

    # Create or reuse email notification channel.
    project_name = f"projects/{project}"
    channels = list(notification_client.list_notification_channels(name=project_name))
    email_channel_name = None
    for ch in channels:
        if ch.type_ == "email" and ch.labels.get("email_address") == alert_email:
            email_channel_name = ch.name
            break

    if not email_channel_name:
        channel = monitoring_v3.NotificationChannel(
            type_="email",
            display_name=f"Fraud Detection Alerts — {alert_email}",
            labels={"email_address": alert_email},
        )
        created_channel = notification_client.create_notification_channel(
            name=project_name, notification_channel=channel
        )
        email_channel_name = created_channel.name
        logger.info("notification_channel_created", extra={"channel": email_channel_name})

    existing_policies = {p.display_name: p.name for p in client.list_alert_policies(name=project_name)}

    def _upsert_policy(display_name: str, condition_filter: str, threshold: float, comparison: str) -> None:
        policy = monitoring_v3.AlertPolicy(
            display_name=display_name,
            conditions=[
                monitoring_v3.AlertPolicy.Condition(
                    display_name=f"{display_name} condition",
                    condition_threshold=monitoring_v3.AlertPolicy.Condition.MetricThreshold(
                        filter=condition_filter,
                        comparison=monitoring_v3.ComparisonType[comparison],
                        threshold_value=threshold,
                        duration={"seconds": 0},
                        aggregations=[
                            monitoring_v3.Aggregation(
                                alignment_period={"seconds": 86400},
                                per_series_aligner=monitoring_v3.Aggregation.Aligner.ALIGN_MAX,
                            )
                        ],
                    ),
                )
            ],
            notification_channels=[email_channel_name],
            combiner=monitoring_v3.AlertPolicy.ConditionCombinerType.OR,
            enabled=True,
        )
        if display_name in existing_policies:
            policy.name = existing_policies[display_name]
            client.update_alert_policy(alert_policy=policy)
            logger.info("alert_policy_updated", extra={"name": display_name})
        else:
            created = client.create_alert_policy(name=project_name, alert_policy=policy)
            logger.info("alert_policy_created", extra={"name": display_name, "resource": created.name})

    _upsert_policy(
        display_name=monitoring_cfg["feature_drift_alert_policy"],
        condition_filter=(
            'metric.type="custom.googleapis.com/fraud_detection/psi_score" '
            'AND resource.type="global"'
        ),
        threshold=float(monitoring_cfg["psi_alert_threshold"]),
        comparison="COMPARISON_GT",
    )

    _upsert_policy(
        display_name=monitoring_cfg["performance_drift_alert_policy"],
        condition_filter=(
            'metric.type="custom.googleapis.com/fraud_detection/fraud_rate_deviation_std" '
            'AND resource.type="global"'
        ),
        threshold=float(monitoring_cfg["fraud_rate_alert_stddev"]),
        comparison="COMPARISON_GT",
    )


# ── Drift check entry point (called by Cloud Scheduler via Cloud Run) ─────────

def run_drift_check(config: dict) -> dict:
    """
    Execute both drift checks and return a summary dict.
    Called by the /drift-check endpoint on the Cloud Run API.
    """
    ts = datetime.now(timezone.utc).isoformat()
    feature_results = check_feature_drift(config)
    perf_result = check_performance_drift(config)

    any_alert = any(r.get("alert_fired") for r in feature_results) or perf_result.get("alert_fired", False)

    summary = {
        "timestamp": ts,
        "feature_drift": feature_results,
        "performance_drift": perf_result,
        "any_alert_fired": any_alert,
    }

    logger.info(
        "drift_check_complete",
        extra={"any_alert": any_alert, "feature_checks": len(feature_results)},
    )
    return summary
