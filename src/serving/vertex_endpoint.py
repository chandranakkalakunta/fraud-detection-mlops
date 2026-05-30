"""
Vertex AI Endpoint management and online prediction wrapper for lgb-v8-no-txnid.

Responsibilities:
- Deploy champion model from Vertex AI Model Registry to a Vertex AI Endpoint
- Load feature engineer transformer from GCS (module-level cache — one load per process)
- Preprocess raw transaction dicts through FraudFeatureEngineer.transform()
- Call Vertex AI Endpoint for inference; extract calibrated fraud probability
- Log prediction metadata (NO PII, NO feature values) to BigQuery prediction_logs
- Return fraud_probability + binary prediction at the calibrated threshold

BigQuery table schema (fraud_detection.prediction_logs):
  prediction_id STRING, timestamp TIMESTAMP, fraud_probability FLOAT64,
  prediction INT64, threshold_used FLOAT64, latency_ms FLOAT64,
  model_version STRING
"""

import io
import time
import uuid
from datetime import datetime, timezone
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)

# Module-level caches — populated lazily on first predict() call.
_TRANSFORMER_CACHE: dict[str, Any] = {}
_ENDPOINT_CACHE: dict[str, Any] = {}


# ── Internal helpers ──────────────────────────────────────────────────────────

def _load_transformer(uri: str, project_id: str) -> Any:
    if uri in _TRANSFORMER_CACHE:
        return _TRANSFORMER_CACHE[uri]

    from google.cloud import storage as gcs

    path = uri.removeprefix("gs://")
    bucket_name, blob_path = path.split("/", 1)
    buf = io.BytesIO()
    gcs.Client(project=project_id).bucket(bucket_name).blob(blob_path).download_to_file(buf)
    buf.seek(0)
    transformer = joblib.load(buf)
    _TRANSFORMER_CACHE[uri] = transformer
    logger.info("transformer_loaded", extra={"uri": uri})
    return transformer


def _resolve_endpoint(config: dict) -> Any:
    """Return a cached Vertex AI Endpoint, finding or creating it as needed."""
    import google.cloud.aiplatform as aip

    serving_cfg = config["serving"]
    cache_key = serving_cfg["endpoint_display_name"]
    if cache_key in _ENDPOINT_CACHE:
        return _ENDPOINT_CACHE[cache_key]

    project = config["gcp"]["project_id"]
    location = config["gcp"]["region"]
    aip.init(project=project, location=location, staging_bucket=config["vertex"]["staging_bucket"])

    resource_name = str(serving_cfg.get("endpoint_resource_name", "")).strip()
    if resource_name:
        endpoint = aip.Endpoint(resource_name)
        logger.info("endpoint_resolved_by_resource_name", extra={"resource_name": resource_name})
    else:
        existing = aip.Endpoint.list(
            filter=f'display_name="{serving_cfg["endpoint_display_name"]}"',
            project=project,
            location=location,
        )
        if existing:
            endpoint = existing[0]
            logger.info("endpoint_found_by_name", extra={"resource_name": endpoint.resource_name})
        else:
            endpoint = aip.Endpoint.create(
                display_name=serving_cfg["endpoint_display_name"],
                project=project,
                location=location,
            )
            logger.info("endpoint_created", extra={"resource_name": endpoint.resource_name})

    _ENDPOINT_CACHE[cache_key] = endpoint
    return endpoint


def _log_to_bq(
    prediction_id: str,
    timestamp: datetime,
    fraud_probability: float,
    prediction: int,
    threshold: float,
    latency_ms: float,
    model_version: str,
    config: dict,
) -> None:
    """Non-blocking BQ insert — caller catches exceptions so a BQ failure never breaks predict()."""
    from google.cloud import bigquery

    project = config["gcp"]["project_id"]
    dataset = config["bigquery"]["dataset"]
    table = config["serving"]["prediction_log_table"]
    table_ref = f"{project}.{dataset}.{table}"

    row = {
        "prediction_id": prediction_id,
        "timestamp": timestamp.isoformat(),
        "fraud_probability": round(fraud_probability, 6),
        "prediction": prediction,
        "threshold_used": round(threshold, 6),
        "latency_ms": round(latency_ms, 2),
        "model_version": model_version,
    }
    errors = bigquery.Client(project=project).insert_rows_json(table_ref, [row])
    if errors:
        logger.warning("bq_insert_error", extra={"errors": str(errors[:1])})


# ── Public API ────────────────────────────────────────────────────────────────

def deploy_model(config: dict) -> str:
    """
    Deploy lgb-v8-no-txnid from Model Registry to a Vertex AI Endpoint.

    Idempotent — if the model is already deployed to the endpoint the call
    returns immediately without re-deploying.

    Returns:
        Endpoint resource name (str).
    """
    import google.cloud.aiplatform as aip

    serving_cfg = config["serving"]
    project = config["gcp"]["project_id"]
    location = config["gcp"]["region"]
    aip.init(project=project, location=location, staging_bucket=config["vertex"]["staging_bucket"])

    endpoint = _resolve_endpoint(config)
    model_resource = str(serving_cfg["model_resource_name"])
    model_root = model_resource.split("@")[0]

    # Check for an existing deployment — avoid duplicate deployments.
    for dm in endpoint.list_models():
        if dm.model.startswith(model_root):
            logger.info(
                "model_already_deployed",
                extra={"model": dm.model, "endpoint": endpoint.resource_name},
            )
            return endpoint.resource_name

    model = aip.Model(model_resource)
    model.deploy(
        endpoint=endpoint,
        deployed_model_display_name=serving_cfg["model_version"],
        machine_type=serving_cfg["machine_type"],
        min_replica_count=int(serving_cfg["min_replica_count"]),
        max_replica_count=int(serving_cfg["max_replica_count"]),
        service_account=config["service_accounts"]["serving"],
        sync=True,
    )

    logger.info(
        "model_deployed",
        extra={
            "model": model_resource,
            "endpoint": endpoint.resource_name,
            "machine_type": serving_cfg["machine_type"],
            "min_replicas": serving_cfg["min_replica_count"],
            "max_replicas": serving_cfg["max_replica_count"],
        },
    )
    return endpoint.resource_name


def predict(
    instances: list[dict],
    config: dict,
    log_to_bq: bool = True,
) -> list[dict]:
    """
    End-to-end online prediction pipeline.

    Flow: raw dict → FraudFeatureEngineer.transform() → Vertex AI Endpoint
          → calibrated threshold → BigQuery log (metadata only, no PII).

    Args:
        instances:  List of raw transaction dicts (unprocessed BQ row format).
        config:     Loaded dev.yaml config (from src.utils.config.load_config).
        log_to_bq:  Write prediction metadata to BigQuery prediction_logs.

    Returns:
        List of result dicts, one per instance:
          prediction_id, fraud_probability, prediction, threshold_used,
          latency_ms, model_version
    """
    serving_cfg = config["serving"]
    threshold = float(serving_cfg["calibrated_threshold"])
    model_version = str(serving_cfg["model_version"])
    project = config["gcp"]["project_id"]

    transformer = _load_transformer(str(serving_cfg["transformer_gcs_uri"]), project)
    endpoint = _resolve_endpoint(config)

    results: list[dict] = []

    for instance in instances:
        t0 = time.monotonic()
        prediction_id = str(uuid.uuid4())
        ts = datetime.now(timezone.utc)

        raw_df = pd.DataFrame([instance])
        processed = transformer.transform(raw_df).select_dtypes(include=[np.number])
        feature_list = processed.values[0].tolist()

        # sklearn pre-built container returns predict_proba when requested.
        response = endpoint.predict(
            instances=[feature_list],
            parameters={"predict_method": "predict_proba"},
        )
        # predict_proba returns [[prob_class_0, prob_class_1]] per instance.
        raw_pred = response.predictions[0]
        fraud_prob = float(raw_pred[1]) if isinstance(raw_pred, (list, tuple)) else float(raw_pred)

        latency_ms = (time.monotonic() - t0) * 1000
        binary_pred = int(fraud_prob >= threshold)

        result = {
            "prediction_id": prediction_id,
            "fraud_probability": round(fraud_prob, 6),
            "prediction": binary_pred,
            "threshold_used": threshold,
            "latency_ms": round(latency_ms, 2),
            "model_version": model_version,
        }

        if log_to_bq:
            try:
                _log_to_bq(
                    prediction_id=prediction_id,
                    timestamp=ts,
                    fraud_probability=fraud_prob,
                    prediction=binary_pred,
                    threshold=threshold,
                    latency_ms=latency_ms,
                    model_version=model_version,
                    config=config,
                )
            except Exception as exc:
                logger.warning("bq_log_suppressed", extra={"error": str(exc)})

        logger.info(
            "prediction_complete",
            extra={
                "prediction_id": prediction_id,
                "fraud_probability": round(fraud_prob, 4),
                "prediction": binary_pred,
                "latency_ms": round(latency_ms, 2),
                "model_version": model_version,
            },
        )
        results.append(result)

    return results
