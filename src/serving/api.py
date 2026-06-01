"""
Cloud Run REST API wrapping the Vertex AI Endpoint.

Endpoints:
  POST /predict  — raw transaction JSON → fraud_probability + prediction + top-3 SHAP
  GET  /health   — model version, endpoint status, today's prediction count
  GET  /metrics  — last 24hr: total predictions, fraud rate, latency p50/p95/p99

Auth: X-API-Key header validated against Secret Manager secret.
Logging: structured JSON throughout, no PII logged.
Deploy:  Cloud Run, min_instances=1, max_instances=5, memory=2Gi, serving-sa.
"""

import io
import os
import time
import uuid as _uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Security
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field

from src.utils.config import load_config
from src.utils.logging import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="Fraud Detection API",
    description="Online fraud scoring — lgb-v8-no-txnid, Platt-calibrated",
    version="1.0.0",
)

# ── Module-level state (lazy-loaded on first request) ────────────────────────
_config: dict | None = None
_transformer: Any = None
_model: Any = None          # calibrated LGBMClassifier loaded from GCS
_shap_explainer: Any = None
_api_key: str | None = None

_API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


def _get_config() -> dict:
    global _config
    if _config is None:
        env = os.getenv("ENV", "dev")
        _config = load_config(env)
    return _config


def _get_api_key() -> str:
    """Load API key from Secret Manager once and cache it."""
    global _api_key
    if _api_key is not None:
        return _api_key

    from google.cloud import secretmanager

    cfg = _get_config()
    project = cfg["gcp"]["project_id"]
    secret_id = cfg["serving"]["api_key_secret"]

    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project}/secrets/{secret_id}/versions/latest"
    response = client.access_secret_version(request={"name": name})
    _api_key = response.payload.data.decode("utf-8").strip()
    logger.info("api_key_loaded_from_secret_manager")
    return _api_key


def _load_transformer_from_gcs() -> Any:
    global _transformer
    if _transformer is not None:
        return _transformer

    from google.cloud import storage as gcs

    cfg = _get_config()
    uri = cfg["serving"]["transformer_gcs_uri"]
    path = uri.removeprefix("gs://")
    bucket_name, blob_path = path.split("/", 1)

    buf = io.BytesIO()
    gcs.Client(project=cfg["gcp"]["project_id"]).bucket(bucket_name).blob(blob_path).download_to_file(buf)
    buf.seek(0)
    _transformer = joblib.load(buf)
    logger.info("transformer_loaded", extra={"uri": uri})
    return _transformer


def _load_model_from_gcs() -> Any:
    """Load the calibrated model from GCS for local inference and SHAP."""
    global _model
    if _model is not None:
        return _model

    from google.cloud import storage as gcs

    cfg = _get_config()
    bucket = cfg["storage"]["artifacts_bucket"]
    model_version = cfg["serving"]["model_version"]
    blob_path = f"models/{model_version}-calibrated/model.joblib"

    buf = io.BytesIO()
    gcs.Client(project=cfg["gcp"]["project_id"]).bucket(bucket).blob(blob_path).download_to_file(buf)
    buf.seek(0)
    _model = joblib.load(buf)
    logger.info("calibrated_model_loaded", extra={"blob": blob_path})
    return _model


def _get_shap_explainer() -> Any:
    """Build SHAP TreeExplainer from base LightGBM estimator (cached)."""
    global _shap_explainer
    if _shap_explainer is not None:
        return _shap_explainer

    import shap

    calibrated = _load_model_from_gcs()
    # CalibratedClassifierCV wraps the base estimator; extract it for TreeExplainer.
    base_estimator = calibrated.calibrated_classifiers_[0].estimator
    _shap_explainer = shap.TreeExplainer(base_estimator)
    logger.info("shap_explainer_ready")
    return _shap_explainer


def _validate_api_key(key: str | None) -> None:
    """Raise HTTP 401/403 if key is missing or invalid."""
    if key is None:
        raise HTTPException(status_code=401, detail="X-API-Key header required")
    expected = _get_api_key()
    if key != expected:
        raise HTTPException(status_code=403, detail="Invalid API key")


# ── Request / response models ─────────────────────────────────────────────────

class TransactionFeatures(BaseModel):
    """Raw transaction fields accepted by /predict. Mirrors BQ column names."""
    TransactionAmt: float = Field(default=68.5, description="Transaction amount in USD")
    TransactionDT: float = Field(default=86400.0, description="Seconds since reference date")
    card1: float | None = Field(default=9500.0)
    card2: float | None = Field(default=325.0)
    card3: float | None = Field(default=150.0)
    card5: float | None = Field(default=226.0)
    addr1: float | None = Field(default=299.0)
    addr2: float | None = Field(default=87.0)
    dist1: float | None = Field(default=None)
    dist2: float | None = Field(default=None)
    C1: float | None = Field(default=1.0)
    C2: float | None = Field(default=1.0)
    C3: float | None = Field(default=0.0)
    C4: float | None = Field(default=0.0)
    C5: float | None = Field(default=0.0)
    C6: float | None = Field(default=1.0)
    C7: float | None = Field(default=0.0)
    C8: float | None = Field(default=0.0)
    C9: float | None = Field(default=1.0)
    C10: float | None = Field(default=0.0)
    C11: float | None = Field(default=1.0)
    C12: float | None = Field(default=0.0)
    C13: float | None = Field(default=1.0)
    C14: float | None = Field(default=1.0)
    D1: float | None = Field(default=14.0)
    D2: float | None = Field(default=None)
    D3: float | None = Field(default=None)
    D4: float | None = Field(default=None)
    D5: float | None = Field(default=None)
    D6: float | None = Field(default=None)
    D7: float | None = Field(default=None)
    D8: float | None = Field(default=None)
    D9: float | None = Field(default=None)
    D10: float | None = Field(default=None)
    D11: float | None = Field(default=None)
    D12: float | None = Field(default=None)
    D13: float | None = Field(default=None)
    D14: float | None = Field(default=None)
    D15: float | None = Field(default=None)

    class Config:
        extra = "allow"


class SHAPFeature(BaseModel):
    feature: str
    shap_value: float
    direction: str  # "increases_fraud_risk" | "decreases_fraud_risk"


class PredictResponse(BaseModel):
    prediction_id: str
    fraud_probability: float
    prediction: int
    threshold_used: float
    latency_ms: float
    model_version: str
    explanation: list[SHAPFeature]


class HealthResponse(BaseModel):
    status: str
    model_version: str
    endpoint_status: str
    predictions_today: int


class MetricsResponse(BaseModel):
    window_hours: int
    total_predictions: int
    fraud_rate: float
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float


# ── Endpoint handlers ─────────────────────────────────────────────────────────

@app.post("/predict", response_model=PredictResponse)
async def predict(
    transaction: TransactionFeatures,
    request: Request,
    api_key: str | None = Security(_API_KEY_HEADER),
) -> PredictResponse:
    _validate_api_key(api_key)

    t0 = time.monotonic()
    cfg = _get_config()
    serving_cfg = cfg["serving"]
    threshold = float(serving_cfg["calibrated_threshold"])
    model_version = str(serving_cfg["model_version"])

    raw_dict = transaction.model_dump()
    transformer = _load_transformer_from_gcs()
    model = _load_model_from_gcs()

    raw_df = pd.DataFrame([raw_dict])

    # Ensure all columns the transformer was fitted on are present (missing → NaN).
    # This covers V1-V339, identity columns, and target-encoding columns not
    # included in the minimal API request schema.
    expected_cols = (
        list(getattr(transformer, "_v_cols_present", []))
        + list(getattr(transformer, "_passthrough_medians", pd.Series()).index)
        + list(getattr(transformer, "target_encode_cols", []))
    )
    for col in expected_cols:
        if col not in raw_df.columns:
            raw_df[col] = np.nan

    processed = transformer.transform(raw_df).select_dtypes(include=[np.number])
    feature_names = processed.columns.tolist()
    feature_arr = processed.values

    # Fraud probability from calibrated model
    fraud_prob = float(model.predict_proba(feature_arr)[0][1])
    binary_pred = int(fraud_prob >= threshold)

    # Per-prediction SHAP (top 3 features)
    shap_features: list[SHAPFeature] = []
    try:
        explainer = _get_shap_explainer()
        import shap as shap_lib
        shap_vals = explainer.shap_values(processed)
        if isinstance(shap_vals, list):
            sv = shap_vals[1][0]
        else:
            sv = shap_vals[0]
        top3_idx = np.argsort(np.abs(sv))[::-1][:3]
        for idx in top3_idx:
            direction = "increases_fraud_risk" if sv[idx] > 0 else "decreases_fraud_risk"
            shap_features.append(SHAPFeature(
                feature=feature_names[idx],
                shap_value=round(float(sv[idx]), 4),
                direction=direction,
            ))
    except Exception as exc:
        logger.warning("shap_per_prediction_failed", extra={"error": str(exc)})

    latency_ms = (time.monotonic() - t0) * 1000
    request_id = request.headers.get("X-Request-ID") or str(_uuid.uuid4())
    prediction_id = str(_uuid.uuid4())

    # Log metadata to BigQuery (no PII, no feature values)
    try:
        from google.cloud import bigquery
        project = cfg["gcp"]["project_id"]
        dataset = cfg["bigquery"]["dataset"]
        table = f"{project}.{dataset}.{cfg['serving']['prediction_log_table']}"
        ts = datetime.now(timezone.utc).isoformat()
        bigquery.Client(project=project).insert_rows_json(table, [{
            "prediction_id": prediction_id,
            "timestamp": ts,
            "fraud_probability": round(fraud_prob, 6),
            "prediction": binary_pred,
            "threshold_used": round(threshold, 6),
            "latency_ms": round(latency_ms, 2),
            "model_version": model_version,
        }])
    except Exception as exc:
        logger.warning("bq_log_suppressed", extra={"error": str(exc)})

    logger.info(
        "predict",
        extra={
            "request_id": request_id,
            "prediction_id": prediction_id,
            "fraud_probability": round(fraud_prob, 4),
            "prediction": binary_pred,
            "latency_ms": round(latency_ms, 2),
        },
    )

    return PredictResponse(
        prediction_id=prediction_id,
        fraud_probability=round(fraud_prob, 6),
        prediction=binary_pred,
        threshold_used=threshold,
        latency_ms=round(latency_ms, 2),
        model_version=model_version,
        explanation=shap_features,
    )


@app.get("/health", response_model=HealthResponse)
async def health(
    api_key: str | None = Security(_API_KEY_HEADER),
) -> HealthResponse:
    _validate_api_key(api_key)

    cfg = _get_config()
    model_version = str(cfg["serving"]["model_version"])

    # Check endpoint reachability
    endpoint_status = "unknown"
    try:
        from src.serving.vertex_endpoint import _resolve_endpoint
        ep = _resolve_endpoint(cfg)
        endpoint_status = "reachable" if ep else "unreachable"
    except Exception:
        endpoint_status = "unreachable"

    # Today's prediction count from BQ
    predictions_today = 0
    try:
        from google.cloud import bigquery
        project = cfg["gcp"]["project_id"]
        dataset = cfg["bigquery"]["dataset"]
        table = cfg["serving"]["prediction_log_table"]
        today = datetime.now(timezone.utc).date().isoformat()
        result = bigquery.Client(project=project).query(
            f"SELECT COUNT(*) AS cnt FROM `{project}.{dataset}.{table}` "
            f"WHERE DATE(timestamp) = '{today}'"
        ).result()
        for row in result:
            predictions_today = row.cnt
    except Exception as exc:
        logger.warning("health_bq_query_failed", extra={"error": str(exc)})

    return HealthResponse(
        status="healthy",
        model_version=model_version,
        endpoint_status=endpoint_status,
        predictions_today=predictions_today,
    )


@app.post("/drift-check")
async def drift_check(
    request: Request,
    api_key: str | None = Security(_API_KEY_HEADER),
) -> dict:
    """Triggered by Cloud Scheduler — runs both drift checks and returns summary."""
    _validate_api_key(api_key)
    from src.monitoring.drift_monitor import run_drift_check
    cfg = _get_config()
    summary = run_drift_check(cfg)
    logger.info("drift_check_triggered", extra={"any_alert": summary.get("any_alert_fired")})
    return summary


@app.get("/metrics", response_model=MetricsResponse)
async def metrics(
    api_key: str | None = Security(_API_KEY_HEADER),
) -> MetricsResponse:
    _validate_api_key(api_key)

    cfg = _get_config()
    project = cfg["gcp"]["project_id"]
    dataset = cfg["bigquery"]["dataset"]
    table = cfg["serving"]["prediction_log_table"]
    window_hours = 24

    try:
        from google.cloud import bigquery

        cutoff = (datetime.now(timezone.utc) - timedelta(hours=window_hours)).isoformat()
        query = f"""
            SELECT
              COUNT(*) AS total,
              AVG(prediction)     AS fraud_rate,
              AVG(latency_ms)     AS avg_latency,
              APPROX_QUANTILES(latency_ms, 100)[OFFSET(50)]  AS p50,
              APPROX_QUANTILES(latency_ms, 100)[OFFSET(95)]  AS p95,
              APPROX_QUANTILES(latency_ms, 100)[OFFSET(99)]  AS p99
            FROM `{project}.{dataset}.{table}`
            WHERE timestamp >= '{cutoff}'
        """
        rows = list(bigquery.Client(project=project).query(query).result())
        row = rows[0] if rows else None

        return MetricsResponse(
            window_hours=window_hours,
            total_predictions=int(row.total) if row and row.total else 0,
            fraud_rate=round(float(row.fraud_rate or 0), 4),
            avg_latency_ms=round(float(row.avg_latency or 0), 2),
            p50_latency_ms=round(float(row.p50 or 0), 2),
            p95_latency_ms=round(float(row.p95 or 0), 2),
            p99_latency_ms=round(float(row.p99 or 0), 2),
        )
    except Exception as exc:
        logger.warning("metrics_bq_query_failed", extra={"error": str(exc)})
        return MetricsResponse(
            window_hours=window_hours,
            total_predictions=0,
            fraud_rate=0.0,
            avg_latency_ms=0.0,
            p50_latency_ms=0.0,
            p95_latency_ms=0.0,
            p99_latency_ms=0.0,
        )
