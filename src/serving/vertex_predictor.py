"""
Minimal Vertex AI custom serving container for lgb-v8-no-txnid.

Vertex AI prediction protocol:
  POST /predict  {"instances": [[f1, f2, ...], ...]}  →  {"predictions": [p1, p2, ...]}
  GET  /health   →  {"status": "healthy"}

Model loaded from AIP_STORAGE_URI (env var injected by Vertex AI at deploy time).
Vertex AI injects this as the GCS path to the model artifacts.
"""

import io
import os

import joblib
import numpy as np
from flask import Flask, jsonify, request

app = Flask(__name__)
_MODEL = None


def _load_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    from google.cloud import storage

    # AIP_STORAGE_URI: gs://bucket/path/to/model/dir  (Vertex AI injects this)
    gcs_uri = os.environ.get("AIP_STORAGE_URI", "").rstrip("/")
    if not gcs_uri:
        raise RuntimeError("AIP_STORAGE_URI not set — must be run as a Vertex AI custom container")

    path = gcs_uri.removeprefix("gs://")
    bucket_name, blob_prefix = path.split("/", 1)

    client = storage.Client()
    buf = io.BytesIO()
    client.bucket(bucket_name).blob(f"{blob_prefix}/model.joblib").download_to_file(buf)
    buf.seek(0)
    _MODEL = joblib.load(buf)
    return _MODEL


@app.route("/health", methods=["GET"])
def health():
    # Lazy-preload model so the first /predict is fast.
    try:
        _load_model()
        return jsonify({"status": "healthy"})
    except Exception as exc:
        return jsonify({"status": "unhealthy", "error": str(exc)}), 500


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    instances = data.get("instances", [])

    if not instances:
        return jsonify({"error": "no instances provided"}), 400

    X = np.array(instances, dtype=np.float64)
    model = _load_model()
    probs = model.predict_proba(X)[:, 1].tolist()
    return jsonify({"predictions": probs})


if __name__ == "__main__":
    port = int(os.environ.get("AIP_HTTP_PORT", 8080))
    app.run(host="0.0.0.0", port=port)
