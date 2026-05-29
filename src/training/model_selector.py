"""
Model selection and Vertex AI Model Registry registration — Phase 2B.

Programmatically compares all 4 Vertex AI Experiment runs (A, B, C, D),
selects the champion by auc_pr, then registers it in Vertex AI Model Registry
with full metadata and sets stage to 'staging'.
"""

import io
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.logging import get_logger

logger = get_logger(__name__)

# Canonical run names — must match names used in xgboost_trainer and lightgbm_trainer
RUN_NAMES = [
    "xgb-variant-a-smote",
    "xgb-variant-b-class-weight",
    "lgb-variant-c-smote",
    "lgb-variant-d-is-unbalance",
]


def fetch_experiment_runs(config: dict) -> list[dict]:
    """
    Fetch all experiment run metrics from Vertex AI Experiments.

    Returns a list of dicts, each containing 'run_name', 'auc_pr', and all
    logged metrics for that run.
    """
    import google.cloud.aiplatform as aip

    project = config["gcp"]["project_id"]
    location = config["gcp"]["region"]
    experiment_name = config["vertex"]["experiment_name"]

    aip.init(project=project, location=location, experiment=experiment_name)

    runs = []
    for run_name in RUN_NAMES:
        try:
            run = aip.ExperimentRun(
                run_name=run_name,
                experiment=experiment_name,
                project=project,
                location=location,
            )
            metrics = run.get_metrics()
            params = run.get_params()
            runs.append({
                "run_name": run_name,
                **metrics,
                **{f"param_{k}": v for k, v in params.items()},
            })
            logger.info(
                "run_fetched",
                extra={"run_name": run_name, "auc_pr": metrics.get("auc_pr")},
            )
        except Exception as exc:
            logger.warning(
                "run_fetch_failed",
                extra={"run_name": run_name, "error": str(exc)},
            )

    return runs


def select_champion(runs: list[dict]) -> dict:
    """Return the run dict with the highest auc_pr."""
    if not runs:
        raise ValueError("No experiment runs available for champion selection.")

    champion = max(runs, key=lambda r: float(r.get("auc_pr", 0.0)))
    logger.info(
        "champion_selected",
        extra={"run_name": champion["run_name"], "auc_pr": champion.get("auc_pr")},
    )
    return champion


def print_comparison_table(runs: list[dict]) -> None:
    """Print all 4 variants side-by-side for quick comparison."""
    metric_keys = ["auc_pr", "auc_roc", "f1", "f2", "precision", "recall", "ks_statistic",
                   "threshold", "training_rows", "test_rows", "imbalance_strategy"]

    col_w = 30
    print(f"\n{'─'*120}")
    print("VERTEX AI EXPERIMENTS — COMPARISON TABLE (all 4 variants)")
    print(f"{'─'*120}")

    header = f"{'Metric':<25}" + "".join(f"{r['run_name']:<{col_w}}" for r in runs)
    print(header)
    print(f"{'─'*120}")

    for key in metric_keys:
        row = f"{key:<25}"
        for r in runs:
            val = r.get(key, "N/A")
            if isinstance(val, float):
                row += f"{val:<{col_w}.4f}"
            else:
                row += f"{str(val):<{col_w}}"
        print(row)

    print(f"{'─'*120}")
    champion = max(runs, key=lambda r: float(r.get("auc_pr", 0.0)))
    print(f"Champion: {champion['run_name']}  (AUC-PR: {champion.get('auc_pr', 0.0):.4f})")
    print(f"{'─'*120}\n")


def _get_model_dir_uri(champion_name: str, config: dict) -> str:
    """Return the GCS directory URI for use as Vertex AI Model Registry artifact_uri."""
    bucket = config["storage"]["artifacts_bucket"]
    return f"gs://{bucket}/models/{champion_name}/"


def _get_model_blob_uri(champion_name: str, config: dict) -> str:
    """Return the GCS file URI for downloading the model joblib."""
    bucket = config["storage"]["artifacts_bucket"]
    return f"gs://{bucket}/models/{champion_name}/model.joblib"


def register_champion_in_model_registry(
    champion_run: dict,
    feature_engineer_gcs_uri: str,
    config: dict,
) -> Any:
    """
    Register the champion model in Vertex AI Model Registry.

    Sets model stage metadata to 'staging'.
    Returns the registered Model resource.
    """
    import google.cloud.aiplatform as aip
    from datetime import datetime, timezone

    project = config["gcp"]["project_id"]
    location = config["gcp"]["region"]
    model_display_name = config["vertex"]["model_display_name"]
    champion_name = champion_run["run_name"]

    aip.init(project=project, location=location)

    model_gcs_uri = _get_model_dir_uri(champion_name, config)

    # Determine serving container based on model type
    if "xgb" in champion_name:
        serving_container = config["vertex"]["serving_container"]["xgboost"]
    else:
        serving_container = config["vertex"]["serving_container"]["lightgbm"]

    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%S")

    model = aip.Model.upload(
        display_name=model_display_name,
        artifact_uri=model_gcs_uri,
        serving_container_image_uri=serving_container,
        description=(
            f"Fraud detection champion — {champion_name}. "
            f"AUC-PR: {champion_run.get('auc_pr', 0):.4f}. "
            f"Registered {timestamp}."
        ),
        labels={
            "stage": "staging",
            "champion_run": champion_name.replace("/", "-"),
            "imbalance_strategy": str(champion_run.get("imbalance_strategy", "unknown")),
        },
    )

    logger.info(
        "model_registered",
        extra={
            "model_resource_name": model.resource_name,
            "display_name": model_display_name,
            "champion_run": champion_name,
            "auc_pr": champion_run.get("auc_pr"),
        },
    )
    return model


def main() -> None:
    from dotenv import load_dotenv
    load_dotenv()

    env = os.getenv("ENV", "dev")
    config = load_config(env)

    # ── Fetch all 4 experiment runs ────────────────────────────────────────────
    runs = fetch_experiment_runs(config)
    if not runs:
        raise RuntimeError("No experiment runs found — run the trainers first.")

    # ── Print comparison table ─────────────────────────────────────────────────
    print_comparison_table(runs)

    # ── Select champion ────────────────────────────────────────────────────────
    champion = select_champion(runs)
    champion_name = champion["run_name"]

    # ── Compute SHAP on champion ───────────────────────────────────────────────
    import joblib
    from google.cloud import storage as gcs
    from src.features.engineer import build_from_config
    from src.training.xgboost_trainer import load_engineered_features, time_based_split, TARGET, SORT_COL
    from src.explainability.shap_explainer import run_shap_for_champion

    model_blob_uri = _get_model_blob_uri(champion_name, config)
    model_buf = io.BytesIO()
    gcs_client = gcs.Client(project=config["gcp"]["project_id"])
    path_no_prefix = model_blob_uri.removeprefix("gs://")
    bucket_name, blob_path = path_no_prefix.split("/", 1)
    gcs_client.bucket(bucket_name).blob(blob_path).download_to_file(model_buf)
    model_buf.seek(0)
    champion_model = joblib.load(model_buf)

    # Reconstruct test set with feature engineering for SHAP
    df = load_engineered_features(config)
    val_frac = float(config["model"]["val_split"])
    _, _, test_df = time_based_split(df, train_frac=0.75, val_frac=val_frac)

    feature_cols = [c for c in df.columns if c not in [TARGET, SORT_COL]]
    X_test_raw = test_df[feature_cols].reset_index(drop=True)
    y_test = test_df[TARGET].astype(int).reset_index(drop=True)

    # Apply feature engineering using fresh engineer fitted on training portion
    train_df_all = df[df.index < len(df) * 0.75].reset_index(drop=True) if SORT_COL not in df.columns else None
    # Use the same split boundaries
    n = len(df.sort_values(SORT_COL))
    train_end = int(n * 0.75)
    val_start = int(train_end * (1 - val_frac))
    df_sorted = df.sort_values(SORT_COL).reset_index(drop=True)
    X_train_fe = df_sorted.iloc[:val_start][feature_cols].reset_index(drop=True)
    y_train_fe = df_sorted[TARGET].astype(int).iloc[:val_start].reset_index(drop=True)

    engineer = build_from_config(config)
    engineer.fit(X_train_fe, y_train_fe)
    X_test_eng = engineer.transform(X_test_raw)

    feature_importance = run_shap_for_champion(
        champion_name=champion_name,
        model=champion_model,
        X_test=X_test_eng,
        config=config,
    )

    # ── Register in Model Registry ────────────────────────────────────────────
    feature_engineer_gcs_uri = (
        f"gs://{config['storage']['artifacts_bucket']}"
        f"/{config['features']['transformer_gcs_prefix'].rstrip('/')}/feature_engineer.joblib"
    )

    registered_model = register_champion_in_model_registry(
        champion_run=champion,
        feature_engineer_gcs_uri=feature_engineer_gcs_uri,
        config=config,
    )

    # ── Print final summary ────────────────────────────────────────────────────
    top_feature = feature_importance.iloc[0]["feature"]
    top_shap = float(feature_importance.iloc[0]["mean_abs_shap"])

    print(f"\n{'═'*60}")
    print("PHASE 2B COMPLETE")
    print(f"{'═'*60}")
    print(f"Champion model       : {champion_name}")
    print(f"AUC-PR               : {champion.get('auc_pr', 0.0):.4f}")
    print(f"AUC-ROC              : {champion.get('auc_roc', 0.0):.4f}")
    print(f"GCS URI              : {_get_model_dir_uri(champion_name, config)}")
    print(f"Model Registry name  : {registered_model.display_name}")
    print(f"Model Registry ID    : {registered_model.resource_name}")
    print(f"Stage                : staging")
    print(f"Top SHAP feature     : {top_feature}  ({top_shap:.6f})")
    print(f"{'═'*60}\n")

    return {
        "champion_name": champion_name,
        "champion_metrics": champion,
        "model_registry_resource": registered_model.resource_name,
        "feature_importance": feature_importance,
    }


if __name__ == "__main__":
    # Import here to avoid circular imports at module level
    from src.utils.config import load_config
    main()
