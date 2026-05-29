"""
SHAP explainability for the overall champion model (highest auc_pr across all 4 variants).

Outputs saved to GCS:
  explainability/shap_feature_importance_top20.png  — bar chart of mean |SHAP|
  explainability/shap_beeswarm.png                  — summary beeswarm plot

Top 10 features are also printed to stdout and logged to Vertex AI Experiments
run metadata for the champion run.
"""

import io
import os
import sys
from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for cloud/headless

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.logging import get_logger

logger = get_logger(__name__)


def load_model_from_gcs(gcs_uri: str, config: dict) -> Any:
    """Load a joblib-serialised model from GCS."""
    from google.cloud import storage as gcs

    path = gcs_uri.removeprefix("gs://")
    bucket_name, blob_path = path.split("/", 1)

    buf = io.BytesIO()
    client = gcs.Client(project=config["gcp"]["project_id"])
    client.bucket(bucket_name).blob(blob_path).download_to_file(buf)
    buf.seek(0)
    return joblib.load(buf)


def _save_figure_to_gcs(fig: plt.Figure, blob_path: str, config: dict) -> str:
    """Save a matplotlib figure as PNG to GCS and return the gs:// URI."""
    from google.cloud import storage as gcs

    bucket_name = config["storage"]["artifacts_bucket"]
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    buf.seek(0)

    client = gcs.Client(project=config["gcp"]["project_id"])
    client.bucket(bucket_name).blob(blob_path).upload_from_file(
        buf, content_type="image/png"
    )
    uri = f"gs://{bucket_name}/{blob_path}"
    logger.info("figure_saved_to_gcs", extra={"uri": uri})
    return uri


def compute_shap_and_save(
    model: Any,
    X_test: pd.DataFrame,
    champion_name: str,
    config: dict,
    top_n: int = 20,
) -> tuple[pd.DataFrame, str, str]:
    """
    Compute SHAP values on the test set for the champion model.

    Returns:
      feature_importance — DataFrame with columns [feature, mean_abs_shap] sorted desc
      bar_plot_uri       — GCS URI for feature importance bar chart
      beeswarm_uri       — GCS URI for beeswarm summary plot
    """
    logger.info(
        "computing_shap_values",
        extra={"model": champion_name, "test_rows": len(X_test), "features": X_test.shape[1]},
    )

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # For binary classification, shap_values may be a list [neg_class, pos_class]
    # (XGBoost returns a 2D array; LightGBM may return a list)
    if isinstance(shap_values, list):
        shap_arr = shap_values[1]  # positive class
    else:
        shap_arr = shap_values

    mean_abs_shap = np.abs(shap_arr).mean(axis=0)
    feature_importance = pd.DataFrame({
        "feature": X_test.columns.tolist(),
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    prefix = config["explainability"]["output_gcs_prefix"].rstrip("/")

    # ── Bar chart: top-N features ─────────────────────────────────────────────
    top_df = feature_importance.head(top_n)
    fig_bar, ax = plt.subplots(figsize=(10, 8))
    ax.barh(top_df["feature"][::-1], top_df["mean_abs_shap"][::-1], color="#e74c3c")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"Top {top_n} Feature Importance — {champion_name}")
    ax.tick_params(axis="y", labelsize=9)
    plt.tight_layout()
    bar_uri = _save_figure_to_gcs(fig_bar, f"{prefix}/shap_feature_importance_top{top_n}.png", config)
    plt.close(fig_bar)

    # ── Beeswarm summary plot ─────────────────────────────────────────────────
    # Use at most 5000 samples to keep render time reasonable
    n_sample = min(5000, len(X_test))
    idx = np.random.default_rng(42).choice(len(X_test), n_sample, replace=False)
    shap_sample = shap_arr[idx]
    X_sample = X_test.iloc[idx].reset_index(drop=True)

    fig_bee = plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_sample, X_sample, show=False, max_display=top_n)
    plt.title(f"SHAP Beeswarm — {champion_name}", pad=12)
    plt.tight_layout()
    beeswarm_uri = _save_figure_to_gcs(fig_bee, f"{prefix}/shap_beeswarm.png", config)
    plt.close(fig_bee)

    # ── CSV of all feature importances ───────────────────────────────────────────
    csv_buf = io.BytesIO(feature_importance.to_csv(index=False).encode())
    from google.cloud import storage as gcs_module
    bucket_name = config["storage"]["artifacts_bucket"]
    gcs_client = gcs_module.Client(project=config["gcp"]["project_id"])
    gcs_client.bucket(bucket_name).blob(f"{prefix}/shap_feature_importance.csv").upload_from_file(
        csv_buf, content_type="text/csv"
    )
    csv_uri = f"gs://{bucket_name}/{prefix}/shap_feature_importance.csv"
    logger.info("shap_csv_saved", extra={"uri": csv_uri})

    return feature_importance, bar_uri, beeswarm_uri


def print_top_features(feature_importance: pd.DataFrame, n: int = 10) -> None:
    """Print top-N features with mean absolute SHAP values to stdout."""
    print(f"\n{'─'*50}")
    print(f"Top {n} features (mean |SHAP|):")
    print(f"{'─'*50}")
    for rank, row in feature_importance.head(n).iterrows():
        print(f"  {rank+1:>2}. {row['feature']:<35}  {row['mean_abs_shap']:.6f}")
    print(f"{'─'*50}")


def log_champion_to_vertex_experiments(
    config: dict,
    champion_run_name: str,
    top_feature: str,
    top_shap_value: float,
    bar_uri: str,
    beeswarm_uri: str,
) -> None:
    """Append champion model and top feature metadata to the existing Vertex AI run."""
    import google.cloud.aiplatform as aip

    project = config["gcp"]["project_id"]
    location = config["gcp"]["region"]
    experiment_name = config["vertex"]["experiment_name"]

    aip.init(project=project, location=location)

    try:
        run = aip.ExperimentRun(
            run_name=champion_run_name,
            experiment=experiment_name,
            project=project,
            location=location,
        )
        run.log_params({
            "champion_model": champion_run_name,
            "top_feature": top_feature,
            "shap_bar_plot_uri": bar_uri,
            "shap_beeswarm_uri": beeswarm_uri,
        })
        run.log_metrics({"top_feature_mean_abs_shap": round(top_shap_value, 6)})
        logger.info(
            "champion_shap_logged_to_experiments",
            extra={"run": champion_run_name, "top_feature": top_feature},
        )
    except Exception as exc:
        logger.warning(
            "could_not_update_experiment_run",
            extra={"run": champion_run_name, "error": str(exc)},
        )


def run_shap_for_champion(
    champion_name: str,
    model: Any,
    X_test: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """
    Convenience entry point called by model_selector after champion is identified.

    Returns the full feature importance DataFrame.
    """
    top_n = int(config["explainability"]["top_n_features"])

    feature_importance, bar_uri, beeswarm_uri = compute_shap_and_save(
        model=model,
        X_test=X_test,
        champion_name=champion_name,
        config=config,
        top_n=top_n,
    )

    print_top_features(feature_importance, n=10)

    top_feature = feature_importance.iloc[0]["feature"]
    top_shap = float(feature_importance.iloc[0]["mean_abs_shap"])

    log_champion_to_vertex_experiments(
        config=config,
        champion_run_name=champion_name,
        top_feature=top_feature,
        top_shap_value=top_shap,
        bar_uri=bar_uri,
        beeswarm_uri=beeswarm_uri,
    )

    logger.info(
        "shap_complete",
        extra={"top_feature": top_feature, "top_shap": round(top_shap, 6), "bar_uri": bar_uri},
    )
    return feature_importance
