"""
v7 champion pipeline — HP-tuned models, registry, SHAP, calibration, report.

Steps (in order):
  1. Load best_params from config/best_params_xgb.yaml and config/best_params_lgb.yaml
  2. Train xgb-v7-tuned and lgb-v7-tuned using the 388-feature v5 feature set
  3. Select champion (higher AUC-PR on test set)
  4. Register champion in Vertex AI Model Registry (stage=production-ready)
  5. Compute SHAP on test set; save bar + beeswarm to GCS; print top 10 features
  6. Platt scaling calibration (CalibratedClassifierCV method='sigmoid') fit on val set
  7. Reliability diagram to GCS
  8. Log calibrated model to Vertex AI Experiments
  9. Save calibrated model to GCS (serving artifact)
 10. Write evaluation/final_evaluation_report.json

Pre-requisites:
  config/best_params_xgb.yaml  (written by hyperparameter_tuner.py --model xgboost)
  config/best_params_lgb.yaml  (written by hyperparameter_tuner.py --model lightgbm)
"""

import io
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.explainability.shap_explainer import compute_shap_and_save, print_top_features
from src.training.metrics import compute_fraud_metrics
from src.training.train_utils import (
    apply_engineer,
    build_no_te_engineer,
    prepare_data_splits,
    train_lgb_from_params,
    train_xgb_from_params,
)
from src.training.xgboost_trainer import (
    load_engineered_features,
    log_to_vertex_experiments,
    save_model_to_gcs,
)
from src.utils.config import load_config
from src.utils.logging import get_logger

logger = get_logger(__name__)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _load_best_params(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Best params file not found: {p}\n"
            "Run hyperparameter_tuner.py for both models first:\n"
            "  python -m src.training.hyperparameter_tuner --model xgboost\n"
            "  python -m src.training.hyperparameter_tuner --model lightgbm"
        )
    with open(p) as f:
        params = yaml.safe_load(f)
    logger.info("best_params_loaded", extra={"path": str(p), "params": params})
    return params


def _register_champion(
    model_name: str,
    auc_pr: float,
    auc_roc: float,
    config: dict,
) -> Any:
    """Register the (uncalibrated) champion model in Vertex AI Model Registry."""
    import google.cloud.aiplatform as aip

    project = config["gcp"]["project_id"]
    location = config["gcp"]["region"]
    model_display_name = config["vertex"]["model_display_name"]

    aip.init(project=project, location=location)

    model_dir_uri = f"gs://{config['storage']['artifacts_bucket']}/models/{model_name}/"
    serving_container = (
        config["vertex"]["serving_container"]["xgboost"]
        if "xgb" in model_name
        else config["vertex"]["serving_container"]["lightgbm"]
    )
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%S")

    registered = aip.Model.upload(
        display_name=model_display_name,
        artifact_uri=model_dir_uri,
        serving_container_image_uri=serving_container,
        description=(
            f"Fraud detection champion (HP-tuned) — {model_name}. "
            f"AUC-PR: {auc_pr:.4f}, AUC-ROC: {auc_roc:.4f}. "
            f"Registered {timestamp}."
        ),
        labels={
            "stage": "production-ready",
            "champion_run": model_name.replace("/", "-"),
            "phase": "v7-hp-tuned",
        },
    )

    logger.info(
        "champion_registered",
        extra={
            "resource_name": registered.resource_name,
            "model_name": model_name,
            "auc_pr": round(auc_pr, 4),
        },
    )
    return registered


def _reliability_diagram_to_gcs(
    y_true: np.ndarray,
    y_proba_uncal: np.ndarray,
    y_proba_cal: np.ndarray,
    champion_name: str,
    config: dict,
) -> str:
    """Plot reliability diagram (before + after calibration) and save to GCS."""
    from sklearn.calibration import calibration_curve
    from google.cloud import storage as gcs

    fop_uncal, mpv_uncal = calibration_curve(y_true, y_proba_uncal, n_bins=10)
    fop_cal, mpv_cal = calibration_curve(y_true, y_proba_cal, n_bins=10)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.plot(mpv_uncal, fop_uncal, "o-", color="#e74c3c", label=f"Uncalibrated — {champion_name}")
    ax.plot(mpv_cal, fop_cal, "s-", color="#2ecc71", label="Platt-calibrated")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives (fraud rate in bin)")
    ax.set_title(f"Reliability Diagram — {champion_name}")
    ax.legend(loc="upper left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    buf.seek(0)
    plt.close(fig)

    bucket_name = config["storage"]["artifacts_bucket"]
    prefix = config["explainability"]["output_gcs_prefix"].rstrip("/")
    blob_path = f"{prefix}/reliability_diagram.png"

    client = gcs.Client(project=config["gcp"]["project_id"])
    client.bucket(bucket_name).blob(blob_path).upload_from_file(
        buf, content_type="image/png"
    )
    uri = f"gs://{bucket_name}/{blob_path}"
    logger.info("reliability_diagram_saved", extra={"uri": uri})
    return uri


# ── Main pipeline ──────────────────────────────────────────────────────────────

def main() -> dict:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=".env")

    env = os.getenv("ENV", "dev")
    config = load_config(env)
    vizier_cfg = config["tuning"]["vizier"]

    # ── Load best params ───────────────────────────────────────────────────────
    xgb_best_params = _load_best_params(vizier_cfg["best_params_xgb_path"])
    lgb_best_params = _load_best_params(vizier_cfg["best_params_lgb_path"])

    print(f"\nBest XGBoost params: {xgb_best_params}")
    print(f"Best LightGBM params: {lgb_best_params}")

    # ── Load data and prepare splits ───────────────────────────────────────────
    df = load_engineered_features(config)
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_data_splits(df, config)

    print(f"\nSplit summary (v7):")
    print(f"  Train : {len(X_train):>7,}  ({y_train.mean():.3%} fraud)")
    print(f"  Val   : {len(X_val):>7,}  ({y_val.mean():.3%} fraud)  stratified")
    print(f"  Test  : {len(X_test):>7,}  ({y_test.mean():.3%} fraud)  temporal")

    # ── Feature engineering (once, shared) ────────────────────────────────────
    engineer = build_no_te_engineer(config)
    X_train_eng, X_val_eng, X_test_eng = apply_engineer(
        engineer, X_train, y_train, X_val, X_test
    )
    n_features = X_train_eng.shape[1]
    print(f"\nFeatures: {n_features}  (no TE, no indicators)")

    # ── Train v7 models ────────────────────────────────────────────────────────
    print("\nTraining xgb-v7-tuned...")
    model_xgb = train_xgb_from_params(
        X_train_eng, y_train, X_val_eng, y_val, xgb_best_params, config
    )
    y_proba_xgb = model_xgb.predict_proba(X_test_eng)[:, 1]
    metrics_xgb = compute_fraud_metrics(
        y_true=y_test.values,
        y_proba=y_proba_xgb,
        imbalance_strategy="scale_pos_weight",
        training_rows=len(X_train),
        test_rows=len(X_test),
    )
    log_to_vertex_experiments(
        config,
        "xgb-v7-tuned",
        {
            "model_type": "xgboost",
            "n_features": n_features,
            "best_iteration": model_xgb.best_iteration,
            "phase": "v7-hp-tuned",
            **xgb_best_params,
        },
        metrics_xgb,
    )
    save_model_to_gcs(model_xgb, "xgb-v7-tuned", config)

    print("\nTraining lgb-v7-tuned...")
    model_lgb = train_lgb_from_params(
        X_train_eng, y_train, X_val_eng, y_val, lgb_best_params, config
    )
    y_proba_lgb = model_lgb.predict_proba(X_test_eng)[:, 1]
    metrics_lgb = compute_fraud_metrics(
        y_true=y_test.values,
        y_proba=y_proba_lgb,
        imbalance_strategy="lgb_is_unbalance",
        training_rows=len(X_train),
        test_rows=len(X_test),
    )
    log_to_vertex_experiments(
        config,
        "lgb-v7-tuned",
        {
            "model_type": "lightgbm",
            "n_features": n_features,
            "best_iteration": model_lgb.best_iteration_,
            "phase": "v7-hp-tuned",
            **lgb_best_params,
        },
        metrics_lgb,
    )
    save_model_to_gcs(model_lgb, "lgb-v7-tuned", config)

    # ── Select champion ────────────────────────────────────────────────────────
    print(f"\nxgb-v7-tuned  AUC-PR={metrics_xgb['auc_pr']:.4f}  best_iter={model_xgb.best_iteration}")
    print(f"lgb-v7-tuned  AUC-PR={metrics_lgb['auc_pr']:.4f}  best_iter={model_lgb.best_iteration_}")

    if metrics_xgb["auc_pr"] >= metrics_lgb["auc_pr"]:
        champion_name = "xgb-v7-tuned"
        champion_model = model_xgb
        champion_metrics = metrics_xgb
        champion_params = xgb_best_params
        y_proba_champion = y_proba_xgb
        champion_best_iter = model_xgb.best_iteration
    else:
        champion_name = "lgb-v7-tuned"
        champion_model = model_lgb
        champion_metrics = metrics_lgb
        champion_params = lgb_best_params
        y_proba_champion = y_proba_lgb
        champion_best_iter = model_lgb.best_iteration_

    print(f"\nChampion: {champion_name}  (AUC-PR={champion_metrics['auc_pr']:.4f})")

    # ── Register in Model Registry ─────────────────────────────────────────────
    registered_model = _register_champion(
        model_name=champion_name,
        auc_pr=champion_metrics["auc_pr"],
        auc_roc=champion_metrics["auc_roc"],
        config=config,
    )

    # ── SHAP on test set ───────────────────────────────────────────────────────
    print("\nComputing SHAP values on test set...")
    top_n = int(config["explainability"]["top_n_features"])
    feature_importance, bar_uri, beeswarm_uri = compute_shap_and_save(
        model=champion_model,
        X_test=X_test_eng,
        champion_name=champion_name,
        config=config,
        top_n=top_n,
    )
    print_top_features(feature_importance, n=10)

    # ── Platt calibration — fit on val, evaluate on test ──────────────────────
    print("\nCalibrating with Platt scaling (val set)...")
    from sklearn.calibration import CalibratedClassifierCV

    calibrated_model = CalibratedClassifierCV(champion_model, cv="prefit", method="sigmoid")
    calibrated_model.fit(X_val_eng, y_val)

    y_proba_cal = calibrated_model.predict_proba(X_test_eng)[:, 1]
    metrics_calibrated = compute_fraud_metrics(
        y_true=y_test.values,
        y_proba=y_proba_cal,
        imbalance_strategy=(
            "scale_pos_weight" if "xgb" in champion_name else "lgb_is_unbalance"
        ),
        training_rows=len(X_train),
        test_rows=len(X_test),
    )

    calibrated_run_name = f"{champion_name}-calibrated"
    log_to_vertex_experiments(
        config,
        calibrated_run_name,
        {
            "model_type": champion_name.split("-")[0],
            "calibration": "platt_sigmoid",
            "calibration_set": "val",
            "n_features": n_features,
            "base_model": champion_name,
            **champion_params,
        },
        metrics_calibrated,
    )

    # ── Reliability diagram ────────────────────────────────────────────────────
    reliability_uri = _reliability_diagram_to_gcs(
        y_true=y_test.values,
        y_proba_uncal=y_proba_champion,
        y_proba_cal=y_proba_cal,
        champion_name=champion_name,
        config=config,
    )

    # ── Save calibrated model to GCS ───────────────────────────────────────────
    calibrated_gcs_uri = save_model_to_gcs(calibrated_model, calibrated_run_name, config)
    print(f"Calibrated model saved: {calibrated_gcs_uri}")

    # ── Final evaluation report ────────────────────────────────────────────────
    report = {
        "champion": {
            "name": champion_name,
            "version": registered_model.version_id if hasattr(registered_model, "version_id") else "1",
            "model_registry_resource": registered_model.resource_name,
            "stage": "production-ready",
            "gcs_base_uri": f"gs://{config['storage']['artifacts_bucket']}/models/{champion_name}/",
            "calibrated_gcs_uri": calibrated_gcs_uri,
        },
        "feature_engineering": {
            "n_features": n_features,
            "target_encoding": False,
            "missing_indicators": False,
            "features_included": [
                "V1-V339 (median imputed)",
                "C1-C14",
                "D1-D15",
                "card1/2/3/5, addr1/addr2, dist1/dist2",
                "TransactionAmt",
                "time_of_day, day_of_week, hour_of_day",
                "log_transaction_amt, amt_to_d1_ratio, is_round_amt",
                "D1_card_norm, D4_card_norm, D10_card_norm",
            ],
        },
        "data_splits": {
            "train_rows": len(X_train),
            "val_rows": len(X_val),
            "test_rows": len(X_test),
            "train_fraud_rate": round(float(y_train.mean()), 4),
            "val_fraud_rate": round(float(y_val.mean()), 4),
            "test_fraud_rate": round(float(y_test.mean()), 4),
            "split_strategy": "time_based_75_25_temporal_test_stratified_val",
        },
        "v7_comparison": {
            "xgb_v7": {
                **metrics_xgb,
                "best_iteration": model_xgb.best_iteration,
            },
            "lgb_v7": {
                **metrics_lgb,
                "best_iteration": model_lgb.best_iteration_,
            },
        },
        "champion_metrics_uncalibrated": {
            **champion_metrics,
            "best_iteration": champion_best_iter,
            "threshold": champion_metrics["threshold"],
            "business_interpretation": (
                f"At threshold {champion_metrics['threshold']:.4f}: "
                f"Precision={champion_metrics['precision']:.3f} "
                f"(of flagged transactions, {champion_metrics['precision']*100:.1f}% are fraud), "
                f"Recall={champion_metrics['recall']:.3f} "
                f"({champion_metrics['recall']*100:.1f}% of fraud transactions caught)."
            ),
        },
        "champion_metrics_calibrated": {
            **metrics_calibrated,
            "calibration_method": "platt_sigmoid",
            "calibration_set": "validation",
            "auc_pr_delta_vs_uncalibrated": round(
                metrics_calibrated["auc_pr"] - champion_metrics["auc_pr"], 4
            ),
        },
        "best_hyperparameters": {
            "xgboost": xgb_best_params,
            "lightgbm": lgb_best_params,
        },
        "shap_top_10": feature_importance.head(10)[["feature", "mean_abs_shap"]].to_dict("records"),
        "explainability_gcs": {
            "shap_bar_plot": bar_uri,
            "shap_beeswarm": beeswarm_uri,
            "reliability_diagram": reliability_uri,
        },
        "baselines": {
            "lr_baseline_auc_pr": 0.2172,
            "xgb_v5_auc_pr": 0.4895,
            "lgb_v5_auc_pr": 0.4913,
        },
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
    }

    out_path = Path("evaluation/final_evaluation_report.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=str))
    logger.info("final_report_saved", extra={"path": str(out_path)})

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n{'═'*80}")
    print("CHAMPION PIPELINE COMPLETE")
    print(f"{'═'*80}")
    print(f"Champion               : {champion_name}")
    print(f"AUC-PR (uncalibrated)  : {champion_metrics['auc_pr']:.4f}")
    print(f"AUC-PR (calibrated)    : {metrics_calibrated['auc_pr']:.4f}  "
          f"(delta: {metrics_calibrated['auc_pr'] - champion_metrics['auc_pr']:+.4f})")
    print(f"AUC-ROC                : {champion_metrics['auc_roc']:.4f}")
    print(f"Best iteration         : {champion_best_iter}")
    print(f"Optimal threshold      : {champion_metrics['threshold']:.4f}")
    print(f"Precision @ threshold  : {champion_metrics['precision']:.4f}")
    print(f"Recall @ threshold     : {champion_metrics['recall']:.4f}")
    print(f"F1 @ threshold         : {champion_metrics['f1']:.4f}")
    print(f"Model Registry         : {registered_model.resource_name}")
    print(f"Stage                  : production-ready")
    print(f"Calibrated model GCS   : {calibrated_gcs_uri}")
    print(f"Reliability diagram    : {reliability_uri}")
    print(f"Report                 : evaluation/final_evaluation_report.json")
    print(f"{'═'*80}")
    print(f"\nBaseline comparison:")
    print(f"  LR baseline  : 0.2172")
    print(f"  v5 best      : 0.4913  (+{0.4913 - 0.2172:.4f} vs LR)")
    print(f"  v7 champion  : {champion_metrics['auc_pr']:.4f}  "
          f"(+{champion_metrics['auc_pr'] - 0.4913:.4f} vs v5)")
    print(f"{'═'*80}\n")

    return report


if __name__ == "__main__":
    main()
