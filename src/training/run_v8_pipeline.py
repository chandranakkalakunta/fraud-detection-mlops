"""
v8 pipeline — TransactionID leakage fix.

TransactionID was identified as rank-7 SHAP feature (mean |SHAP| = 0.241) in
lgb-v7-tuned despite being a unique row identifier.  Its importance reflects
temporal ordering correlation with the time-based split — lower IDs map to
earlier (training) transactions.  This is data leakage.

This pipeline:
  1. Re-runs feature engineering with TransactionID excluded via EXCLUDED_FEATURES
     in engineer.py (387 features, down from 388).  Saves updated transformer to GCS.
  2. Retrains LightGBM with identical Vizier best params (config/best_params_lgb.yaml),
     identical split strategy, early_stopping_rounds=100.
  3. Logs to Vertex AI Experiments as lgb-v8-no-txnid.
  4. Recomputes SHAP on test set — asserts TransactionID absent.
     Overwrites GCS SHAP plots (shap_feature_importance_top20.png, shap_beeswarm.png).
  5. Applies Platt scaling calibration (fit on val set, eval on test).
     Overwrites reliability_diagram.png.
  6. Registers lgb-v8-no-txnid as production-ready in Vertex AI Model Registry.
  7. Demotes lgb-v7-tuned labels: stage production-ready → staging.
  8. Rewrites evaluation/final_evaluation_report.json with v8 as champion.

Pre-requisites:
  config/best_params_lgb.yaml  (written by hyperparameter_tuner.py --model lightgbm)
"""

import io
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
)
from src.training.xgboost_trainer import (
    load_engineered_features,
    log_to_vertex_experiments,
    save_model_to_gcs,
)
from src.utils.config import load_config
from src.utils.logging import get_logger

logger = get_logger(__name__)

RUN_NAME = "lgb-v8-no-txnid"
V7_MODEL_RESOURCE = "projects/65768314585/locations/asia-south1/models/9011760028973006848"
V7_AUC_PR = 0.539283


def _demote_v7_to_staging(config: dict) -> None:
    """Update lgb-v7-tuned labels: stage production-ready → staging."""
    import google.cloud.aiplatform as aip

    aip.init(project=config["gcp"]["project_id"], location=config["gcp"]["region"])
    model = aip.Model(V7_MODEL_RESOURCE)
    model.update(
        labels={
            "stage": "staging",
            "champion_run": "lgb-v7-tuned",
            "phase": "v7-hp-tuned",
            "demoted-reason": "txnid-leakage",
        }
    )
    logger.info("v7_demoted_to_staging", extra={"resource": V7_MODEL_RESOURCE})


def _register_v8_champion(auc_pr: float, auc_roc: float, config: dict) -> Any:
    """Register lgb-v8-no-txnid in Vertex AI Model Registry as production-ready."""
    import google.cloud.aiplatform as aip

    aip.init(project=config["gcp"]["project_id"], location=config["gcp"]["region"])
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%S")
    model_dir_uri = f"gs://{config['storage']['artifacts_bucket']}/models/{RUN_NAME}/"
    serving_container = config["vertex"]["serving_container"]["lightgbm"]

    registered = aip.Model.upload(
        display_name=config["vertex"]["model_display_name"],
        artifact_uri=model_dir_uri,
        serving_container_image_uri=serving_container,
        description=(
            f"Fraud detection champion (v8, TransactionID leakage fixed) — {RUN_NAME}. "
            f"AUC-PR: {auc_pr:.4f}, AUC-ROC: {auc_roc:.4f}. "
            f"Registered {timestamp}."
        ),
        labels={
            "stage": "production-ready",
            "champion-run": RUN_NAME,
            "phase": "v8-txnid-fix",
        },
    )
    logger.info(
        "v8_champion_registered",
        extra={"resource_name": registered.resource_name, "auc_pr": round(auc_pr, 4)},
    )
    return registered


def _reliability_diagram_to_gcs(
    y_true: np.ndarray,
    y_proba_uncal: np.ndarray,
    y_proba_cal: np.ndarray,
    config: dict,
) -> str:
    from sklearn.calibration import calibration_curve
    from google.cloud import storage as gcs

    fop_uncal, mpv_uncal = calibration_curve(y_true, y_proba_uncal, n_bins=10)
    fop_cal, mpv_cal = calibration_curve(y_true, y_proba_cal, n_bins=10)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.plot(mpv_uncal, fop_uncal, "o-", color="#e74c3c", label=f"Uncalibrated — {RUN_NAME}")
    ax.plot(mpv_cal, fop_cal, "s-", color="#2ecc71", label="Platt-calibrated")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives (fraud rate in bin)")
    ax.set_title(f"Reliability Diagram — {RUN_NAME}")
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
    client.bucket(bucket_name).blob(blob_path).upload_from_file(buf, content_type="image/png")
    uri = f"gs://{bucket_name}/{blob_path}"
    logger.info("reliability_diagram_saved", extra={"uri": uri})
    return uri


def main() -> dict:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=".env")

    env = os.getenv("ENV", "dev")
    config = load_config(env)
    vizier_cfg = config["tuning"]["vizier"]

    # Load LGB best params (identical to v7)
    lgb_params_path = Path(vizier_cfg["best_params_lgb_path"])
    if not lgb_params_path.exists():
        raise FileNotFoundError(
            f"Best LGB params not found: {lgb_params_path}\n"
            "Run: python -m src.training.hyperparameter_tuner --model lightgbm"
        )
    with open(lgb_params_path) as f:
        lgb_best_params = yaml.safe_load(f)
    logger.info("best_params_loaded", extra={"path": str(lgb_params_path), "params": lgb_best_params})
    print(f"LGB best params: {lgb_best_params}")

    # Load data and prepare splits (time-based, identical to v7)
    df = load_engineered_features(config)
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_data_splits(df, config)

    print(f"\nSplit summary (v8):")
    print(f"  Train : {len(X_train):>7,}  ({y_train.mean():.3%} fraud)")
    print(f"  Val   : {len(X_val):>7,}  ({y_val.mean():.3%} fraud)  stratified")
    print(f"  Test  : {len(X_test):>7,}  ({y_test.mean():.3%} fraud)  temporal")

    # Feature engineering — TransactionID now excluded via EXCLUDED_FEATURES in engineer.py
    engineer = build_no_te_engineer(config)
    X_train_eng, X_val_eng, X_test_eng = apply_engineer(
        engineer, X_train, y_train, X_val, X_test
    )
    n_features = X_train_eng.shape[1]

    print(f"\nFeatures: {n_features}  (no TE, no indicators, TransactionID excluded)")
    assert n_features == 387, f"Expected 387 features after TransactionID removal, got {n_features}"
    assert "TransactionID" not in X_train_eng.columns, "TransactionID still present in features!"
    assert X_train_eng.isnull().sum().sum() == 0, "NaN in train features"
    assert X_val_eng.isnull().sum().sum() == 0, "NaN in val features"
    assert X_test_eng.isnull().sum().sum() == 0, "NaN in test features"
    print(f"  NaN check: PASSED (0 NaN in all splits)")

    # Save updated transformer to GCS
    transformer_uri = engineer.save_to_gcs(config)
    print(f"  Transformer saved: {transformer_uri}")

    # Train lgb-v8-no-txnid
    print(f"\nTraining {RUN_NAME}...")
    model_lgb = train_lgb_from_params(
        X_train_eng, y_train, X_val_eng, y_val, lgb_best_params, config
    )
    y_proba = model_lgb.predict_proba(X_test_eng)[:, 1]
    metrics = compute_fraud_metrics(
        y_true=y_test.values,
        y_proba=y_proba,
        imbalance_strategy="lgb_is_unbalance",
        training_rows=len(X_train),
        test_rows=len(X_test),
    )

    log_to_vertex_experiments(
        config,
        RUN_NAME,
        {
            "model_type": "lightgbm",
            "n_features": n_features,
            "best_iteration": model_lgb.best_iteration_,
            "phase": "v8-txnid-fix",
            "txnid_excluded": "true",
            **lgb_best_params,
        },
        metrics,
    )
    save_model_to_gcs(model_lgb, RUN_NAME, config)

    print(f"\n{RUN_NAME}  AUC-PR={metrics['auc_pr']:.4f}  best_iter={model_lgb.best_iteration_}")

    # Register as production-ready
    print(f"\nRegistering {RUN_NAME} as production-ready...")
    registered_model = _register_v8_champion(metrics["auc_pr"], metrics["auc_roc"], config)

    # SHAP on test set — overwrites v7 GCS plots
    print(f"\nComputing SHAP values on test set...")
    top_n = int(config["explainability"]["top_n_features"])
    feature_importance, bar_uri, beeswarm_uri = compute_shap_and_save(
        model=model_lgb,
        X_test=X_test_eng,
        champion_name=RUN_NAME,
        config=config,
        top_n=top_n,
    )
    print_top_features(feature_importance, n=10)

    assert "TransactionID" not in feature_importance["feature"].tolist(), (
        "TransactionID still appears in SHAP output — exclusion failed!"
    )
    print(f"\n  TransactionID confirmed ABSENT from all {len(feature_importance)} SHAP features.")

    # Platt calibration (fit on val, eval on test)
    print(f"\nCalibrating with Platt scaling (val set)...")
    from sklearn.calibration import CalibratedClassifierCV

    calibrated_model = CalibratedClassifierCV(model_lgb, cv="prefit", method="sigmoid")
    calibrated_model.fit(X_val_eng, y_val)

    y_proba_cal = calibrated_model.predict_proba(X_test_eng)[:, 1]
    metrics_calibrated = compute_fraud_metrics(
        y_true=y_test.values,
        y_proba=y_proba_cal,
        imbalance_strategy="lgb_is_unbalance",
        training_rows=len(X_train),
        test_rows=len(X_test),
    )

    calibrated_run_name = f"{RUN_NAME}-calibrated"
    log_to_vertex_experiments(
        config,
        calibrated_run_name,
        {
            "model_type": "lightgbm",
            "calibration": "platt_sigmoid",
            "calibration_set": "val",
            "n_features": n_features,
            "base_model": RUN_NAME,
            "phase": "v8-txnid-fix",
            **lgb_best_params,
        },
        metrics_calibrated,
    )

    reliability_uri = _reliability_diagram_to_gcs(
        y_true=y_test.values,
        y_proba_uncal=y_proba,
        y_proba_cal=y_proba_cal,
        config=config,
    )

    calibrated_gcs_uri = save_model_to_gcs(calibrated_model, calibrated_run_name, config)
    print(f"Calibrated model saved: {calibrated_gcs_uri}")

    # Demote v7 to staging
    print(f"\nDemoting lgb-v7-tuned to staging...")
    _demote_v7_to_staging(config)

    # Rewrite evaluation/final_evaluation_report.json
    auc_pr_delta = round(metrics["auc_pr"] - V7_AUC_PR, 6)
    if abs(auc_pr_delta) > 0.02:
        interpretation = "TransactionID was inflating AUC-PR via temporal leakage."
    else:
        interpretation = "TransactionID was marginal — model is clean and retains full signal."

    report = {
        "champion": {
            "name": RUN_NAME,
            "version": registered_model.version_id if hasattr(registered_model, "version_id") else "1",
            "model_registry_resource": registered_model.resource_name,
            "stage": "production-ready",
            "gcs_base_uri": f"gs://{config['storage']['artifacts_bucket']}/models/{RUN_NAME}/",
            "calibrated_gcs_uri": calibrated_gcs_uri,
        },
        "feature_engineering": {
            "n_features": n_features,
            "target_encoding": False,
            "missing_indicators": False,
            "excluded_features": ["TransactionID"],
            "exclusion_reason": (
                "TransactionID ranked 7th in lgb-v7 SHAP (mean |SHAP|=0.241) despite being a "
                "unique row identifier with no genuine fraud signal.  Temporal ordering "
                "correlation with the time-based split caused the model to learn "
                "train/test membership.  Excluded permanently via EXCLUDED_FEATURES "
                "in src/features/engineer.py."
            ),
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
        "v7_vs_v8_comparison": {
            "lgb_v7_auc_pr": V7_AUC_PR,
            "lgb_v8_auc_pr": metrics["auc_pr"],
            "auc_pr_delta": auc_pr_delta,
            "interpretation": interpretation,
            "txnid_v7_shap_rank": 7,
            "txnid_v7_mean_abs_shap": 0.241025,
        },
        "champion_metrics_uncalibrated": {
            **metrics,
            "best_iteration": model_lgb.best_iteration_,
            "business_interpretation": (
                f"At threshold {metrics['threshold']:.4f}: "
                f"Precision={metrics['precision']:.3f} "
                f"(of flagged transactions, {metrics['precision']*100:.1f}% are fraud), "
                f"Recall={metrics['recall']:.3f} "
                f"({metrics['recall']*100:.1f}% of fraud transactions caught)."
            ),
        },
        "champion_metrics_calibrated": {
            **metrics_calibrated,
            "calibration_method": "platt_sigmoid",
            "calibration_set": "validation",
            "auc_pr_delta_vs_uncalibrated": round(
                metrics_calibrated["auc_pr"] - metrics["auc_pr"], 4
            ),
        },
        "best_hyperparameters": {
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
            "lgb_v7_auc_pr": V7_AUC_PR,
        },
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
    }

    out_path = Path("evaluation/final_evaluation_report.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=str))
    logger.info("final_report_saved", extra={"path": str(out_path)})

    print(f"\n{'═'*80}")
    print("V8 PIPELINE COMPLETE — TransactionID Leakage Fix")
    print(f"{'═'*80}")
    print(f"Champion               : {RUN_NAME}")
    print(f"Feature count          : {n_features}  (TransactionID excluded)")
    print(f"AUC-PR (uncalibrated)  : {metrics['auc_pr']:.4f}")
    print(f"AUC-PR (calibrated)    : {metrics_calibrated['auc_pr']:.4f}  "
          f"(delta: {metrics_calibrated['auc_pr'] - metrics['auc_pr']:+.4f})")
    print(f"AUC-ROC                : {metrics['auc_roc']:.4f}")
    print(f"Best iteration         : {model_lgb.best_iteration_}")
    print(f"Optimal threshold      : {metrics['threshold']:.4f}")
    print(f"Precision @ threshold  : {metrics['precision']:.4f}")
    print(f"Recall @ threshold     : {metrics['recall']:.4f}")
    print(f"F1 @ threshold         : {metrics['f1']:.4f}")
    print(f"Model Registry         : {registered_model.resource_name}")
    print(f"Stage                  : production-ready")
    print(f"Calibrated model GCS   : {calibrated_gcs_uri}")
    print(f"{'═'*80}")
    print(f"\nAUC-PR comparison — TransactionID leakage analysis:")
    print(f"  lgb-v7-tuned    (with TransactionID)   : {V7_AUC_PR:.4f}")
    print(f"  lgb-v8-no-txnid (without TransactionID): {metrics['auc_pr']:.4f}")
    print(f"  Delta                                  : {auc_pr_delta:+.4f}")
    print(f"  Interpretation                         : {interpretation}")
    print(f"{'═'*80}\n")

    return report


if __name__ == "__main__":
    main()
