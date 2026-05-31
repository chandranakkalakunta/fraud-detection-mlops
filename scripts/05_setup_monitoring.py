"""
Phase 4 monitoring setup — idempotent.

Creates:
  1. Vertex AI Model Monitoring on the deployed endpoint
  2. Cloud Scheduler job — daily 6am IST drift check
  3. Cloud Monitoring alert policies — feature_drift_alert, performance_drift_alert

Prerequisites:
  - scripts/04_deploy_serving.py completed
  - VERTEX_ENDPOINT_RESOURCE_NAME set in .env
  - CLOUD_RUN_API_URL set in .env (Cloud Run deployment complete)

Usage:
  set -a && source .env && set +a
  ENV=dev python scripts/05_setup_monitoring.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_config
from src.utils.logging import get_logger

logger = get_logger(__name__)


def step1_vertex_model_monitoring(config: dict) -> None:
    endpoint_resource = str(config["serving"].get("endpoint_resource_name", "")).strip()
    if not endpoint_resource:
        print("⚠ VERTEX_ENDPOINT_RESOURCE_NAME not set — skipping Vertex AI Model Monitoring")
        print("  Set it in .env after running 04_deploy_serving.py, then re-run this script.")
        return

    try:
        from src.monitoring.drift_monitor import setup_model_monitoring
        print("Enabling Vertex AI Model Monitoring…")
        setup_model_monitoring(config)
        print("✓ Model monitoring enabled")
    except Exception as exc:
        print(f"⚠ Vertex AI Model Monitoring skipped: {exc}")
        print("  This is expected if the endpoint has no deployed model yet.")


def step2_cloud_scheduler(config: dict) -> None:
    api_url = str(config["serving"].get("cloud_run_api_url", "")).strip()
    if not api_url:
        print("⚠ CLOUD_RUN_API_URL not set — skipping Cloud Scheduler job")
        print("  Deploy Cloud Run first, then re-run this script.")
        return

    from src.monitoring.drift_monitor import create_scheduler_job
    print(f"Creating Cloud Scheduler job (schedule: {config['monitoring']['scheduler_schedule']} UTC)…")
    create_scheduler_job(config, api_url)
    print(f"✓ Cloud Scheduler job created — daily 6am IST → POST {api_url}/drift-check")


def step3_alert_policies(config: dict) -> None:
    try:
        from src.monitoring.drift_monitor import create_alert_policies
        print("Creating Cloud Monitoring alert policies…")
        create_alert_policies(config)
        print(f"✓ Alert policy: {config['monitoring']['feature_drift_alert_policy']}")
        print(f"✓ Alert policy: {config['monitoring']['performance_drift_alert_policy']}")
        print(f"  Notifications → {config['monitoring']['alert_email']}")
    except Exception as exc:
        print(f"⚠ Alert policy creation deferred: {exc}")
        print("  Custom metrics are created on first drift check run.")
        print("  Re-run this script after the first drift check completes.")


def main() -> None:
    env = os.getenv("ENV", "dev")
    config = load_config(env)
    project = config["gcp"]["project_id"]

    print(f"\n{'='*60}")
    print(f"Phase 4 Monitoring Setup — {project}")
    print(f"{'='*60}\n")

    print("Step 1/3 — Vertex AI Model Monitoring")
    step1_vertex_model_monitoring(config)

    print("\nStep 2/3 — Cloud Scheduler daily drift check")
    step2_cloud_scheduler(config)

    print("\nStep 3/3 — Cloud Monitoring alert policies")
    step3_alert_policies(config)

    print(f"\n{'='*60}")
    print("Phase 4 Monitoring Setup COMPLETE")
    print(f"  Drift check: daily 6am IST")
    print(f"  Features monitored: {config['monitoring']['monitored_features']}")
    print(f"  PSI alert threshold: {config['monitoring']['psi_alert_threshold']}")
    print(f"  Alerts → {config['monitoring']['alert_email']}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
