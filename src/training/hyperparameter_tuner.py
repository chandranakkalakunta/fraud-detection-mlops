"""
Vertex AI Vizier hyperparameter tuning — XGBoost and LightGBM.

Uses the Vizier v1 API (google.cloud.aiplatform_v1.VizierServiceClient) with
GAUSSIAN_PROCESS_BANDIT algorithm.  Each trial trains a model with the suggested
params, evaluates AUC-PR on the test set, reports back to Vizier, and logs to
Vertex AI Experiments.

XGBoost search space:
  max_depth         INT    [4, 10]
  learning_rate     DOUBLE [0.01, 0.15]  log scale
  min_child_weight  INT    [1, 20]
  subsample         DOUBLE [0.6, 1.0]
  colsample_bytree  DOUBLE [0.5, 1.0]
  scale_pos_weight  DOUBLE [15, 40]
  n_estimators      INT    [500, 2000]

LightGBM search space:
  num_leaves        INT    [31, 256]
  learning_rate     DOUBLE [0.01, 0.15]  log scale
  min_child_samples INT    [50, 200]
  subsample         DOUBLE [0.6, 1.0]
  colsample_bytree  DOUBLE [0.5, 1.0]
  n_estimators      INT    [500, 2000]
  reg_alpha         DOUBLE [0.0, 1.0]
  reg_lambda        DOUBLE [0.0, 1.0]

Usage:
  python -m src.training.hyperparameter_tuner --model xgboost
  python -m src.training.hyperparameter_tuner --model lightgbm
"""

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.training.metrics import compute_fraud_metrics
from src.training.train_utils import train_lgb_from_params, train_xgb_from_params
from src.training.xgboost_trainer import log_to_vertex_experiments
from src.utils.logging import get_logger

logger = get_logger(__name__)

_XGB_INT_PARAMS = {"max_depth", "min_child_weight", "n_estimators"}
_LGB_INT_PARAMS = {"num_leaves", "min_child_samples", "n_estimators"}


class VizierHPTuner:
    """Gaussian Process Bandit HP search via Vertex AI Vizier."""

    def __init__(self, model_type: str, config: dict) -> None:
        assert model_type in ("xgboost", "lightgbm"), f"Unknown model_type: {model_type}"
        self.model_type = model_type
        self.config = config
        self.project = config["gcp"]["project_id"]
        self.region = config["gcp"]["region"]
        self.parent = f"projects/{self.project}/locations/{self.region}"
        self._int_params = _XGB_INT_PARAMS if model_type == "xgboost" else _LGB_INT_PARAMS

        from google.cloud.aiplatform_v1 import VizierServiceClient
        self._client = VizierServiceClient(
            client_options={"api_endpoint": f"{self.region}-aiplatform.googleapis.com"}
        )

    # ── Search space definitions ────────────────────────────────────────────────

    def _xgb_search_space(self) -> list:
        from google.cloud.aiplatform_v1.types import StudySpec
        PS = StudySpec.ParameterSpec
        return [
            PS(
                parameter_id="max_depth",
                integer_value_spec=PS.IntegerValueSpec(min_value=4, max_value=10),
            ),
            PS(
                parameter_id="learning_rate",
                double_value_spec=PS.DoubleValueSpec(min_value=0.01, max_value=0.15),
                scale_type=PS.ScaleType.UNIT_LOG_SCALE,
            ),
            PS(
                parameter_id="min_child_weight",
                integer_value_spec=PS.IntegerValueSpec(min_value=1, max_value=20),
            ),
            PS(
                parameter_id="subsample",
                double_value_spec=PS.DoubleValueSpec(min_value=0.6, max_value=1.0),
            ),
            PS(
                parameter_id="colsample_bytree",
                double_value_spec=PS.DoubleValueSpec(min_value=0.5, max_value=1.0),
            ),
            PS(
                parameter_id="scale_pos_weight",
                double_value_spec=PS.DoubleValueSpec(min_value=15.0, max_value=40.0),
            ),
            PS(
                parameter_id="n_estimators",
                integer_value_spec=PS.IntegerValueSpec(min_value=500, max_value=2000),
            ),
        ]

    def _lgb_search_space(self) -> list:
        from google.cloud.aiplatform_v1.types import StudySpec
        PS = StudySpec.ParameterSpec
        return [
            PS(
                parameter_id="num_leaves",
                integer_value_spec=PS.IntegerValueSpec(min_value=31, max_value=256),
            ),
            PS(
                parameter_id="learning_rate",
                double_value_spec=PS.DoubleValueSpec(min_value=0.01, max_value=0.15),
                scale_type=PS.ScaleType.UNIT_LOG_SCALE,
            ),
            PS(
                parameter_id="min_child_samples",
                integer_value_spec=PS.IntegerValueSpec(min_value=50, max_value=200),
            ),
            PS(
                parameter_id="subsample",
                double_value_spec=PS.DoubleValueSpec(min_value=0.6, max_value=1.0),
            ),
            PS(
                parameter_id="colsample_bytree",
                double_value_spec=PS.DoubleValueSpec(min_value=0.5, max_value=1.0),
            ),
            PS(
                parameter_id="n_estimators",
                integer_value_spec=PS.IntegerValueSpec(min_value=500, max_value=2000),
            ),
            PS(
                parameter_id="reg_alpha",
                double_value_spec=PS.DoubleValueSpec(min_value=0.0, max_value=1.0),
            ),
            PS(
                parameter_id="reg_lambda",
                double_value_spec=PS.DoubleValueSpec(min_value=0.0, max_value=1.0),
            ),
        ]

    # ── Study lifecycle ─────────────────────────────────────────────────────────

    def _get_or_create_study(self, display_name: str) -> Any:
        """Return an existing study with this display_name, or create a new one."""
        from google.cloud.aiplatform_v1.types import Study, StudySpec

        existing = list(self._client.list_studies(parent=self.parent))
        for study in existing:
            if study.display_name == display_name:
                logger.info(
                    "reusing_existing_study",
                    extra={"study_name": study.name, "display_name": display_name},
                )
                self._cleanup_pending_trials(study.name)
                return study

        parameters = (
            self._xgb_search_space()
            if self.model_type == "xgboost"
            else self._lgb_search_space()
        )
        study_spec = StudySpec(
            algorithm=StudySpec.Algorithm.ALGORITHM_UNSPECIFIED,  # defaults to GP Bandit on backend
            metrics=[
                StudySpec.MetricSpec(
                    metric_id="auc_pr",
                    goal=StudySpec.MetricSpec.GoalType.MAXIMIZE,
                )
            ],
            parameters=parameters,
        )
        study = self._client.create_study(
            parent=self.parent,
            study=Study(display_name=display_name, study_spec=study_spec),
        )
        logger.info(
            "study_created",
            extra={"study_name": study.name, "display_name": display_name},
        )
        return study

    # ── Trial utilities ─────────────────────────────────────────────────────────

    def _cleanup_pending_trials(self, study_name: str) -> None:
        """Mark ACTIVE/REQUESTED trials as infeasible — cleans up incomplete runs."""
        from google.cloud.aiplatform_v1.types import Trial
        try:
            trials = list(self._client.list_trials(parent=study_name))
            cleaned = 0
            for trial in trials:
                if trial.state in (Trial.State.ACTIVE, Trial.State.REQUESTED):
                    self._infeasible_trial(trial, "cleanup_previous_incomplete_run")
                    cleaned += 1
            if cleaned:
                logger.info("cleaned_pending_trials", extra={"count": cleaned, "study": study_name})
        except Exception as exc:
            logger.warning("cleanup_pending_trials_failed", extra={"error": str(exc)})

    def _extract_params(self, trial: Any) -> dict:
        """Extract parameter values from a Vizier Trial into a plain dict."""
        params: dict = {}
        for p in trial.parameters:
            # p.value is a Python float in aiplatform_v1 — no .number_value wrapper
            value = float(p.value)
            if p.parameter_id in self._int_params:
                value = int(round(value))
            params[p.parameter_id] = value
        return params

    def _complete_trial(self, trial: Any, auc_pr: float) -> None:
        from google.cloud.aiplatform_v1.types import Measurement
        measurement = Measurement(
            metrics=[Measurement.Metric(metric_id="auc_pr", value=auc_pr)]
        )
        self._client.complete_trial(
            request={
                "name": trial.name,
                "final_measurement": measurement,
                "trial_infeasible": False,
            }
        )

    def _infeasible_trial(self, trial: Any, reason: str) -> None:
        self._client.complete_trial(
            request={
                "name": trial.name,
                "trial_infeasible": True,
                "infeasible_reason": reason[:200],
            }
        )

    # ── Main search loop ────────────────────────────────────────────────────────

    def run(
        self,
        X_train_eng: pd.DataFrame,
        y_train: pd.Series,
        X_val_eng: pd.DataFrame,
        y_val: pd.Series,
        X_test_eng: pd.DataFrame,
        y_test: pd.Series,
    ) -> dict:
        """
        Run HP search, log every trial to Vertex AI Experiments.

        Expects pre-engineered feature DataFrames (output of apply_engineer) —
        the engineer is not re-applied per trial.

        Returns the best params dict found across all trials.
        """
        vizier_cfg = self.config["tuning"]["vizier"]
        max_trials = int(vizier_cfg["max_trials"])
        parallel_trials = int(vizier_cfg["parallel_trials"])
        study_key = "xgb_study_name" if self.model_type == "xgboost" else "lgb_study_name"
        study_display = vizier_cfg[study_key]

        study = self._get_or_create_study(study_display)

        imbalance_strategy = (
            "scale_pos_weight" if self.model_type == "xgboost" else "lgb_is_unbalance"
        )
        trials_completed = 0
        best_auc_pr = 0.0
        best_params: dict = {}

        print(f"\n{'═'*80}")
        print(f"Vizier HP search — {self.model_type.upper()}  ({max_trials} trials, parallel={parallel_trials})")
        print(f"Study: {study_display}")
        print(f"{'═'*80}")
        print(f"{'Trial':>6}  {'AUC-PR':>8}  {'best_iter':>10}  {'Note'}")
        print(f"{'─'*80}")

        while trials_completed < max_trials:
            remaining = max_trials - trials_completed
            batch_size = min(parallel_trials, remaining)

            # Suggest batch
            operation = self._client.suggest_trials(
                request={
                    "parent": study.name,
                    "suggestion_count": batch_size,
                    "client_id": f"fraud-tuner-{self.model_type}",
                }
            )
            response = operation.result(timeout=600)

            for trial in response.trials:
                trial_num = trials_completed + 1
                trial_params = self._extract_params(trial)

                try:
                    if self.model_type == "xgboost":
                        model = train_xgb_from_params(
                            X_train_eng, y_train, X_val_eng, y_val,
                            trial_params, self.config,
                        )
                        best_iter = model.best_iteration
                    else:
                        model = train_lgb_from_params(
                            X_train_eng, y_train, X_val_eng, y_val,
                            trial_params, self.config,
                        )
                        best_iter = model.best_iteration_

                    y_proba = model.predict_proba(X_test_eng)[:, 1]
                    metrics = compute_fraud_metrics(
                        y_true=y_test.values,
                        y_proba=y_proba,
                        imbalance_strategy=imbalance_strategy,
                        training_rows=len(X_train_eng),
                        test_rows=len(X_test_eng),
                    )
                    auc_pr = float(metrics["auc_pr"])

                    self._complete_trial(trial, auc_pr)

                    run_name = f"{self.model_type}-vizier-trial-{trial_num:03d}"
                    log_to_vertex_experiments(
                        self.config,
                        run_name,
                        {
                            "model_type": self.model_type,
                            "trial_number": trial_num,
                            "best_iteration": best_iter,
                            **trial_params,
                        },
                        metrics,
                    )

                    is_best = auc_pr > best_auc_pr
                    if is_best:
                        best_auc_pr = auc_pr
                        best_params = dict(trial_params)

                    print(
                        f"{trial_num:>6}  {auc_pr:>8.4f}  {best_iter:>10}  "
                        f"{'★ NEW BEST' if is_best else ''}"
                    )
                    logger.info(
                        "trial_completed",
                        extra={
                            "trial_num": trial_num,
                            "auc_pr": round(auc_pr, 4),
                            "best_auc_pr": round(best_auc_pr, 4),
                            "best_iter": best_iter,
                            "params": trial_params,
                        },
                    )

                except Exception as exc:
                    logger.error(
                        "trial_failed",
                        extra={"trial_num": trial_num, "error": str(exc)},
                    )
                    self._infeasible_trial(trial, str(exc))
                    print(f"{trial_num:>6}  {'FAILED':>8}  {'N/A':>10}  {str(exc)[:40]}")

                trials_completed += 1

        print(f"{'─'*80}")
        print(f"Search complete.  Best AUC-PR: {best_auc_pr:.4f}")
        print(f"{'═'*80}\n")

        # Cross-check with Vizier's own optimal trial
        try:
            optimal_resp = self._client.list_optimal_trials(parent=study.name)
            if optimal_resp.optimal_trials:
                vizier_best = self._extract_params(optimal_resp.optimal_trials[0])
                logger.info("vizier_optimal_trial", extra={"params": vizier_best})
                best_params = vizier_best
        except Exception as exc:
            logger.warning("list_optimal_trials_failed", extra={"error": str(exc)})

        return best_params


# ── Script entrypoint ─────────────────────────────────────────────────────────

def _run_tuning(model_type: str) -> None:
    import os
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=".env")

    from src.training.train_utils import (
        apply_engineer,
        build_no_te_engineer,
        prepare_data_splits,
    )
    from src.training.xgboost_trainer import load_engineered_features
    from src.utils.config import load_config

    env = os.getenv("ENV", "dev")
    config = load_config(env)

    print(f"Loading data...")
    df = load_engineered_features(config)
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_data_splits(df, config)

    print(f"Engineering features...")
    engineer = build_no_te_engineer(config)
    X_train_eng, X_val_eng, X_test_eng = apply_engineer(
        engineer, X_train, y_train, X_val, X_test
    )
    print(f"Features: {X_train_eng.shape[1]}")

    tuner = VizierHPTuner(model_type=model_type, config=config)
    best_params = tuner.run(X_train_eng, y_train, X_val_eng, y_val, X_test_eng, y_test)

    vizier_cfg = config["tuning"]["vizier"]
    if model_type == "xgboost":
        out_path = Path(vizier_cfg["best_params_xgb_path"])
    else:
        out_path = Path(vizier_cfg["best_params_lgb_path"])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        yaml.dump(best_params, f, default_flow_style=False, sort_keys=True)

    print(f"\nBest params saved to: {out_path}")
    for k, v in sorted(best_params.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["xgboost", "lightgbm"], required=True)
    args = parser.parse_args()
    _run_tuning(args.model)
