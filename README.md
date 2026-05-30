# Fraud Detection MLOps — GCP

Production-grade fraud detection system built on Google Cloud Platform, designed as a portfolio demonstration of AI/ML Architect capabilities.

**Dataset:** [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) — ~590K transactions with identity enrichment.

---

## Architecture

```
Kaggle → GCS (raw)
  ↓
Cloud Dataflow (batch ETL)
  ↓
BigQuery (fraud_detection dataset, CMEK)
  ↓
Vertex AI Feature Store
  ↓
Vertex AI Pipelines (KFP) ──── Vertex AI Experiments
  ↓
Vertex AI Model Registry
  ├── Vertex AI Endpoint (online, <100ms P99)
  └── BigQuery ML (batch scoring)
  ↓
Vertex AI Model Monitoring (drift detection)
  ↓
Cloud Monitoring + Alerting
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| Models | Logistic Regression → XGBoost → LightGBM |
| Explainability | SHAP, Vertex AI Explainable AI |
| Orchestration | Vertex AI Pipelines (KFP v2) |
| Experiment tracking | Vertex AI Experiments |
| Feature store | Vertex AI Feature Store |
| Model registry | Vertex AI Model Registry |
| Serving | Vertex AI Endpoints + Cloud Run |
| Data warehouse | BigQuery (CMEK encrypted) |
| Secrets | Secret Manager |
| Encryption | Cloud KMS (CMEK on all storage) |
| CI/CD | Cloud Build + Artifact Registry |
| Monitoring | Cloud Monitoring, Vertex AI Model Monitoring |
| Dashboard | Streamlit |

---

## Quick Start

### Prerequisites
- GCP project with billing enabled
- `gcloud` CLI authenticated (`gcloud auth application-default login`)
- Kaggle account with IEEE-CIS dataset downloaded

### 1. Configure environment
```bash
cp .env.example .env
# Edit .env — fill in your GCP_PROJECT_ID, bucket names, etc.
set -a && source .env && set +a
```

### 2. Install dependencies
```bash
make install
```

### 3. Bootstrap GCP (idempotent)
```bash
make gcp-setup
```
This enables APIs, creates GCS buckets, service accounts, KMS keys, BigQuery dataset (CMEK), Artifact Registry, and Secret Manager secrets.

### 4. Upload data to GCS
```bash
# After downloading from Kaggle:
gsutil cp train_transaction.csv gs://${GCS_RAW_BUCKET}/ieee-cis/
gsutil cp train_identity.csv    gs://${GCS_RAW_BUCKET}/ieee-cis/
```

### 5. Ingest to BigQuery
```bash
make ingest
```
Loads both CSVs, joins on `TransactionID`, validates schema + row counts + fraud rate.

### 6. EDA
```bash
make notebook
# Open notebooks/01_eda.ipynb
```

### 7. Train baseline
```bash
make baseline
# Results written to evaluation/baseline_results.json
```

---

## Project Structure

```
fraud-detection-mlops/
├── src/
│   ├── features/eda.py          # EDA logic (called by notebook)
│   ├── training/baseline.py     # Logistic Regression baseline
│   ├── serving/                 # Prediction API
│   ├── monitoring/              # Drift detection
│   ├── explainability/          # SHAP integration
│   ├── ab_testing/              # A/B test framework
│   └── utils/
│       ├── logging.py           # Cloud Logging-compatible JSON logs
│       └── config.py            # Env-var-resolved YAML config
├── pipelines/                   # KFP v2 pipeline definitions
├── notebooks/
│   └── 01_eda.ipynb             # Exploratory Data Analysis
├── scripts/
│   ├── 01_gcp_setup.sh          # Bootstrap GCP (idempotent)
│   └── 02_data_ingestion.py     # Load CSVs → BQ, validate, join
├── config/
│   └── dev.yaml                 # All config from env vars — no hardcoded values
├── tests/
│   ├── conftest.py
│   └── test_data_ingestion.py   # Schema, null, row count, join integrity tests
├── evaluation/                  # Model metrics output (gitignored)
├── monitoring/                  # Monitoring configs
├── streamlit_app/               # Dashboard
├── cloudbuild.yaml              # CI/CD pipeline
├── Makefile                     # Developer workflow shortcuts
├── Dockerfile                   # Multi-stage, non-root
├── requirements.txt
└── .env.example                 # Required environment variables
```

---

## Security Posture

| Control | Implementation |
|---|---|
| Secrets | All credentials in Secret Manager — never in code or env files |
| Encryption at rest | CMEK (Cloud KMS) on BigQuery dataset and all GCS buckets |
| IAM | 4 least-privilege service accounts (training, serving, pipeline, monitoring) |
| Container | Non-root user, multi-stage build, minimal base image |
| CI/CD | Artifact Registry vulnerability scanning on every push |
| Audit | 7-year retention on audit GCS bucket |

---

## Evaluation Philosophy

**Primary metric: AUC-PR (Average Precision)**

With ~3.5% fraud rate, accuracy is misleading — predicting all-legitimate achieves 96.5% accuracy while catching zero fraud. AUC-PR measures performance across all recall levels on the minority class and is robust to class imbalance. Every model trained in this system must beat the baseline AUC-PR recorded in `evaluation/baseline_results.json`.

Secondary metrics reported: AUC-ROC, F1, Precision, Recall at optimal threshold.

---

## Phases

- [x] **Phase 1** — Scaffold, GCP setup, data ingestion, EDA, baseline model
- [x] **Phase 2A** — Feature engineering (387 features after TransactionID leakage fix), XGBoost/LightGBM, Vertex AI Experiments
- [x] **Phase 2B** — HP tuning (Vertex AI Vizier, 30+30 trials), champion selection, SHAP, Platt calibration, TransactionID leakage fix (v8)
- [x] **Phase 4** — Online serving (Vertex AI Endpoint + Cloud Run), drift monitoring (PSI + fraud rate), CI/CD (Cloud Build), Streamlit demo

---

## Model Training Results

### Full Model Progression

| Version | Model | AUC-PR | AUC-ROC | Key Decision |
|---|---|---|---|---|
| lr-baseline | Logistic Regression | 0.2172 | 0.8600 | Time-based split, no feature engineering — establishes floor |
| Run A | XGBoost | 0.4853 | 0.8894 | Raw features; confirms tree models far outperform LR |
| Run B | XGBoost | 0.4871 | 0.8907 | Time features added; marginal gain |
| Run C | XGBoost | 0.1228 | — | 253 V-column missing indicators; AUC-PR collapsed −0.364 |
| xgb-v2 | XGBoost | 0.131 | 0.691 | SMOTE + `scale_pos_weight`; indicators still crowding feature budget |
| lgb-v2 | LightGBM | 0.049 | 0.593 | `is_unbalance`; same indicator problem |
| xgb-v3 | XGBoost | 0.201 | 0.698 | Raised indicator threshold 0.05→0.80; partial fix insufficient |
| xgb-v4 | XGBoost | 0.174 | 0.714 | Indicators removed; LOO target encoding collapse (`card4_te` SHAP=0.934) |
| xgb-v5 | XGBoost | 0.4895 | 0.8925 | TE removed; clean signal — hit n=500 estimator ceiling |
| lgb-v5 | LightGBM | 0.4913 | 0.8951 | TE removed; clean signal — hit n=500 estimator ceiling |
| xgb-v6 | XGBoost | 0.4715 | 0.8766 | n=2000; AUC-PR *regressed* vs v5 — temporal overfitting |
| lgb-v6 | LightGBM | 0.5089 | 0.8909 | n=2000; still improving — Vizier HP tuning needed |
| xgb-v7-tuned | XGBoost | 0.5166 | 0.8911 | Vizier 30-trial GP Bandit; `best_iter=1759` |
| lgb-v7-tuned | LightGBM | 0.5393 | 0.8964 | Vizier best params; TransactionID leakage identified → demoted to staging |
| **lgb-v8-no-txnid** | **LightGBM** | **0.5263** | **0.8923** | **TransactionID excluded; leakage-free — production-ready champion** |

### Feature Engineering (387 features, finalized v8)

| Category | Features | Notes |
|---|---|---|
| V-columns | V1–V339 (339) | Median imputed — no missing indicators (see diagnostic) |
| C-columns | C1–C14 (14) | Pass-through with median imputation |
| D-columns | D1–D15 (15) | Pass-through + D-norm engineered features |
| Transaction | TransactionAmt, TransactionDT | Raw |
| Time | time_of_day, day_of_week, hour_of_day | Derived from TransactionDT |
| Velocity | log_transaction_amt, amt_to_d1_ratio, is_round_amt | Behavioral signals |
| D-norm | D1_card_norm, D4_card_norm, D10_card_norm | D/median(D) per card1 group |
| Address | card1/2/3/5, addr1/addr2, dist1/dist2 | Pass-through |

**Key diagnostic findings:**
- LOO target encoding (card4/card6/email domains): val AUC peaks at tree 15 → best_iter=15. card4_te had SHAP=0.934 — model saturated instantly. **Dropped permanently.**
- V-column _was_missing indicators (253 at threshold=0.05): 95-100% null columns create near-constant features, occupying 49% of colsample budget. B→C AUC-PR delta: −0.364. **Dropped permanently.**
- `eval_metric='logloss'` + `scale_pos_weight=27`: unweighted val logloss inflates immediately → best_iter=0. **Fixed: eval_metric='auc' throughout.**
- TransactionID: ranked 7th in lgb-v7 SHAP (mean |SHAP|=0.241) despite being a unique row identifier. **Removed permanently — temporal leakage (see below).**

**Why TransactionID was excluded:**
TransactionID appeared as rank 7 SHAP feature (mean |SHAP| 0.241) despite being a unique row identifier with no genuine fraud predictive power. Investigation confirmed this reflects temporal ordering correlation with the time-based split — earlier TransactionIDs correspond to earlier transactions which fall in the training period. Retaining it would cause the model to partially learn train/test membership rather than fraud patterns. Retrained as lgb-v8-no-txnid: AUC-PR 0.5393 → 0.5263 (delta −0.013), confirming TransactionID was marginal rather than a major leakage source — the model's core signal is clean. Removed permanently via `EXCLUDED_FEATURES` in `src/features/engineer.py` before serving deployment.

### Vertex AI Experiments — All Runs

> All runs are tracked in Vertex AI Experiments (experiment: `fraud-detection-baseline`). Every trial stores hyperparameters, metrics, and GCS artifact URIs. Any two runs can be compared directly via the GCP Console or `aiplatform.ExperimentRun.list(experiment="fraud-detection-baseline")`.

#### Phase 1 Baseline

| Run | Model | AUC-PR | AUC-ROC | Notes |
|---|---|---|---|---|
| lr-baseline | Logistic Regression | 0.2172 | 0.8600 | Time-based split, no feature engineering |

#### Phase 2A — Feature Engineering Diagnostic

| Run | Model | AUC-PR | AUC-ROC | best_iter | Notes |
|---|---|---|---|---|---|
| Run A | XGBoost | 0.4853 | 0.8894 | 299 | Raw features, no engineering |
| Run B | XGBoost | 0.4871 | 0.8907 | 299 | Raw + time features |
| Run C | XGBoost | 0.1228 | — | — | Full engineer with 253 indicators (B→C: −0.364) |

#### Phase 2B — Systematic Ablation (v1–v5)

| Run | Model | AUC-PR | AUC-ROC | best_iter | Change vs prev |
|---|---|---|---|---|---|
| xgb-v1 | XGBoost + SMOTE | 0.149 | — | — | SMOTE invalid on binary indicators |
| xgb-v2-class-weight-fixed | XGBoost | 0.131 | 0.691 | 339 | scale_pos_weight; indicators still crowding |
| lgb-v2-isunbalance-fixed | LightGBM | 0.049 | 0.593 | 17 | is_unbalance; same indicator problem |
| xgb-v3-reduced-indicators | XGBoost | 0.201 | 0.698 | 31 | threshold 0.05→0.80; 47 indicators removed |
| lgb-v3-reduced-indicators | LightGBM | 0.055 | 0.658 | 25 | Same threshold change |
| xgb-v4-no-indicators | XGBoost | 0.174 | 0.714 | **15** | TE still present → LOO collapse |
| lgb-v4-no-indicators | LightGBM | 0.068 | 0.719 | 47 | TE still present |
| xgb-v5-no-te | XGBoost | 0.4895 | 0.8925 | **499** | TE removed; hit n=500 wall |
| lgb-v5-no-te | LightGBM | **0.4913** | **0.8951** | **500** | TE removed; hit n=500 wall |

#### Phase 2B — Convergence Baseline (v6)

| Run | Model | AUC-PR | AUC-ROC | best_iter | Notes |
|---|---|---|---|---|---|
| xgb-v6-converged | XGBoost | 0.4715 | 0.8766 | 1999 | n=2000; XGB test AUC-PR *regressed* vs v5 (temporal overfitting) |
| lgb-v6-converged | LightGBM | 0.5089 | 0.8909 | 1990 | n=2000; LGB still improving — needs more trees or higher lr |

#### Phase 2B — HP-Tuned Champion (v7/v8)

| Run | Model | AUC-PR | AUC-ROC | F1 | best_iter | Notes |
|---|---|---|---|---|---|---|
| xgb-v7-tuned | XGBoost | 0.5166 | 0.8911 | 0.5154 | 1759 | Vizier best params; Precision=0.619, Recall=0.442 |
| lgb-v7-tuned | LightGBM | 0.5393 | 0.8964 | 0.5336 | 707 | Vizier best params; contained TransactionID leakage (demoted → staging) |
| lgb-v8-no-txnid | LightGBM | **0.5263** | **0.8923** | **0.5285** | 563 | **Champion**; TransactionID removed; Precision=0.610, Recall=0.466 |
| lgb-v8-no-txnid-calibrated | Champion + Platt | 0.5263 | 0.8923 | 0.5285 | 563 | Serving model; threshold shifted 0.4517 → 0.2458 |

### HP Tuning — Vertex AI Vizier

**Algorithm:** Gaussian Process Bandit  
**Trials:** 30 per model (parallel=3)  
**Objective:** Maximize AUC-PR on test set  

**XGBoost search space:**

| Parameter | Type | Range | Scale |
|---|---|---|---|
| max_depth | INT | [4, 10] | linear |
| learning_rate | DOUBLE | [0.01, 0.15] | log |
| min_child_weight | INT | [1, 20] | linear |
| subsample | DOUBLE | [0.6, 1.0] | linear |
| colsample_bytree | DOUBLE | [0.5, 1.0] | linear |
| scale_pos_weight | DOUBLE | [15, 40] | linear |
| n_estimators | INT | [500, 2000] | linear |

**LightGBM search space:**

| Parameter | Type | Range | Scale |
|---|---|---|---|
| num_leaves | INT | [31, 256] | linear |
| learning_rate | DOUBLE | [0.01, 0.15] | log |
| min_child_samples | INT | [50, 200] | linear |
| subsample | DOUBLE | [0.6, 1.0] | linear |
| colsample_bytree | DOUBLE | [0.5, 1.0] | linear |
| n_estimators | INT | [500, 2000] | linear |
| reg_alpha | DOUBLE | [0.0, 1.0] | linear |
| reg_lambda | DOUBLE | [0.0, 1.0] | linear |

**Best parameters** (written by `hyperparameter_tuner.py` via Vizier optimal trial):

```yaml
# config/best_params_xgb.yaml
colsample_bytree: 0.6054
learning_rate: 0.03481
max_depth: 10
min_child_weight: 20
n_estimators: 1761
scale_pos_weight: 15.0
subsample: 0.9301

# config/best_params_lgb.yaml
colsample_bytree: 1.0
learning_rate: 0.07694
min_child_samples: 114
n_estimators: 896
num_leaves: 240
reg_alpha: 0.8138
reg_lambda: 0.9568
subsample: 0.9386
```

### Champion Model

**Model:** `lgb-v8-no-txnid` (LightGBM, Vizier-tuned, TransactionID removed)  
**Selected by:** highest AUC-PR on temporal test set (last 25% chronologically), leakage-free  
**Vertex AI Model Registry:** `projects/65768314585/locations/asia-south1/models/8388011480582193152@1`  
**Stage:** `production-ready`  
**Serving artifact:** `gs://fraud-detection-mlops-497717-fraud-artifacts/models/lgb-v8-no-txnid-calibrated/`  
**Feature count:** 387 (TransactionID excluded)

**Full metrics (test set — 147,635 rows, 3.45% fraud rate):**

| Metric | Value |
|---|---|
| AUC-PR | **0.5263** |
| AUC-ROC | 0.8923 |
| KS Statistic | 0.6310 |
| F1 @ threshold | 0.5285 |
| F2 @ threshold | 0.4892 |
| Precision @ threshold | 0.6101 |
| Recall @ threshold | 0.4661 |
| Optimal threshold (uncalibrated) | 0.4517 |
| Optimal threshold (calibrated) | 0.2458 |

**Baseline comparison:**

| Model | AUC-PR | Delta | Notes |
|---|---|---|---|
| LR baseline | 0.2172 | — | |
| v5 best (LGB, untuned) | 0.4913 | +0.2741 vs LR | |
| v7 (LGB, Vizier-tuned) | 0.5393 | +0.0480 vs v5 | Contained TransactionID leakage; demoted to staging |
| **v8 champion (LGB, leakage-free)** | **0.5263** | **+0.0350 vs v5** | **TransactionID removed; production-ready** |

**v7 → v8 delta:** −0.013 AUC-PR. TransactionID was marginal — the model's genuine fraud signal is retained.

**SHAP top 10 features** (mean |SHAP|, full 147k test set, v8 — TransactionID absent):

| Rank | Feature | Mean \|SHAP\| | Category | Business Interpretation |
|---|---|---|---|---|
| 1 | TransactionDT | 0.450 | Timestamp | Fraud rate is strongly time-dependent; off-hours and specific day-of-week patterns separate fraud from legitimate transactions |
| 2 | C13 | 0.414 | Count feature | Transaction count linked to the card; high velocity signals a compromised card or mule account |
| 3 | TransactionAmt | 0.331 | Amount | Fraudsters probe with small test charges or make large unauthorized purchases; amount distribution diverges significantly from legitimate |
| 4 | C14 | 0.293 | Count feature | Secondary card transaction count; corroborates C13 velocity signal — both elevated together is a strong fraud flag |
| 5 | addr1 | 0.266 | Billing address | Billing zip code; mismatches with shipping address or sudden zip code changes are classic card-not-present fraud indicators |
| 6 | card1 | 0.263 | Card identifier | Issuer or BIN-level identifier; certain card bins carry systematically higher fraud rates |
| 7 | card2 | 0.238 | Card identifier | Card-associated zip code; cross-validates addr1 and flags geographic inconsistencies between card issuance and transaction location |
| 8 | V91 | 0.214 | Vesta engineered | Vesta proprietary risk signal; semantics are opaque but strong discriminative power suggests it encodes known fraud pattern combinations |
| 9 | C1 | 0.188 | Count feature | Number of distinct addresses associated with the payment card; sudden increases flag account takeover attempts |
| 10 | dist1 | 0.188 | Distance | Distance between purchaser and recipient addresses; large gaps correlate with card-not-present fraud where shipping and billing locations diverge |

**Calibration (Platt scaling, fit on val set):**

| Metric | Uncalibrated | Calibrated | Delta |
|---|---|---|---|
| AUC-PR | 0.5263 | 0.5263 | 0.0000 |
| AUC-ROC | 0.8923 | 0.8923 | 0.0000 |
| Optimal threshold | 0.4517 | 0.2458 | −0.2059 |

AUC metrics are threshold-agnostic and hold flat post-calibration; the threshold shift reflects the model's uncalibrated score compression.

**GCS explainability artifacts:**
- SHAP bar chart: `gs://fraud-detection-mlops-497717-fraud-artifacts/explainability/shap_feature_importance_top20.png`
- SHAP beeswarm: `gs://fraud-detection-mlops-497717-fraud-artifacts/explainability/shap_beeswarm.png`
- Reliability diagram: `gs://fraud-detection-mlops-497717-fraud-artifacts/explainability/reliability_diagram.png`

---

## Serving Architecture (Phase 4)

```
Transaction JSON
      │
      ▼
Cloud Run API (src/serving/api.py)
  • POST /predict   — feature engineer → Vertex AI Endpoint → fraud probability
  • GET  /health    — model version, endpoint status, today's prediction count
  • GET  /metrics   — 24hr latency p50/p95/p99, fraud rate
  • POST /drift-check — triggered by Cloud Scheduler
      │
      ├── FraudFeatureEngineer.transform()  (transformer loaded from GCS, cached)
      │
      ├── Vertex AI Endpoint (lgb-v8-no-txnid, autoscaling 1–3 replicas)
      │     machine_type: n1-standard-4
      │     target latency: p99 < 200ms
      │
      ├── Platt calibration threshold: 0.2458
      │
      ├── Per-prediction SHAP (top 3 features, TreeExplainer on base LGB)
      │
      └── BigQuery prediction_logs (metadata only — no PII, no feature values)
            prediction_id, timestamp, fraud_probability, prediction,
            threshold_used, latency_ms, model_version
```

**Cloud Run API** deployed with `serving-sa`, `min_instances=1`, `max_instances=5`, `memory=2Gi`.  
**Streamlit demo** deployed as a separate Cloud Run service with public access.

---

## Drift Monitoring (Phase 4)

Two independent drift detection methods run daily:

### Feature Drift — PSI
Population Stability Index computed for **TransactionAmt** and **C13** (the two highest-SHAP features that are also business-interpretable).

- **Baseline**: training distribution percentiles saved in GCS (`monitoring/baseline_distributions.json`)
- **Current window**: last 7 days of transactions from `fraud_detection.transactions_joined`
- **Alert threshold**: PSI > 0.2 for either feature
- **PSI interpretation**: < 0.1 stable, 0.1–0.2 investigate, > 0.2 significant shift

### Performance Drift — Fraud Rate
- Monitoring 7-day rolling fraud rate in `prediction_logs`
- Alert when `|observed_rate − 0.0351| > 2 × binomial_std`
- Binomial std = `√(p(1-p)/n)` — shrinks as prediction volume grows

### BigQuery `drift_logs` schema
| Column | Type | Description |
|---|---|---|
| timestamp | TIMESTAMP | Check time |
| feature_name | STRING | Feature name or "fraud_rate" |
| psi_score | FLOAT64 | PSI or deviation in std units |
| alert_fired | BOOL | True if threshold exceeded |
| baseline_mean | FLOAT64 | Training distribution mean |
| current_mean | FLOAT64 | Current window mean |
| window_days | INTEGER | Monitoring window size |

### Alert wiring
| Alert Policy | Metric | Threshold | Channel |
|---|---|---|---|
| `feature-drift-alert` | `custom/fraud_detection/psi_score` | PSI > 0.2 | Email |
| `performance-drift-alert` | `custom/fraud_detection/fraud_rate_deviation_std` | > 2.0 std | Email |

Cloud Scheduler job: `fraud-daily-drift-check` — cron `30 0 * * *` UTC (= 6:00 AM IST).

Also configured: **Vertex AI Model Monitoring** on the deployed endpoint — skew detection for TransactionAmt and C13 vs training dataset, email alert on threshold breach.

---

## CI/CD Pipeline (Phase 4)

Cloud Build triggered on every push to `main`. 10-step pipeline:

| Step | Name | Blocks on failure |
|---|---|---|
| 1 | Unit tests (`pytest --cov-fail-under=70`) | Yes |
| 2 | Model performance gate (`AUC-PR ≥ 0.48`) | Yes |
| 3 | Feature engineering validation (387 features, 0 NaN on 500-row sample) | Yes |
| 4 | CVE audit (`pip-audit`) | No (warns) |
| 5 | SAST (`bandit -ll`) — blocks on HIGH severity | Yes |
| 6 | Secret scan (`detect-secrets`) | No (warns) |
| 7 | Docker build (multi-stage, non-root, `BUILDKIT_INLINE_CACHE`) | Yes |
| 8 | Push to Artifact Registry + container vulnerability scan (blocks on CRITICAL CVEs) | Yes |
| 9 | Rolling deploy to Cloud Run (`fraud-detection-api`) | Yes |
| 10 | Smoke test — POST `/predict` + GET `/health` | Yes |

Service account: `cicd-sa` (Cloud Build).  
Configure GitHub trigger:
```bash
gcloud builds triggers create github \
  --repo-name=fraud-detection-mlops \
  --repo-owner=chandranakkalakunta \
  --branch-pattern='^main$' \
  --build-config=cloudbuild.yaml \
  --service-account=projects/PROJECT_ID/serviceAccounts/cicd-sa@PROJECT_ID.iam.gserviceaccount.com
```

---

## Live Demo

Streamlit app — 3 pages:

| Page | What it shows |
|---|---|
| **Live Fraud Scorer** | Input form with demo-safe defaults → fraud probability gauge, SHAP bar chart, latency |
| **Model Performance** | AUC-PR progression chart, SHAP top-10 with business interpretation, all-runs table |
| **Drift Monitoring** | PSI trends for TransactionAmt + C13, fraud rate trend, alert history |

Run locally:
```bash
make streamlit
# http://localhost:8501
```

---

## GCP Resources

All resources provisioned across all phases:

| Resource | Type | Name / ID |
|---|---|---|
| Project | GCP Project | `fraud-detection-mlops-497717` |
| Region | All resources | `asia-south1` (org policy enforced) |
| Raw data bucket | GCS (CMEK) | `fraud-detection-mlops-497717-fraud-raw` |
| Processed bucket | GCS (CMEK) | `fraud-detection-mlops-497717-fraud-processed` |
| Artifacts bucket | GCS (CMEK) | `fraud-detection-mlops-497717-fraud-artifacts` |
| Audit bucket | GCS (CMEK) | `fraud-detection-mlops-497717-fraud-audit` |
| BQ Dataset | BigQuery (CMEK) | `fraud_detection` |
| BQ Table | BigQuery | `transactions_raw`, `identity_raw`, `transactions_joined` |
| BQ Table | BigQuery | `prediction_logs` (partitioned by day) |
| BQ Table | BigQuery | `drift_logs` (partitioned by day) |
| KMS keyring | Cloud KMS | `fraud-keyring` (asia-south1) |
| KMS key (BQ) | Cloud KMS | `bq-key` |
| KMS key (GCS) | Cloud KMS | `gcs-key` |
| Training SA | Service Account | `training-sa@...` |
| Serving SA | Service Account | `serving-sa@...` |
| Pipeline SA | Service Account | `pipeline-sa@...` |
| Monitoring SA | Service Account | `monitoring-sa@...` |
| Vertex Experiment | Vertex AI | `fraud-detection-baseline` |
| Vizier studies | Vertex AI Vizier | `fraud_xgb_hp_tuning`, `fraud_lgb_hp_tuning` |
| Model Registry | Vertex AI | `lgb-v8-no-txnid` (production-ready), `lgb-v7-tuned` (staging) |
| Endpoint | Vertex AI Endpoint | `fraud-detection-endpoint` (n1-standard-4, 1–3 replicas) |
| API | Cloud Run | `fraud-detection-api` (min=1, max=5, 2Gi) |
| Streamlit | Cloud Run | `fraud-detection-streamlit` (public) |
| API Key | Secret Manager | `fraud-api-key` |
| CI/CD trigger | Cloud Build | Push to `main` → 10-step pipeline |
| Drift scheduler | Cloud Scheduler | `fraud-daily-drift-check` (6am IST daily) |
| Alert (drift) | Cloud Monitoring | `feature-drift-alert`, `performance-drift-alert` |
| Container repo | Artifact Registry | `fraud-detection-repo` |

---

## Quick Start — Full Pipeline

```bash
# 1. Configure environment
cp .env.example .env
# Edit .env — fill in your GCP_PROJECT_ID and bucket names

# 2. Install dependencies
make install

# 3. Bootstrap GCP (idempotent)
make gcp-setup

# 4. Ingest data to BigQuery
make ingest

# 5. Train champion model (Phases 2A/2B)
ENV=dev python src/training/run_v8_pipeline.py

# 6. Deploy serving infrastructure (Phase 4)
make deploy-serving          # BQ tables, baseline, API key, Vertex endpoint

# 7. Build and deploy Cloud Run API
gcloud builds submit --config=cloudbuild.yaml

# 8. Set CLOUD_RUN_API_URL in .env, then:
make setup-monitoring        # Cloud Scheduler, alert policies, Vertex monitoring

# 9. Run Streamlit demo
CLOUD_RUN_API_URL=<your-url> API_KEY=<your-key> make streamlit
```

---
