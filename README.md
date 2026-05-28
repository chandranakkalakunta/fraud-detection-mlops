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
- [ ] **Phase 2** — Feature engineering, Vertex AI Feature Store, XGBoost/LightGBM
- [ ] **Phase 3** — Vertex AI Pipelines (KFP), Experiments tracking
- [ ] **Phase 4** — Online serving, Vertex AI Endpoint, drift monitoring
- [ ] **Phase 5** — A/B testing framework, Streamlit dashboard, SHAP explainability
