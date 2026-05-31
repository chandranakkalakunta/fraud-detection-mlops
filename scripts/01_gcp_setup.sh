#!/usr/bin/env bash
# GCP project bootstrap — idempotent, safe to run multiple times.
# All values come from environment variables. Source .env before running:
#   set -a && source .env && set +a && bash scripts/01_gcp_setup.sh

set -euo pipefail

# ─── Validation ──────────────────────────────────────────────────────────────
required_vars=(
  GCP_PROJECT_ID GCP_REGION
  GCS_RAW_BUCKET GCS_PROCESSED_BUCKET GCS_ARTIFACTS_BUCKET GCS_AUDIT_BUCKET
  BQ_DATASET BQ_LOCATION
  TRAINING_SA SERVING_SA PIPELINE_SA MONITORING_SA
  AR_REPOSITORY AR_REGION
  ALERT_EMAIL
)
for var in "${required_vars[@]}"; do
  [[ -z "${!var:-}" ]] && { echo "ERROR: ${var} is not set"; exit 1; }
done

PROJECT="${GCP_PROJECT_ID}"
REGION="${GCP_REGION}"

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

# ─── APIs ─────────────────────────────────────────────────────────────────────
log "Enabling required APIs..."
gcloud services enable \
  aiplatform.googleapis.com \
  bigquery.googleapis.com \
  bigquerystorage.googleapis.com \
  dataflow.googleapis.com \
  cloudbuild.googleapis.com \
  artifactregistry.googleapis.com \
  run.googleapis.com \
  cloudscheduler.googleapis.com \
  monitoring.googleapis.com \
  logging.googleapis.com \
  secretmanager.googleapis.com \
  cloudkms.googleapis.com \
  storage.googleapis.com \
  iam.googleapis.com \
  compute.googleapis.com \
  --project="${PROJECT}" \
  --quiet
log "APIs enabled."

# ─── KMS Keyring + Key ───────────────────────────────────────────────────────
KMS_KEYRING="fraud-keyring"
KMS_BQ_KEY="bq-key"
KMS_GCS_KEY="gcs-key"
KMS_LOCATION="${GCP_REGION}"   # KMS region matches GCS/Vertex region; BQ_LOCATION must also be regional

log "Creating KMS keyring: ${KMS_KEYRING}..."
gcloud kms keyrings create "${KMS_KEYRING}" \
  --location="${KMS_LOCATION}" \
  --project="${PROJECT}" 2>/dev/null || log "Keyring already exists — skipping."

for key in "${KMS_BQ_KEY}" "${KMS_GCS_KEY}"; do
  log "Creating KMS key: ${key}..."
  gcloud kms keys create "${key}" \
    --keyring="${KMS_KEYRING}" \
    --location="${KMS_LOCATION}" \
    --purpose=encryption \
    --project="${PROJECT}" 2>/dev/null || log "Key ${key} already exists — skipping."
done

KMS_BQ_KEY_FULL="projects/${PROJECT}/locations/${KMS_LOCATION}/keyRings/${KMS_KEYRING}/cryptoKeys/${KMS_BQ_KEY}"
KMS_GCS_KEY_FULL="projects/${PROJECT}/locations/${KMS_LOCATION}/keyRings/${KMS_KEYRING}/cryptoKeys/${KMS_GCS_KEY}"

# Grant BigQuery and GCS service accounts access to KMS keys for CMEK
PROJECT_NUMBER="$(gcloud projects describe "${PROJECT}" --format='value(projectNumber)')"
BQ_SA="bq-${PROJECT_NUMBER}@bigquery-encryption.iam.gserviceaccount.com"
GCS_SA="service-${PROJECT_NUMBER}@gs-project-accounts.iam.gserviceaccount.com"

grant_kms() {
  local key="$1" member="$2"
  gcloud kms keys add-iam-policy-binding "${key}" \
    --keyring="${KMS_KEYRING}" \
    --location="${KMS_LOCATION}" \
    --member="serviceAccount:${member}" \
    --role="roles/cloudkms.cryptoKeyEncrypterDecrypter" \
    --project="${PROJECT}" \
    --condition=None 2>/dev/null || log "KMS binding for ${member} already exists — skipping."
}

log "Granting BQ SA KMS access..."
grant_kms "${KMS_BQ_KEY}" "${BQ_SA}"

log "Granting GCS SA KMS access..."
grant_kms "${KMS_GCS_KEY}" "${GCS_SA}"

# ─── GCS Buckets ─────────────────────────────────────────────────────────────
create_bucket() {
  local bucket="$1" label="$2"
  log "Creating bucket gs://${bucket} (${label})..."
  if ! gcloud storage buckets describe "gs://${bucket}" --project="${PROJECT}" &>/dev/null; then
    gcloud storage buckets create "gs://${bucket}" \
      --project="${PROJECT}" \
      --location="${GCP_REGION}" \
      --default-storage-class=STANDARD \
      --uniform-bucket-level-access \
      --default-encryption-key="${KMS_GCS_KEY_FULL}"
    log "Bucket gs://${bucket} created."
  else
    log "Bucket gs://${bucket} already exists — skipping."
  fi
}

create_bucket "${GCS_RAW_BUCKET}"       "raw"
create_bucket "${GCS_PROCESSED_BUCKET}" "processed"
create_bucket "${GCS_ARTIFACTS_BUCKET}" "artifacts"
create_bucket "${GCS_AUDIT_BUCKET}"     "audit"

# Audit bucket: 7-year retention for compliance
log "Setting retention policy on audit bucket..."
gcloud storage buckets update "gs://${GCS_AUDIT_BUCKET}" \
  --retention-period=2557d \
  --project="${PROJECT}" 2>/dev/null || true

# ─── Service Accounts ─────────────────────────────────────────────────────────
create_sa() {
  local sa_name="$1" display="$2"
  local sa_email="${sa_name}@${PROJECT}.iam.gserviceaccount.com"
  log "Creating service account: ${sa_email}..."
  gcloud iam service-accounts create "${sa_name}" \
    --display-name="${display}" \
    --project="${PROJECT}" 2>/dev/null || log "SA ${sa_name} already exists — skipping."
}

create_sa "${TRAINING_SA%%@*}"   "Vertex AI training jobs"
create_sa "${SERVING_SA%%@*}"    "Vertex AI endpoint serving"
create_sa "${PIPELINE_SA%%@*}"   "Vertex AI Pipelines orchestration"
create_sa "${MONITORING_SA%%@*}" "Model monitoring and alerting"

# ─── IAM Bindings (least-privilege) ──────────────────────────────────────────
log "Assigning IAM roles (least-privilege)..."

bind() {
  local sa_email="$1" role="$2"
  gcloud projects add-iam-policy-binding "${PROJECT}" \
    --member="serviceAccount:${sa_email}" \
    --role="${role}" \
    --condition=None \
    --quiet 2>/dev/null || true
}

# Training SA: read data, write artifacts, use Vertex AI, use BQ
TRAIN_EMAIL="${TRAINING_SA}"
bind "${TRAIN_EMAIL}" "roles/bigquery.dataViewer"
bind "${TRAIN_EMAIL}" "roles/bigquery.jobUser"
bind "${TRAIN_EMAIL}" "roles/storage.objectAdmin"
bind "${TRAIN_EMAIL}" "roles/aiplatform.user"
bind "${TRAIN_EMAIL}" "roles/cloudkms.cryptoKeyEncrypterDecrypter"

# Serving SA: read model artifacts, write predictions
SERVE_EMAIL="${SERVING_SA}"
bind "${SERVE_EMAIL}" "roles/storage.objectViewer"
bind "${SERVE_EMAIL}" "roles/aiplatform.user"
bind "${SERVE_EMAIL}" "roles/bigquery.dataEditor"
bind "${SERVE_EMAIL}" "roles/bigquery.jobUser"

# Pipeline SA: orchestrate Vertex Pipelines, CI/CD builds, and Cloud Run deployments
PIPE_EMAIL="${PIPELINE_SA}"
bind "${PIPE_EMAIL}" "roles/aiplatform.user"
bind "${PIPE_EMAIL}" "roles/storage.objectAdmin"
bind "${PIPE_EMAIL}" "roles/bigquery.dataEditor"
bind "${PIPE_EMAIL}" "roles/bigquery.jobUser"
bind "${PIPE_EMAIL}" "roles/iam.serviceAccountUser"
bind "${PIPE_EMAIL}" "roles/logging.logWriter"
bind "${PIPE_EMAIL}" "roles/artifactregistry.writer"
bind "${PIPE_EMAIL}" "roles/run.admin"
bind "${PIPE_EMAIL}" "roles/iam.serviceAccountTokenCreator"

# Pipeline SA needs to act as serving-sa when deploying Cloud Run with --service-account
gcloud iam service-accounts add-iam-policy-binding "${SERVING_SA}" \
  --member="serviceAccount:${PIPE_EMAIL}" \
  --role="roles/iam.serviceAccountUser" \
  --condition=None \
  --quiet 2>/dev/null || true

# Monitoring SA: read metrics, write alerts
MON_EMAIL="${MONITORING_SA}"
bind "${MON_EMAIL}" "roles/monitoring.metricWriter"
bind "${MON_EMAIL}" "roles/monitoring.viewer"
bind "${MON_EMAIL}" "roles/logging.logWriter"
bind "${MON_EMAIL}" "roles/aiplatform.viewer"
bind "${MON_EMAIL}" "roles/bigquery.dataViewer"
bind "${MON_EMAIL}" "roles/bigquery.jobUser"

log "IAM bindings applied."

# ─── BigQuery Dataset with CMEK ──────────────────────────────────────────────
log "Creating BigQuery dataset: ${BQ_DATASET}..."
if ! bq ls --project_id="${PROJECT}" "${BQ_DATASET}" &>/dev/null; then
  bq mk \
    --project_id="${PROJECT}" \
    --location="${BQ_LOCATION}" \
    --dataset \
    --description="IEEE-CIS Fraud Detection — raw and engineered features" \
    "${PROJECT}:${BQ_DATASET}"

  # Apply CMEK default encryption to the dataset
  bq update \
    --project_id="${PROJECT}" \
    --default_kms_key="${KMS_BQ_KEY_FULL}" \
    "${PROJECT}:${BQ_DATASET}"
  log "BigQuery dataset ${BQ_DATASET} created with CMEK."
else
  log "BigQuery dataset ${BQ_DATASET} already exists — skipping."
fi

# ─── Artifact Registry ───────────────────────────────────────────────────────
log "Creating Artifact Registry: ${AR_REPOSITORY}..."
gcloud artifacts repositories create "${AR_REPOSITORY}" \
  --repository-format=docker \
  --location="${AR_REGION}" \
  --description="Fraud detection Docker images" \
  --project="${PROJECT}" 2>/dev/null || log "Artifact Registry already exists — skipping."

# ─── Secret Manager ──────────────────────────────────────────────────────────
log "Creating Secret Manager secrets (empty placeholders)..."

create_secret_if_missing() {
  local name="$1"
  if ! gcloud secrets describe "${name}" --project="${PROJECT}" &>/dev/null; then
    gcloud secrets create "${name}" \
      --replication-policy=user-managed \
      --locations="${GCP_REGION}" \
      --project="${PROJECT}"
    log "Secret ${name} created. Add a version with your actual value."
  else
    log "Secret ${name} already exists — skipping."
  fi
}

create_secret_if_missing "kaggle-credentials"
create_secret_if_missing "bq-cmek-key-name"
create_secret_if_missing "alert-email"

# Store KMS key name as a secret for runtime reference
echo -n "${KMS_BQ_KEY_FULL}" | \
  gcloud secrets versions add "bq-cmek-key-name" \
    --data-file=- \
    --project="${PROJECT}" 2>/dev/null || true

# ─── Cloud Monitoring Alert Policy ───────────────────────────────────────────
log "GCP setup complete for project: ${PROJECT}"
log "Next steps:"
log "  1. Add Kaggle credentials to Secret Manager: kaggle-credentials"
log "  2. Upload IEEE-CIS data to gs://${GCS_RAW_BUCKET}/ieee-cis/"
log "  3. Run: python scripts/02_data_ingestion.py"
