# Issues Log — fraud-detection-mlops

All bugs, misconfigurations, and schema errors encountered and resolved during Phase 1 setup.

---

## 1. `ModuleNotFoundError: No module named 'pythonjsonlogger'`

**When:** First `pytest` run.  
**Cause:** `python-json-logger` not installed; project had no dedicated venv.  
**Fix:** Installed package. Also discovered the active Python was the HR RAG project's venv — unrelated project bleed.  
**Commit:** `6a7e251`

---

## 2. Wrong virtual environment

**When:** pytest output showed `/enterprise-hr-rag/venv/bin/python3.13`.  
**Cause:** No project-local venv existed; system/other-project venv was picked up.  
**Fix:** Created `venv/` using Python 3.11 (matching Dockerfile). Added `requirements-dev.txt` — lean install for local dev, excludes KFP/pipeline-components which cause grpcio version conflicts.  
**Commit:** `31100ff`

---

## 3. Six pytest failures after dependency fix

**When:** After installing `python-json-logger==4.1.0` (upgraded from pinned 2.0.7).  
**Root causes (3 separate bugs):**

| Bug | Detail |
|---|---|
| `gcs_uri("bucket", "", "file")` → double slash | `f".../{''}/..."` always emits a separator even with empty prefix |
| `validate_join_integrity` KeyError on all 5 integrity tests | `dict(rows[0])` returns `{}` on a MagicMock; production code relied on implicit `keys()` protocol |
| `DeprecationWarning` on `pythonjsonlogger.jsonlogger` | v4 moved formatter to `pythonjsonlogger.json` |

**Fix:** Guard `gcs_uri` against empty prefix; replace `dict(row)` with explicit field indexing; update import to `pythonjsonlogger.json.JsonFormatter`.  
**Commit:** `6a7e251`

---

## 4. `lightgbm==4.3.0` — no pre-built arm64 wheel for Python 3.11

**When:** `venv/bin/pip install -r requirements-dev.txt`.  
**Cause:** LightGBM 4.3.0 has no binary wheel for `macosx_11_0_arm64` / Python 3.11; build from source failed with a CMake 4.3.2 / lipo path error.  
**Fix:** Bumped to `lightgbm==4.6.0` in both `requirements.txt` and `requirements-dev.txt`.  
**Commit:** `31100ff`

---

## 5. `grpcio` version conflict — full `requirements.txt` install fails locally

**When:** Attempting `pip install -r requirements.txt` in the project venv.  
**Cause:** `kfp==2.7.0` and `google-cloud-pipeline-components==2.13.1` require incompatible grpcio versions; pip resolver backtracks through dozens of versions and fails.  
**Fix:** Created `requirements-dev.txt` — excludes KFP, Dataflow, and pipeline components. Those are only needed in CI/GCP containers. Local dev uses the lean file.  
**Commit:** `31100ff`

---

## 6. `bash: ${BQ_LOCATION,,}: bad substitution`

**When:** Running `scripts/01_gcp_setup.sh` on macOS.  
**Cause:** `${VAR,,}` (lowercase expansion) is a bash 4+ feature. macOS ships bash 3.2.  
**Fix:** Replaced with `"$(echo "${BQ_LOCATION}" | tr '[:upper:]' '[:lower:]')"`.  
**Commit:** `1cc37cd`

---

## 7. `bash: raw: unbound variable` — associative arrays

**When:** `scripts/01_gcp_setup.sh`, bucket creation loop.  
**Cause:** `declare -A` (associative arrays) is also bash 4+. macOS bash 3.2 doesn't support them. Two uses: bucket map and service account map.  
**Fix:** Replaced both with plain functions (`create_bucket`, `create_sa`) taking explicit arguments.  
**Commit:** `1cc37cd`

---

## 8. GCS bucket creation: KMS region mismatch (`us` vs `US_CENTRAL1`)

**When:** `gcloud storage buckets create` with CMEK.  
**Cause:** `BQ_LOCATION=US` (multi-region) → KMS keyring in `us`. GCS buckets were set to `us-central1`. GCS requires KMS key in the same region as the bucket.  
**Fix:** Changed `BQ_LOCATION` to `US_CENTRAL1` and unified KMS location derivation to `GCP_REGION`.  
**Commit:** `1cc37cd`

---

## 9. Org policy blocks `us-central1` — `constraints/gcp.resourceLocations`

**When:** GCS bucket create still failing after KMS fix.  
**Cause:** The GCP org (`chandra-idle-org`, ID `238463404898`) has an org policy restricting all resources to `in:asia-south1-locations`.  
**Diagnosis:** `gcloud resource-manager org-policies describe constraints/gcp.resourceLocations --organization=238463404898`  
**Fix:** Updated `.env`, `.env.example`, all GCP region references to `asia-south1`. Rebuilt KMS keyring in `asia-south1`.  
**Commit:** `1cc37cd`

---

## 10. BigQuery KMS SA uses project number, not project ID

**When:** `python scripts/02_data_ingestion.py` — `403 Permission denied on Cloud KMS key`.  
**Cause:** Script granted KMS access to `bq-fraud-detection-mlops-497717@bigquery-encryption.iam.gserviceaccount.com` (project ID). The actual BQ encryption SA is `bq-65768314585@bigquery-encryption.iam.gserviceaccount.com` (project **number**).  
**Fix:** Changed script to derive SA from `PROJECT_NUMBER` (fetched via `gcloud projects describe`). Granted correct SA access manually to unblock.  
**Commit:** `1cc37cd`

---

## 11. Secret Manager `--replication-policy=automatic` blocked by org policy

**When:** Secret creation in `01_gcp_setup.sh`.  
**Cause:** `automatic` replication uses global endpoints, which the org policy prohibits.  
**Fix:** Changed to `--replication-policy=user-managed --locations=asia-south1`.  
**Commit:** `1cc37cd`

---

## 12. Identity schema: `id_12`–`id_38` typed as `FLOAT`, actual values are strings

**When:** `python scripts/02_data_ingestion.py` — `400 CSV too many errors` on `identity_raw`.  
**Cause:** Schema defined `id_12`–`id_38` as `FLOAT`. Many are categorical: `id_12` = "Found"/"NotFound", `id_15` = "New"/"Found"/"NotFound", `id_30` = OS version string, `id_35`–`id_38` = "T"/"F".  
**Fix:** Changed all 38 `id_` fields to `STRING`. Type casting to numeric happens in the feature engineering layer, not at ingestion. Tests updated to match (38 STRING fields).  
**Commit:** `1cc37cd`

---

## 13. Random train/test split — data leakage

**When:** Code review of `src/training/baseline.py`.  
**Cause:** `train_test_split(..., stratify=y)` shuffles randomly — future transactions appear in training set, past transactions appear in test set. Fraud patterns evolve over time; this produces optimistically inflated scores that don't reflect production performance.  
**Fix:** Sort by `TransactionDT` ascending; use first 75% for training, last 25% for test. Also switched cross-validation from `StratifiedKFold` to `TimeSeriesSplit` for consistency.  
**Commit:** pending
