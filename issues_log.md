# Issues Log — fraud-detection-mlops

All bugs, misconfigurations, and schema errors encountered and resolved during development.

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
**Commit:** `726b447`

---

## 14. Python module naming — `scripts/02_data_ingestion.py` not importable

**When:** Writing unit tests for ingestion logic.  
**Cause:** Python module names cannot start with a digit. A file named `02_data_ingestion.py` cannot be imported with `import 02_data_ingestion` — the interpreter rejects names beginning with a numeral. All logic co-located in the script was therefore untestable.  
**Fix:** Extracted all core logic (schemas, `load_csv_to_bq`, `validate_table`, `create_joined_table`, `validate_join_integrity`, `gcs_uri`) into `src/ingestion/loader.py` as a proper importable module. `scripts/02_data_ingestion.py` became a thin entry point that imports from `src.ingestion.loader`. Tests import `src.ingestion.loader` directly with no workarounds.  
**Commit:** `1cc37cd`

---

## 15. Python 3.13 incompatibility — pinned packages require Python ≤ 3.12

**When:** First `pip install -r requirements.txt` attempt in the initial (wrong) venv, which used Python 3.13 from the enterprise-hr-rag project.  
**Cause:** Pinned versions in `requirements.txt` — `numpy==1.26.4`, `scikit-learn==1.4.2`, `pandas==2.2.2` — have no pre-built wheels for Python 3.13 and their C extensions do not compile cleanly against it. The error manifested as build failures and import errors during installation.  
**Fix:** Installed Python 3.11 via Homebrew (`brew install python@3.11`). Created the project venv explicitly with `/opt/homebrew/opt/python@3.11/bin/python3.11 -m venv venv`. Python 3.11 matches the version pinned in the project's `Dockerfile`, ensuring local and container environments are consistent.  
**Commit:** `31100ff`

---

## Phase 2A — Feature Engineering & Diagnostic Runs

---

## 16. LOO target encoding collapse — `best_iter=15`, `card4_te` SHAP=0.934

**When:** Phase 2A ablation run xgb-v4-no-indicators.  
**Cause:** Leave-one-out target encoding for `card4`, `card6`, `P_emaildomain`, `R_emaildomain` created near-perfect fraud-rate proxies. `card4_te` had SHAP=0.934 — the model saturated on this single feature within 15 trees and immediately overfit. `best_iter=15` at `n_estimators=500` meant 97% of training capacity was wasted.  
**Fix:** Removed all target-encoded columns (`target_encode_cols=[]`). AUC-PR recovered from 0.174 (v4, TE present) to 0.4895 (v5, TE removed). Declared permanently dropped in `FraudFeatureEngineer`.  
**Commit:** `b56fa89`

---

## 17. V-column `_was_missing` indicators crowding `colsample_bytree` budget

**When:** Phase 2A ablation Run C (full engineer with 253 indicators).  
**Cause:** 253 binary `_was_missing` indicators were created for V-columns with >5% null rate. Of these, 206 had 80–100% null rate — meaning the indicator was 1 for nearly every row, making it near-constant and informationally worthless. These 253 features occupied ~49% of the `colsample_bytree=0.8` budget per tree, crowding out genuinely predictive features. AUC-PR collapsed from 0.4871 (Run B, no indicators) to 0.1228 (Run C, 253 indicators) — a delta of −0.364.  
**Fix:** Removed all V-column indicators permanently. V-columns are now median-imputed only. The threshold-based partial fix (threshold 0.05→0.80, v3) still left 206 near-constant indicators and only recovered to AUC-PR=0.201 — insufficient. Full removal in v5 restored AUC-PR to 0.4913.  
**Commit:** `b56fa89`

---

## 18. `eval_metric='logloss'` incompatible with `scale_pos_weight`

**When:** Phase 2A/2B XGBoost runs v2 and v4.  
**Cause:** `scale_pos_weight=27` upweights the fraud class during training, pushing predicted probabilities toward high values for most rows. Unweighted validation logloss treats these inflated probabilities as errors on the 96.5% legitimate class, spiking immediately at tree 1. Early stopping fires at `best_iter=0`, yielding AUC-ROC=0.4984 (random).  
**Fix:** Changed `eval_metric='auc'` throughout. AUC is a ranking metric — threshold-free and immune to calibration bias from class weighting. AUC-PR is still computed from `predict_proba()` post-training as the primary reporting metric.  
**Commit:** `b56fa89`

---

## Phase 2B — Hyperparameter Tuning (Vertex AI Vizier)

---

## 19. Vizier API rejects study `display_name` containing hyphens

**When:** First `run_tuning("xgboost")` call to Vertex AI Vizier.  
**Cause:** Vizier study display names must match `[a-z0-9_]+` — hyphens are rejected with `400 InvalidArgument`. Initial study names used hyphens: `fraud-xgb-hp-tuning`.  
**Fix:** Changed to underscores throughout: `fraud_xgb_hp_tuning`, `fraud_lgb_hp_tuning`. Updated `.env` and `.env.example`.  
**Commit:** `b56fa89`

---

## 20. `AttributeError: 'float' object has no attribute 'number_value'` — Vizier trial parameters

**When:** First trial parameter extraction in `VizierHPTuner._extract_params()`.  
**Cause:** Google Cloud documentation and older SDK examples show `trial.parameters[i].value.number_value` for accessing numeric trial values. In `google-cloud-aiplatform==1.59.0`, `Trial.Parameter.value` is already a Python `float` — not a protobuf `Value` wrapper. Calling `.number_value` on a float raises `AttributeError`.  
**Fix:** Changed to `float(p.value)` directly. Integer parameters round-tripped through `int(round(value))`.  
**Commit:** `b56fa89`

---

## 21. `AttributeError: GAUSSIAN_PROCESS_BANDIT` — missing enum in aiplatform v1.59.0

**When:** `VizierHPTuner._get_or_create_study()` — setting Vizier algorithm.  
**Cause:** `StudySpec.Algorithm.GAUSSIAN_PROCESS_BANDIT` does not exist in `google-cloud-aiplatform==1.59.0`. The enum variant was added in a later SDK version.  
**Fix:** Changed to `StudySpec.Algorithm.ALGORITHM_UNSPECIFIED`. The Vizier backend defaults to Gaussian Process Bandit when the algorithm is unspecified — confirmed in GCP documentation.  
**Commit:** `b56fa89`

---

## 22. Stale `ACTIVE`/`REQUESTED` Vizier trials blocking `suggest_trials`

**When:** Resuming a Vizier study after an interrupted tuning run.  
**Cause:** Vizier enforces a cap on concurrent active trials. Trials left in `ACTIVE` or `REQUESTED` state from a previous (interrupted) run count against this cap. `suggest_trials` either returns 0 trials or hangs waiting for capacity to free.  
**Fix:** Added `_cleanup_pending_trials()` method to `VizierHPTuner`. On study reuse, it iterates all trials and marks any `ACTIVE`/`REQUESTED` ones as infeasible before requesting new suggestions. Called automatically in `_get_or_create_study()` when an existing study is found.  
**Commit:** `b56fa89`

---

## Phase 2B — Champion Pipeline & Leakage Fix

---

## 23. `TransactionID` data leakage via temporal ordering correlation

**When:** SHAP analysis on champion model `lgb-v7-tuned`.  
**Cause:** `TransactionID` is a monotonically increasing unique row identifier — lower IDs correspond to earlier transactions, which fall in the training period under the time-based split. The model learned to partially associate low IDs with the training distribution, placing `TransactionID` at SHAP rank 7 (mean |SHAP|=0.241) despite it carrying zero genuine fraud signal.  
**Diagnosis:** SHAP `TreeExplainer` on full 147k test set. TransactionID ranked above `card2`, `V91`, and `C1` — all legitimate features. Its rank 7 position was suspicious for a row identifier.  
**Fix:** Added `EXCLUDED_FEATURES = ["TransactionID"]` at module level in `src/features/engineer.py`. The `transform()` method drops all columns in `EXCLUDED_FEATURES` at the top of processing, before any imputation or feature creation, so the leakage cannot re-enter via any code path. Retrained as `lgb-v8-no-txnid` (387 features). AUC-PR: 0.5393 → 0.5263 (delta −0.013). Since |delta| < 0.02, TransactionID was marginal — the model's core signal is clean. `lgb-v7-tuned` demoted to staging; `lgb-v8-no-txnid` registered as production-ready.  
**Commit:** `64d277c`
