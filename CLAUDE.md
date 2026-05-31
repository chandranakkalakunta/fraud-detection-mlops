# fraud-detection-mlops/CLAUDE.md
# Project-specific additions — extends ~/.claude/CLAUDE.md

## PROJECT
- GCP Project: fraud-detection-mlops-497717
- Region: asia-south1
- Service account: pipeline-sa@fraud-detection-mlops-497717.iam.gserviceaccount.com
- Champion model: lgb-v8-no-txnid, AUC-PR gate: 0.48

## REQUIRED IAM ROLES FOR PIPELINE-SA
- logging.logWriter
- storage.objectAdmin
- artifactregistry.writer
- run.admin, run.invoker
- secretmanager.secretAccessor
- iam.serviceAccountTokenCreator
