"""
Ingest IEEE-CIS Fraud Detection CSVs from GCS into BigQuery.

Expected GCS layout:
  gs://<raw_bucket>/ieee-cis/train_transaction.csv
  gs://<raw_bucket>/ieee-cis/train_identity.csv

Outputs:
  <dataset>.transactions_raw     (train_transaction.csv)
  <dataset>.identity_raw         (train_identity.csv)
  <dataset>.transactions_joined  (inner join on TransactionID)

Idempotent: re-running overwrites existing tables (WRITE_TRUNCATE + CREATE OR REPLACE).
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from google.cloud import bigquery, storage

from src.ingestion.loader import (
    IDENTITY_SCHEMA,
    TRANSACTION_SCHEMA,
    create_joined_table,
    gcs_uri,
    load_csv_to_bq,
    validate_join_integrity,
    validate_table,
)
from src.utils.config import load_config
from src.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    env = os.getenv("ENV", "dev")
    config = load_config(env)

    project = config["gcp"]["project_id"]
    dataset = config["bigquery"]["dataset"]
    raw_bucket = config["storage"]["raw_bucket"]
    raw_prefix = config["storage"]["raw_data_prefix"]
    tables = config["bigquery"]["tables"]
    val = config["data_validation"]

    bq_client = bigquery.Client(project=project)
    results: dict = {"tables": {}, "join_integrity": None}

    # 1. Load transactions_raw
    txn_uri = gcs_uri(raw_bucket, raw_prefix, "train_transaction.csv")
    txn_table_ref = f"{project}.{dataset}.{tables['transactions_raw']}"
    load_csv_to_bq(bq_client, txn_uri, txn_table_ref, TRANSACTION_SCHEMA, config)
    results["tables"]["transactions_raw"] = validate_table(
        bq_client, tables["transactions_raw"],
        val["min_rows_transactions"], "transactions_raw", config,
    )

    # 2. Load identity_raw
    id_uri = gcs_uri(raw_bucket, raw_prefix, "train_identity.csv")
    id_table_ref = f"{project}.{dataset}.{tables['identity_raw']}"
    load_csv_to_bq(bq_client, id_uri, id_table_ref, IDENTITY_SCHEMA, config)
    results["tables"]["identity_raw"] = validate_table(
        bq_client, tables["identity_raw"],
        val["min_rows_identity"], "identity_raw", config,
    )

    # 3. Join into transactions_joined
    create_joined_table(bq_client, config)

    # 4. Validate join integrity
    validate_join_integrity(bq_client, config)
    results["join_integrity"] = "passed"

    logger.info("ingestion_complete", extra={"results": results})
    print(json.dumps({"event": "ingestion_complete", "results": results}, indent=2))


if __name__ == "__main__":
    main()
