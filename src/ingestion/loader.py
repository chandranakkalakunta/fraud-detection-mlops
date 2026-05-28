"""
Core ingestion logic: schemas, BQ loading, validation, joining.
Called by scripts/02_data_ingestion.py and unit-tested directly.
"""

import json
from typing import Optional

from google.cloud import bigquery, storage

from src.utils.logging import get_logger

logger = get_logger(__name__)


# ─── Schema definitions ───────────────────────────────────────────────────────

TRANSACTION_SCHEMA = [
    bigquery.SchemaField("TransactionID", "INTEGER", mode="REQUIRED"),
    bigquery.SchemaField("isFraud", "INTEGER", mode="NULLABLE"),
    bigquery.SchemaField("TransactionDT", "INTEGER", mode="NULLABLE"),
    bigquery.SchemaField("TransactionAmt", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("ProductCD", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("card1", "INTEGER", mode="NULLABLE"),
    bigquery.SchemaField("card2", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("card3", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("card4", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("card5", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("card6", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("addr1", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("addr2", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("dist1", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("dist2", "FLOAT", mode="NULLABLE"),
    bigquery.SchemaField("P_emaildomain", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("R_emaildomain", "STRING", mode="NULLABLE"),
] + [
    bigquery.SchemaField(f"C{i}", "FLOAT", mode="NULLABLE") for i in range(1, 15)
] + [
    bigquery.SchemaField(f"D{i}", "FLOAT", mode="NULLABLE") for i in range(1, 16)
] + [
    bigquery.SchemaField(f"M{i}", "STRING", mode="NULLABLE") for i in range(1, 10)
] + [
    bigquery.SchemaField(f"V{i}", "FLOAT", mode="NULLABLE") for i in range(1, 340)
]

IDENTITY_SCHEMA = [
    bigquery.SchemaField("TransactionID", "INTEGER", mode="REQUIRED"),
] + [
    bigquery.SchemaField(f"id_{str(i).zfill(2)}", "STRING", mode="NULLABLE")
    for i in range(1, 12)
] + [
    bigquery.SchemaField(f"id_{str(i).zfill(2)}", "FLOAT", mode="NULLABLE")
    for i in range(12, 39)
] + [
    bigquery.SchemaField("DeviceType", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("DeviceInfo", "STRING", mode="NULLABLE"),
]


# ─── Helpers ─────────────────────────────────────────────────────────────────

def gcs_uri(bucket: str, prefix: str, filename: str) -> str:
    stripped = prefix.rstrip("/")
    path = f"{stripped}/{filename}" if stripped else filename
    return f"gs://{bucket}/{path}"


def load_csv_to_bq(
    bq_client: bigquery.Client,
    source_uri: str,
    table_ref: str,
    schema: list,
    config: dict,
) -> bigquery.LoadJob:
    job_config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.CSV,
        skip_leading_rows=1,
        schema=schema,
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        null_marker="",
        allow_quoted_newlines=True,
        destination_encryption_configuration=bigquery.EncryptionConfiguration(
            kms_key_name=config["bigquery"]["kms_key_name"]
        ),
    )
    logger.info("starting_bq_load", extra={"source": source_uri, "destination": table_ref})
    job = bq_client.load_table_from_uri(source_uri, table_ref, job_config=job_config)
    job.result()
    table = bq_client.get_table(table_ref)
    logger.info(
        "bq_load_complete",
        extra={"table": table_ref, "rows": table.num_rows, "bytes": table.num_bytes},
    )
    return job


def validate_table(
    bq_client: bigquery.Client,
    table_name: str,
    min_rows: int,
    label: str,
    config: dict,
) -> dict:
    """Row count + column count validation. Returns a results dict."""
    project = config["gcp"]["project_id"]
    dataset = config["bigquery"]["dataset"]
    full_ref = f"`{project}.{dataset}.{table_name}`"

    row_count_query = f"SELECT COUNT(*) as cnt FROM {full_ref}"
    row_count = list(bq_client.query(row_count_query).result())[0]["cnt"]

    column_count_query = (
        f"SELECT COUNT(*) as cnt FROM `{project}.{dataset}`.INFORMATION_SCHEMA.COLUMNS "
        f"WHERE table_name = '{table_name}'"
    )
    column_count = list(bq_client.query(column_count_query).result())[0]["cnt"]

    validation = {
        "table": table_name,
        "label": label,
        "row_count": row_count,
        "column_count": column_count,
        "min_rows_required": min_rows,
        "row_count_pass": row_count >= min_rows,
    }

    if not validation["row_count_pass"]:
        logger.error(
            "validation_failed_row_count",
            extra={"table": table_name, "expected_min": min_rows, "actual": row_count},
        )
    else:
        logger.info("validation_passed", extra=validation)

    return validation


def create_joined_table(bq_client: bigquery.Client, config: dict) -> None:
    project = config["gcp"]["project_id"]
    dataset = config["bigquery"]["dataset"]
    tables = config["bigquery"]["tables"]
    joined_table = f"`{project}.{dataset}.{tables['transactions_joined']}`"
    kms_key = config["bigquery"]["kms_key_name"]

    identity_cols = ", ".join(
        f"i.{f.name}" for f in IDENTITY_SCHEMA if f.name != "TransactionID"
    )

    join_query = f"""
        CREATE OR REPLACE TABLE {joined_table}
        OPTIONS (
          kms_key_name = '{kms_key}',
          description = 'IEEE-CIS Fraud Detection — transactions joined with identity'
        ) AS
        SELECT
          t.*,
          {identity_cols}
        FROM `{project}.{dataset}.{tables['transactions_raw']}` AS t
        LEFT JOIN `{project}.{dataset}.{tables['identity_raw']}` AS i
          USING (TransactionID)
    """

    logger.info("creating_joined_table", extra={"destination": joined_table})
    bq_client.query(join_query).result()

    joined = bq_client.get_table(f"{project}.{dataset}.{tables['transactions_joined']}")
    logger.info(
        "joined_table_created",
        extra={"table": tables["transactions_joined"], "rows": joined.num_rows},
    )


def validate_join_integrity(bq_client: bigquery.Client, config: dict) -> None:
    project = config["gcp"]["project_id"]
    dataset = config["bigquery"]["dataset"]
    tables = config["bigquery"]["tables"]

    integrity_query = f"""
        SELECT
          COUNT(*) AS total_joined,
          COUNTIF(TransactionID IS NULL) AS null_txn_ids,
          COUNTIF(isFraud IS NULL) AS null_labels,
          ROUND(COUNTIF(isFraud = 1) / COUNT(*) * 100, 4) AS fraud_rate_pct,
          COUNT(DISTINCT TransactionID) AS unique_txn_ids
        FROM `{project}.{dataset}.{tables['transactions_joined']}`
    """
    rows = list(bq_client.query(integrity_query).result())
    row = rows[0]
    stats = {
        "total_joined": row["total_joined"],
        "null_txn_ids": row["null_txn_ids"],
        "null_labels": row["null_labels"],
        "fraud_rate_pct": row["fraud_rate_pct"],
        "unique_txn_ids": row["unique_txn_ids"],
    }

    fraud_rate = stats["fraud_rate_pct"]
    val = config["data_validation"]

    checks = {
        "no_null_transaction_ids": stats["null_txn_ids"] == 0,
        "no_null_labels": stats["null_labels"] == 0,
        "fraud_rate_in_range": (
            val["expected_fraud_rate_min"] * 100
            <= fraud_rate
            <= val["expected_fraud_rate_max"] * 100
        ),
        "no_duplicate_transaction_ids": stats["unique_txn_ids"] == stats["total_joined"],
    }

    failed = [k for k, v in checks.items() if not v]
    if failed:
        logger.error("join_integrity_checks_failed", extra={"failed": failed, "stats": stats})
        raise RuntimeError(f"Join integrity failed: {failed}")

    logger.info("join_integrity_passed", extra={"checks": checks, "stats": stats})
