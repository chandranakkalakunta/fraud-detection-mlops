"""
Phase 2A — Feature Engineering Pipeline

Loads transactions_joined from BigQuery, applies FraudFeatureEngineer
on a time-based 75/25 split, saves fitted transformers to GCS.

Usage:
    python scripts/03_feature_engineering.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
load_dotenv()

from google.cloud import bigquery

from src.features.engineer import build_from_config
from src.utils.config import load_config
from src.utils.logging import get_logger

logger = get_logger(__name__)

TARGET = "isFraud"

NUMERIC_FEATURES = (
    ["TransactionAmt", "card1", "card2", "card3", "card5",
     "addr1", "addr2", "dist1", "dist2"]
    + [f"C{i}" for i in range(1, 15)]
    + [f"D{i}" for i in range(1, 16)]
)

CATEGORICAL_FEATURES = [
    "ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain",
    "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9",
]

V_FEATURES = [f"V{i}" for i in range(1, 340)]


def load_data(config: dict):
    project = config["gcp"]["project_id"]
    dataset = config["bigquery"]["dataset"]
    table = config["bigquery"]["tables"]["transactions_joined"]

    all_cols = (
        ["TransactionDT"]
        + NUMERIC_FEATURES
        + CATEGORICAL_FEATURES
        + V_FEATURES
        + [TARGET]
    )
    col_list = ", ".join(all_cols)

    query = f"""
        SELECT {col_list}
        FROM `{project}.{dataset}.{table}`
        WHERE {TARGET} IS NOT NULL
    """

    print(f"Loading from BigQuery: {project}.{dataset}.{table} ...")
    client = bigquery.Client(project=project)
    df = client.query(query).to_dataframe(progress_bar_type=None)
    print(f"  Loaded {len(df):,} rows × {len(df.columns)} columns")
    print(f"  Fraud rate: {df[TARGET].mean():.4%}")
    return df


def main() -> None:
    env = os.getenv("ENV", "dev")
    config = load_config(env)

    # ── Load ──────────────────────────────────────────────────────────────────
    df = load_data(config)

    # ── Time-based 75/25 split ────────────────────────────────────────────────
    df_sorted = df.sort_values("TransactionDT").reset_index(drop=True)
    split = int(len(df_sorted) * 0.75)

    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES + V_FEATURES + ["TransactionDT"]
    X = df_sorted[feature_cols]
    y = df_sorted[TARGET].astype(int)

    X_train, X_test = X.iloc[:split].reset_index(drop=True), X.iloc[split:].reset_index(drop=True)
    y_train, y_test = y.iloc[:split].reset_index(drop=True), y.iloc[split:].reset_index(drop=True)

    print(f"\nTime-based split:")
    print(f"  Train: {len(X_train):,} rows  (fraud rate {y_train.mean():.4%})")
    print(f"  Test : {len(X_test):,} rows  (fraud rate {y_test.mean():.4%})")

    # ── Build and fit engineer ─────────────────────────────────────────────────
    engineer = build_from_config(config)

    print("\nFitting engineer on training data ...")
    X_train_eng = engineer.fit_transform(X_train, y_train)

    print("Transforming test data ...")
    X_test_eng = engineer.transform(X_test)

    # ── Report ────────────────────────────────────────────────────────────────
    was_missing_cols = [c for c in X_train_eng.columns if c.endswith("_was_missing")]
    # NaN breakdown: engineered (should be 0) vs pass-through (raw nulls remain)
    engineered_cols = (
        ["time_of_day", "day_of_week", "hour_of_day",
         "log_transaction_amt", "amt_to_d1_ratio", "is_round_amt"]
        + [f"{c}_card_norm" for c in ["D1", "D4", "D10"]]
        + was_missing_cols
        + [f"{c}_te" for c in ["card4", "card6", "P_emaildomain", "R_emaildomain"]]
    )
    engineered_cols = [c for c in engineered_cols if c in X_train_eng.columns]
    passthrough_cols = [c for c in X_train_eng.columns if c not in engineered_cols]

    nan_eng_train = int(X_train_eng[engineered_cols].isnull().sum().sum())
    nan_pass_train = int(X_train_eng[passthrough_cols].isnull().sum().sum())
    nan_eng_test = int(X_test_eng[engineered_cols].isnull().sum().sum())
    nan_pass_test = int(X_test_eng[passthrough_cols].isnull().sum().sum())

    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING REPORT")
    print("=" * 60)
    print(f"  Input feature columns          : {len(feature_cols)}")
    print(f"  Output features (train)        : {X_train_eng.shape[1]}")
    print(f"  Output features (test)         : {X_test_eng.shape[1]}")
    print(f"  _was_missing indicators created: {len(was_missing_cols)}")
    print(f"  V-columns with >5%% nulls      : {len(engineer._high_null_v_cols)}")
    print(f"  NaN in engineered cols (train) : {nan_eng_train}  ← must be 0")
    print(f"  NaN in engineered cols (test)  : {nan_eng_test}  ← must be 0")
    print(f"  NaN in pass-through cols(train): {nan_pass_train:,}  (raw C/D/M/addr nulls)")
    print(f"  NaN in pass-through cols(test) : {nan_pass_test:,}  (raw C/D/M/addr nulls)")
    print(f"  Global fraud rate (train)      : {engineer._global_mean:.4%}")

    if was_missing_cols:
        print(f"\n  First 10 _was_missing columns:")
        for c in sorted(was_missing_cols)[:10]:
            print(f"    {c}")
        if len(was_missing_cols) > 10:
            print(f"    ... and {len(was_missing_cols) - 10} more")

    # Show D-norm columns (quick sanity)
    d_norm_cols = [c for c in X_train_eng.columns if c.endswith("_card_norm")]
    print(f"\n  D-norm columns: {d_norm_cols}")

    # Show time feature columns
    time_cols = ["time_of_day", "day_of_week", "hour_of_day"]
    print(f"  Time feature ranges (train):")
    for col in time_cols:
        print(f"    {col}: [{X_train_eng[col].min()}, {X_train_eng[col].max()}]")

    # ── Save to GCS ───────────────────────────────────────────────────────────
    print("\nSaving fitted transformer to GCS ...")
    gcs_uri = engineer.save_to_gcs(config)
    print(f"\n  Transformer saved to: {gcs_uri}")
    print("=" * 60)


if __name__ == "__main__":
    main()
