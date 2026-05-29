"""
Vertex AI Feature Store setup for fraud transaction features.

Purpose: identical feature computation at training and serving time —
eliminates training-serving skew by ensuring the same engineered features
are written once and read by both the training pipeline and the prediction
service.

Feature group: fraud_transactions
Entity type: transaction (keyed by TransactionID)
"""

import io
from typing import Optional

import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)

# Feature definitions: (feature_id, value_type, description)
# Only the engineered output features are registered — raw columns are not
# served directly; they flow through the engineer transform first.
FEATURE_DEFINITIONS: list[tuple[str, str, str]] = [
    # ── Time features ──────────────────────────────────────────────────────
    ("time_of_day",            "INT64",  "Seconds since midnight (TransactionDT % 86400)"),
    ("day_of_week",            "INT64",  "Day of week 0–6 derived from TransactionDT"),
    ("hour_of_day",            "INT64",  "Hour of day 0–23 derived from TransactionDT"),
    # ── Velocity / amount ──────────────────────────────────────────────────
    ("log_transaction_amt",    "DOUBLE", "log1p(TransactionAmt) — reduces right skew"),
    ("amt_to_d1_ratio",        "DOUBLE", "TransactionAmt / (D1 + 1) — spending relative to account age"),
    ("is_round_amt",           "INT64",  "1 if TransactionAmt is an integer, else 0"),
    # ── D-norm features ────────────────────────────────────────────────────
    ("D1_card_norm",           "DOUBLE", "D1 minus per-card1 median — normalised account age"),
    ("D4_card_norm",           "DOUBLE", "D4 minus per-card1 median — normalised D4"),
    ("D10_card_norm",          "DOUBLE", "D10 minus per-card1 median — normalised D10"),
    # ── Target-encoded categoricals ────────────────────────────────────────
    ("card4_te",               "DOUBLE", "card4 leave-one-out target encoding"),
    ("card6_te",               "DOUBLE", "card6 leave-one-out target encoding"),
    ("P_emaildomain_te",       "DOUBLE", "purchaser email domain LOO target encoding"),
    ("R_emaildomain_te",       "DOUBLE", "recipient email domain LOO target encoding"),
    # ── Pass-through numeric (after median imputation) ─────────────────────
    ("TransactionAmt",         "DOUBLE", "Raw transaction amount (after imputation)"),
    ("card1",                  "INT64",  "Card issuer identifier"),
    ("addr1",                  "DOUBLE", "Billing region code (imputed)"),
    ("addr2",                  "DOUBLE", "Billing country code (imputed)"),
    ("dist1",                  "DOUBLE", "Distance (imputed)"),
    ("dist2",                  "DOUBLE", "Distance 2 (imputed)"),
]

# Add C and D pass-through columns programmatically
for _i in range(1, 15):
    FEATURE_DEFINITIONS.append((f"C{_i}", "DOUBLE", f"Counting feature C{_i} (imputed)"))
for _i in list(range(1, 16)):
    FEATURE_DEFINITIONS.append((f"D{_i}", "DOUBLE", f"Timedelta feature D{_i} (imputed)"))


class FeatureStoreManager:
    """
    Manages the Vertex AI Feature Store lifecycle for fraud detection features.

    All GCP calls are deferred to explicit method calls so the class is safely
    instantiable in unit tests without triggering network I/O.
    """

    ENTITY_TYPE_ID = "transaction"

    def __init__(self, config: dict) -> None:
        self._project = config["gcp"]["project_id"]
        self._location = config["gcp"]["region"]
        self._featurestore_id = config["vertex"]["featurestore_id"]
        self._pipeline_sa = config["service_accounts"]["pipeline"]
        self._featurestore = None
        self._entity_type = None

    def _init_aiplatform(self) -> None:
        import google.cloud.aiplatform as aip
        aip.init(project=self._project, location=self._location)

    def setup(self, online_node_count: int = 1) -> None:
        """
        Create Feature Store and entity type if they don't already exist.
        Idempotent — safe to call on every pipeline run.
        """
        import google.cloud.aiplatform as aip
        from google.api_core.exceptions import AlreadyExists, NotFound

        self._init_aiplatform()

        # Create or fetch feature store
        try:
            self._featurestore = aip.Featurestore.create(
                featurestore_id=self._featurestore_id,
                online_store_fixed_node_count=online_node_count,
                project=self._project,
                location=self._location,
            )
            logger.info(
                "featurestore_created",
                extra={"featurestore_id": self._featurestore_id},
            )
        except AlreadyExists:
            self._featurestore = aip.Featurestore(
                featurestore_name=self._featurestore_id,
                project=self._project,
                location=self._location,
            )
            logger.info(
                "featurestore_exists",
                extra={"featurestore_id": self._featurestore_id},
            )

        # Create or fetch entity type
        try:
            self._entity_type = self._featurestore.create_entity_type(
                entity_type_id=self.ENTITY_TYPE_ID,
                description="IEEE-CIS fraud transaction — engineered features",
            )
            logger.info(
                "entity_type_created",
                extra={"entity_type_id": self.ENTITY_TYPE_ID},
            )
        except AlreadyExists:
            self._entity_type = aip.EntityType(
                entity_type_name=(
                    f"projects/{self._project}/locations/{self._location}"
                    f"/featurestores/{self._featurestore_id}"
                    f"/entityTypes/{self.ENTITY_TYPE_ID}"
                )
            )
            logger.info(
                "entity_type_exists",
                extra={"entity_type_id": self.ENTITY_TYPE_ID},
            )

        self._register_features()

    def _register_features(self) -> None:
        """Create all features on the entity type. Skips already-existing features."""
        import google.cloud.aiplatform as aip
        from google.api_core.exceptions import AlreadyExists

        if self._entity_type is None:
            raise RuntimeError("Call setup() first.")

        existing = {f.name for f in self._entity_type.list_features()}

        feature_configs: dict[str, aip.Feature.Meta] = {}
        for feature_id, value_type, description in FEATURE_DEFINITIONS:
            if feature_id not in existing:
                feature_configs[feature_id] = aip.Feature.Meta(
                    value_type=value_type,
                    description=description,
                )

        if feature_configs:
            self._entity_type.batch_create_features(feature_configs=feature_configs)
            logger.info(
                "features_registered",
                extra={"count": len(feature_configs)},
            )
        else:
            logger.info("features_already_registered", extra={"count": len(FEATURE_DEFINITIONS)})

    def ingest_features(
        self,
        df: pd.DataFrame,
        entity_id_col: str = "TransactionID",
        feature_time_col: Optional[str] = None,
    ) -> None:
        """
        Write a batch of engineered features into the feature store.

        df must contain entity_id_col plus engineered feature columns.
        feature_time_col — if provided — must be a datetime column marking
        when the features were computed.
        """
        if self._entity_type is None:
            raise RuntimeError("Call setup() before ingest_features().")

        import google.cloud.aiplatform as aip

        feature_ids = [
            fid for fid, _, _ in FEATURE_DEFINITIONS
            if fid in df.columns
        ]

        logger.info(
            "ingesting_features",
            extra={"rows": len(df), "feature_count": len(feature_ids)},
        )

        self._entity_type.write_feature_values(
            instances=df[[entity_id_col] + feature_ids].to_dict("records"),
            # entity_id must be provided under the key equal to entity_id_col
        )

        logger.info("feature_ingestion_complete", extra={"rows": len(df)})

    def serve_features(
        self,
        entity_ids: list[str],
        feature_ids: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """
        Online serving: fetch feature values for a list of entity IDs.

        entity_ids — list of TransactionID values (as strings)
        feature_ids — subset of features to fetch; None fetches all registered features
        Returns a DataFrame indexed by entity_id with one column per feature.

        This path is used by the prediction service to ensure serving features are
        identical to those seen at training time, eliminating training-serving skew.
        """
        if self._entity_type is None:
            raise RuntimeError("Call setup() before serve_features().")

        if feature_ids is None:
            feature_ids = [fid for fid, _, _ in FEATURE_DEFINITIONS]

        logger.info(
            "serving_features",
            extra={"entity_count": len(entity_ids), "feature_count": len(feature_ids)},
        )

        response = self._entity_type.read(
            entity_ids=entity_ids,
            feature_ids=feature_ids,
        )

        logger.info("feature_serving_complete", extra={"entity_count": len(entity_ids)})
        return response

    @property
    def featurestore_resource_name(self) -> str:
        if self._featurestore is None:
            raise RuntimeError("Call setup() first.")
        return self._featurestore.resource_name
