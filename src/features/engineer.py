"""
Feature engineering for IEEE-CIS Fraud Detection.

All transformers fit on training data only — no leakage.
Fitted state serialised to GCS as a single joblib artifact.
"""

import io
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)

# V1–V339 as declared in TRANSACTION_SCHEMA
ALL_V_COLS: list[str] = [f"V{i}" for i in range(1, 340)]

# Non-V numeric pass-through columns requiring explicit median imputation.
# Covers the IEEE-CIS columns that can have meaningful null rates.
# D-norm cols (D1, D4, D10) are also in this list; their per-card normalisation
# creates a *new* feature while the raw column is imputed here.
_PASSTHROUGH_IMPUTE_COLS: list[str] = (
    ["TransactionAmt", "card1", "card2", "card3", "card5"]
    + [f"C{i}" for i in range(1, 15)]   # C1-C14
    + [f"D{i}" for i in range(1, 16)]   # D1-D15 all (incl. d_norm cols D1, D4, D10)
    + ["addr1", "addr2", "dist1", "dist2"]
)

# ─── Defaults (overridden via constructor or build_from_config) ───────────────
_DEFAULT_D_NORM_COLS = ["D1", "D4", "D10"]
_DEFAULT_D_GROUP_COL = "card1"
_DEFAULT_TARGET_ENC_COLS = ["card4", "card6", "P_emaildomain", "R_emaildomain"]
_LOO_SMOOTHING = 10  # shrinkage factor for test-time encoding

# ── V-column missing-indicator decision ──────────────────────────────────────
# _was_missing binary indicators for V-columns were REMOVED in v4 after a
# controlled diagnostic (feature_diagnostic.py, 2026-05):
#
#   Run A — raw V-values, median imputation, no indicators: AUC-PR = 0.4853
#   Run C — full engineer including 253 indicators:          AUC-PR = 0.1228
#   B→C delta: −0.364 AUC-PR
#
# Raising the indicator threshold from 0.05 → 0.80 (v3) reduced indicators
# from 253 to 206 but AUC-PR only recovered to 0.2009 — still well below
# the no-indicator baseline.  The 5–80% null V-column indicators also appear
# to be non-informative noise that crowds out the colsample_bytree budget.
#
# V-columns are now imputed with training medians only.  If a future analysis
# shows that missingness in specific V-columns is a genuine fraud signal
# (e.g. via chi-squared test of P(fraud|missing) ≠ P(fraud|present)), a
# targeted per-column indicator can be added back explicitly.


class FraudFeatureEngineer:
    """
    Stateful feature transformer for the IEEE-CIS fraud dataset.

    Usage:
        eng = FraudFeatureEngineer(...)
        X_train_eng = eng.fit_transform(X_train, y_train)
        X_test_eng  = eng.transform(X_test)
        eng.save_to_gcs(config)
    """

    def __init__(
        self,
        missing_threshold: float = 1.0,  # unused — kept for GCS back-compat
        d_norm_cols: list[str] = _DEFAULT_D_NORM_COLS,
        d_group_col: str = _DEFAULT_D_GROUP_COL,
        target_encode_cols: list[str] = _DEFAULT_TARGET_ENC_COLS,
        smoothing: float = _LOO_SMOOTHING,
    ):
        self.missing_threshold = missing_threshold  # no-op; indicators removed
        self.d_norm_cols = list(d_norm_cols)
        self.d_group_col = d_group_col
        self.target_encode_cols = list(target_encode_cols)
        self.smoothing = smoothing

        # Populated during fit — never touched by transform
        self._is_fitted: bool = False
        self._v_cols_present: list[str] = []
        self._high_null_v_cols: list[str] = []  # always empty — back-compat for older artifacts
        self._v_medians: Optional[pd.Series] = None
        self._d_group_medians: dict[str, pd.Series] = {}
        self._cat_stats: dict[str, pd.DataFrame] = {}
        self._global_mean: float = 0.0
        self._passthrough_medians: pd.Series = pd.Series(dtype=float)

    # ─── Fit ──────────────────────────────────────────────────────────────────

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "FraudFeatureEngineer":
        """Fit all transformers on training data only."""
        # 1. Discover V-columns present in this dataset
        self._v_cols_present = [c for c in ALL_V_COLS if c in X.columns]
        if self._v_cols_present:
            # No indicator logic — see module-level comment.
            # fillna(0): columns that are 100% null in training get median=NaN;
            # impute with 0 so downstream transform never produces NaN from median.
            self._v_medians = X[self._v_cols_present].median().fillna(0.0)

        # 2. Per-card D-column medians for normalisation
        for col in self.d_norm_cols:
            if col in X.columns and self.d_group_col in X.columns:
                self._d_group_medians[col] = (
                    X.groupby(self.d_group_col, sort=False)[col].median()
                )

        # 3. Global target mean (LOO / unseen-category fallback)
        self._global_mean = float(y.mean())

        # 4. Category stats for leave-one-out target encoding
        for col in self.target_encode_cols:
            if col in X.columns:
                tmp = pd.DataFrame({"cat": X[col].values, "y": y.values})
                self._cat_stats[col] = (
                    tmp.groupby("cat", sort=False)["y"]
                    .agg(["sum", "count"])
                    .astype(float)
                )

        # 5. Medians for non-V numeric pass-through columns (C/D/addr/dist/card2/3/5)
        passthrough_present = [c for c in _PASSTHROUGH_IMPUTE_COLS if c in X.columns]
        if passthrough_present:
            self._passthrough_medians = X[passthrough_present].median().fillna(0.0)
        else:
            self._passthrough_medians = pd.Series(dtype=float)

        self._is_fitted = True
        logger.info(
            "feature_engineer_fitted",
            extra={
                "v_cols_present": len(self._v_cols_present),
                "passthrough_cols_imputed": len(self._passthrough_medians),
                "global_fraud_rate": round(self._global_mean, 4),
            },
        )
        return self

    # ─── Transform ────────────────────────────────────────────────────────────

    def transform(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Apply all transformations.

        Pass y only for the training split to enable LOO target encoding.
        Leave y=None for test / inference (uses category means from fit).
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before transform()")

        df = X.copy()
        # Collect all brand-new columns here; concat once at the end to avoid
        # the pandas fragmentation penalty that comes from ~250 individual inserts.
        new_cols: dict[str, object] = {}

        # ── Pass-through imputation (applied first — downstream features use imputed vals) ──
        for col in self._passthrough_medians.index:
            if col in df.columns:
                df[col] = df[col].fillna(self._passthrough_medians[col])

        # ── Time features ──────────────────────────────────────────────────
        dt = df["TransactionDT"]
        new_cols["time_of_day"] = (dt % 86400).astype(np.int32)
        new_cols["day_of_week"] = ((dt // 86400) % 7).astype(np.int32)
        new_cols["hour_of_day"] = ((dt % 86400) // 3600).astype(np.int32)

        # ── Velocity / amount features ────────────────────────────────────
        new_cols["log_transaction_amt"] = np.log1p(df["TransactionAmt"])
        new_cols["amt_to_d1_ratio"] = df["TransactionAmt"] / (df["D1"].fillna(0) + 1)
        new_cols["is_round_amt"] = (df["TransactionAmt"] % 1 == 0).astype(np.int8)

        # ── D-column per-card median normalisation ────────────────────────
        # D1, D4, D10 are already imputed by the pass-through step above.
        for col in self.d_norm_cols:
            if col in df.columns and self.d_group_col in df.columns:
                card_med = df[self.d_group_col].map(self._d_group_medians[col])
                # Unknown / NaN card groups → 0 (neutral: no deviation from median)
                new_cols[f"{col}_card_norm"] = (df[col] - card_med).fillna(0.0)

        # ── V-column imputation (no indicators — see module-level comment) ────
        for col in self._v_cols_present:
            if col not in df.columns:
                continue
            df[col] = df[col].fillna(self._v_medians[col])

        # ── Target encoding (drop raw cat cols, add _te) ──────────────────
        te_drops = []
        for col in self.target_encode_cols:
            if col not in df.columns:
                continue
            new_cols[f"{col}_te"] = self._target_encode_col(df[col], col, y)
            te_drops.append(col)
        if te_drops:
            df = df.drop(columns=te_drops)

        # Single concat for all new columns — O(1) block operations vs O(N) inserts
        if new_cols:
            df = pd.concat(
                [df, pd.DataFrame(new_cols, index=df.index)], axis=1, copy=False
            )

        return df

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        return self.fit(X, y).transform(X, y)

    # ─── Internals ────────────────────────────────────────────────────────────

    def _target_encode_col(
        self,
        series: pd.Series,
        col: str,
        y: Optional[pd.Series],
    ) -> np.ndarray:
        """
        LOO encoding when y is provided (training); smoothed mean when y is None (test).

        LOO: encoded_i = (sum_cat - y_i) / max(count_cat - 1, 1)
        Singleton or unseen → global mean.
        Test-time smoothing: blend category mean with global mean using
        count / (count + smoothing) as the weight.
        """
        stats = self._cat_stats[col]
        cat_sum = series.map(stats["sum"]).fillna(0.0).values
        cat_count = series.map(stats["count"]).fillna(0.0).values

        if y is not None:
            # Leave-one-out: subtract current row from its category aggregate
            y_arr = y.values
            loo_sum = cat_sum - y_arr
            loo_count = np.maximum(cat_count - 1, 1)
            cat_mean = loo_sum / loo_count
            encoded = np.where(cat_count > 1, cat_mean, self._global_mean)
        else:
            safe_count = np.where(cat_count > 0, cat_count, 1.0)
            raw_mean = np.where(cat_count > 0, cat_sum / safe_count, self._global_mean)
            w = cat_count / (cat_count + self.smoothing)
            smoothed = w * raw_mean + (1.0 - w) * self._global_mean
            encoded = np.where(cat_count > 0, smoothed, self._global_mean)

        return encoded

    # ─── GCS persistence ──────────────────────────────────────────────────────

    def save_to_gcs(self, config: dict) -> str:
        """Serialize fitted engineer to GCS as a joblib file."""
        from google.cloud import storage as gcs

        bucket_name = config["storage"]["artifacts_bucket"]
        prefix = config["features"].get("transformer_gcs_prefix", "transformers/").rstrip("/")
        blob_path = f"{prefix}/feature_engineer.joblib"

        buf = io.BytesIO()
        joblib.dump(self, buf)
        buf.seek(0)

        client = gcs.Client(project=config["gcp"]["project_id"])
        blob = client.bucket(bucket_name).blob(blob_path)
        blob.upload_from_file(buf, content_type="application/octet-stream")

        uri = f"gs://{bucket_name}/{blob_path}"
        logger.info("transformer_saved_to_gcs", extra={"uri": uri})
        return uri

    @classmethod
    def load_from_gcs(cls, gcs_uri: str, project: str) -> "FraudFeatureEngineer":
        """Load a previously saved engineer from GCS."""
        from google.cloud import storage as gcs

        path = gcs_uri.removeprefix("gs://")
        bucket_name, blob_path = path.split("/", 1)

        buf = io.BytesIO()
        client = gcs.Client(project=project)
        client.bucket(bucket_name).blob(blob_path).download_to_file(buf)
        buf.seek(0)
        eng = joblib.load(buf)
        # Back-compat: older serialised engineers pre-date pass-through imputation
        if not hasattr(eng, "_passthrough_medians"):
            eng._passthrough_medians = pd.Series(dtype=float)
        return eng


# ─── Factory ──────────────────────────────────────────────────────────────────

def build_from_config(config: dict) -> FraudFeatureEngineer:
    """Instantiate engineer with all parameters from config/dev.yaml."""
    return FraudFeatureEngineer(
        missing_threshold=float(config["features"]["missing_indicator_threshold"]),
        d_norm_cols=config["features"]["d_normalize_cols"],
        d_group_col=config["features"]["d_normalize_group_col"],
        target_encode_cols=config["features"]["target_encode_cols"],
    )
