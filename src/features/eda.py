"""
EDA utilities for IEEE-CIS Fraud Detection dataset.
All heavy logic lives here; the notebook only calls these functions.
"""

import os
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils.logging import get_logger

logger = get_logger(__name__)

plt.rcParams.update({
    "figure.dpi": 120,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
})


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_from_bigquery(config: dict, limit: Optional[int] = None) -> pd.DataFrame:
    """Load transactions_joined from BigQuery for EDA."""
    from google.cloud import bigquery

    project = config["gcp"]["project_id"]
    dataset = config["bigquery"]["dataset"]
    table = config["bigquery"]["tables"]["transactions_joined"]
    limit_clause = f"LIMIT {limit}" if limit else ""

    query = f"""
        SELECT
          isFraud, TransactionAmt, TransactionDT,
          ProductCD, card4, card6,
          P_emaildomain, R_emaildomain,
          C1, C2, C3, C4, C5, C6, C7, C8, C9, C10, C11, C12, C13, C14,
          D1, D2, D3, D4, D5, D6, D7, D8, D9, D10, D11, D12, D13, D14, D15,
          M1, M2, M3, M4, M5, M6, M7, M8, M9,
          V1, V2, V3, V4, V5, V6, V7, V8, V9, V10
        FROM `{project}.{dataset}.{table}`
        {limit_clause}
    """
    client = bigquery.Client(project=project)
    df = client.query(query).to_dataframe(progress_bar_type=None)
    logger.info("eda_data_loaded", extra={"rows": len(df), "columns": len(df.columns)})
    return df


# ─── Plot 1: Class distribution ───────────────────────────────────────────────

def plot_class_distribution(df: pd.DataFrame) -> plt.Figure:
    """Bar chart of fraud vs legitimate with count and percentage labels."""
    counts = df["isFraud"].value_counts().sort_index()
    labels = ["Legitimate (0)", "Fraud (1)"]
    colors = ["#4C9BE8", "#E84C4C"]
    pcts = counts / counts.sum() * 100

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, counts.values, color=colors, width=0.5)
    for bar, count, pct in zip(bars, counts.values, pcts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.01,
            f"{count:,}\n({pct:.2f}%)",
            ha="center", va="bottom", fontsize=10,
        )
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.set_title("Class Distribution — Severe Imbalance (~3.5% Fraud)", pad=12)
    ax.set_ylabel("Transaction Count")
    fig.tight_layout()
    return fig


# ─── Plot 2: Null heatmap ─────────────────────────────────────────────────────

def plot_null_heatmap(df: pd.DataFrame) -> plt.Figure:
    """Heatmap showing null percentage per column, grouped by feature type."""
    null_pct = (df.isnull().mean() * 100).reset_index()
    null_pct.columns = ["Feature", "Null%"]
    null_pct["Group"] = null_pct["Feature"].apply(_feature_group)
    null_pct = null_pct.sort_values(["Group", "Null%"], ascending=[True, False])

    pivot = null_pct.set_index("Feature")[["Null%"]].T

    fig, ax = plt.subplots(figsize=(18, 2.5))
    sns.heatmap(
        pivot,
        ax=ax,
        cmap="YlOrRd",
        vmin=0, vmax=100,
        linewidths=0.3,
        cbar_kws={"label": "Null %", "shrink": 0.8},
        xticklabels=True,
    )
    ax.set_yticklabels([""], rotation=0)
    ax.set_title("Null Percentage by Feature (V-features can exceed 90% missing)", pad=10)
    plt.xticks(rotation=90, fontsize=7)
    fig.tight_layout()
    return fig


def _feature_group(name: str) -> str:
    for prefix, group in [
        ("V", "V"), ("C", "C"), ("D", "D"), ("M", "M"),
        ("id_", "identity"), ("card", "card"), ("addr", "addr"),
    ]:
        if name.startswith(prefix):
            return group
    return "other"


# ─── Plot 3: TransactionAmt by fraud label ────────────────────────────────────

def plot_transaction_amount(df: pd.DataFrame) -> plt.Figure:
    """Log-scale KDE of TransactionAmt split by isFraud label."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    legit = df.loc[df["isFraud"] == 0, "TransactionAmt"].clip(upper=5000)
    fraud = df.loc[df["isFraud"] == 1, "TransactionAmt"].clip(upper=5000)

    # Left: KDE overlay
    sns.kdeplot(legit, ax=axes[0], label="Legit", color="#4C9BE8", fill=True, alpha=0.4)
    sns.kdeplot(fraud, ax=axes[0], label="Fraud", color="#E84C4C", fill=True, alpha=0.4)
    axes[0].set_title("TransactionAmt Distribution (KDE)")
    axes[0].set_xlabel("Amount (clipped at $5,000)")
    axes[0].legend()

    # Right: box plots on log scale
    plot_df = df[["TransactionAmt", "isFraud"]].copy()
    plot_df["label"] = plot_df["isFraud"].map({0: "Legit", 1: "Fraud"})
    plot_df["log_amt"] = np.log1p(plot_df["TransactionAmt"])
    sns.boxplot(
        data=plot_df, x="label", y="log_amt", ax=axes[1],
        palette={"Legit": "#4C9BE8", "Fraud": "#E84C4C"},
    )
    axes[1].set_title("log(1 + TransactionAmt) by Label")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("log(1 + Amount)")

    fig.suptitle("Fraud transactions skew toward lower amounts, with long high-value tail", y=1.01)
    fig.tight_layout()
    return fig


# ─── Plot 4: Card types and email domains ─────────────────────────────────────

def plot_categorical_fraud_rates(df: pd.DataFrame) -> plt.Figure:
    """Bar charts of fraud rate for top card types and email domains."""
    fig, axes = plt.subplots(1, 3, figsize=(17, 4))

    _fraud_rate_bar(df, "card4", "Card Network (card4)", axes[0], top_n=10)
    _fraud_rate_bar(df, "card6", "Card Type (card6)", axes[1], top_n=8)
    _fraud_rate_bar(df, "P_emaildomain", "Purchaser Email Domain", axes[2], top_n=12)

    fig.suptitle("Fraud Rate by Category (min 500 transactions)", y=1.02)
    fig.tight_layout()
    return fig


def _fraud_rate_bar(
    df: pd.DataFrame, col: str, title: str, ax: plt.Axes, top_n: int = 10
) -> None:
    stats = (
        df.groupby(col, observed=True)["isFraud"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "fraud_rate", "count": "n"})
        .query("n >= 500")
        .nlargest(top_n, "fraud_rate")
        .reset_index()
    )
    bars = ax.barh(stats[col], stats["fraud_rate"] * 100, color="#E84C4C", alpha=0.8)
    for bar, n in zip(bars, stats["n"]):
        ax.text(
            bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
            f"n={n:,}", va="center", fontsize=8,
        )
    ax.set_title(title)
    ax.set_xlabel("Fraud Rate (%)")
    ax.invert_yaxis()


# ─── Plot 5: D-column temporal correlations ───────────────────────────────────

def plot_d_column_correlations(df: pd.DataFrame) -> plt.Figure:
    """Heatmap of D-feature correlations with isFraud and TransactionAmt."""
    d_cols = [c for c in df.columns if c.startswith("D")]
    cols_of_interest = d_cols + ["isFraud", "TransactionAmt", "TransactionDT"]
    corr = df[cols_of_interest].corr()

    # Focus on correlations with isFraud and TransactionAmt
    focus = corr[["isFraud", "TransactionAmt", "TransactionDT"]].loc[d_cols].sort_values("isFraud")

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        focus, ax=ax, cmap="RdBu_r", center=0,
        vmin=-0.3, vmax=0.3,
        annot=True, fmt=".2f", linewidths=0.5,
        cbar_kws={"label": "Pearson r"},
    )
    ax.set_title("D-Feature Correlation with isFraud / TransactionAmt / TransactionDT", pad=12)
    fig.tight_layout()
    return fig
