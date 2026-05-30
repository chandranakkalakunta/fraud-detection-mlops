"""
Fraud Detection MLOps — Streamlit Demo App

Three pages:
  1. Live Fraud Scorer      — input form → Cloud Run /predict → gauge + SHAP bar
  2. Model Performance      — AUC-PR progression, SHAP top-10, experiment table
  3. Drift Monitoring       — PSI trends, fraud rate, alert history from BigQuery

Designed for recruiter / interview demos.
Default form values allow one-click Submit with no manual input required.

Environment:
  CLOUD_RUN_API_URL — Cloud Run base URL (e.g. https://fraud-detection-api-xxx.run.app)
  API_KEY           — API key for X-API-Key header (from Secret Manager in prod)
  GCP_PROJECT_ID    — for BQ and GCS queries
  GCP_REGION        — Vertex AI region
  ENV               — "dev" for config loading
"""

import io
import json
import os
from datetime import datetime, timedelta, timezone

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection MLOps Demo",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ────────────────────────────────────────────────────────────────
API_URL = os.getenv("CLOUD_RUN_API_URL", "").rstrip("/")
API_KEY = os.getenv("API_KEY", "")
_HEADERS = {"X-API-Key": API_KEY, "Content-Type": "application/json"}

MODEL_PROGRESSION = [
    {"version": "lr-baseline",    "model": "Logistic Regression", "auc_pr": 0.2172, "auc_roc": 0.8600, "note": "Establishes floor; no feature engineering"},
    {"version": "Run A",          "model": "XGBoost",             "auc_pr": 0.4853, "auc_roc": 0.8894, "note": "Raw features; tree models far outperform LR"},
    {"version": "xgb-v5-no-te",  "model": "XGBoost",             "auc_pr": 0.4895, "auc_roc": 0.8925, "note": "TE removed; clean signal"},
    {"version": "lgb-v5-no-te",  "model": "LightGBM",            "auc_pr": 0.4913, "auc_roc": 0.8951, "note": "TE removed; best untuned"},
    {"version": "lgb-v6-converged","model": "LightGBM",           "auc_pr": 0.5089, "auc_roc": 0.8909, "note": "n=2000; still improving"},
    {"version": "xgb-v7-tuned",  "model": "XGBoost",             "auc_pr": 0.5166, "auc_roc": 0.8911, "note": "Vizier 30-trial GP Bandit"},
    {"version": "lgb-v7-tuned",  "model": "LightGBM",            "auc_pr": 0.5393, "auc_roc": 0.8964, "note": "TransactionID leakage → demoted"},
    {"version": "lgb-v8-no-txnid","model": "LightGBM ★ Champion","auc_pr": 0.5263, "auc_roc": 0.8923, "note": "Leakage fixed; production-ready"},
]

SHAP_TOP10 = [
    {"feature": "TransactionDT",  "mean_abs_shap": 0.450, "interpretation": "Fraud is strongly time-dependent; off-hours patterns separate fraud"},
    {"feature": "C13",            "mean_abs_shap": 0.414, "interpretation": "Card transaction count; high velocity → compromised card"},
    {"feature": "TransactionAmt", "mean_abs_shap": 0.331, "interpretation": "Fraudsters probe with small amounts or large unauthorized purchases"},
    {"feature": "C14",            "mean_abs_shap": 0.293, "interpretation": "Secondary card count; corroborates C13 velocity signal"},
    {"feature": "addr1",          "mean_abs_shap": 0.266, "interpretation": "Billing zip; mismatches with prior transactions flag fraud"},
    {"feature": "card1",          "mean_abs_shap": 0.263, "interpretation": "Card BIN/issuer; certain bins have higher fraud rates"},
    {"feature": "card2",          "mean_abs_shap": 0.238, "interpretation": "Card-associated zip; cross-validates addr1"},
    {"feature": "V91",            "mean_abs_shap": 0.214, "interpretation": "Vesta proprietary risk signal; strong discriminative power"},
    {"feature": "C1",             "mean_abs_shap": 0.188, "interpretation": "Distinct addresses on card; sudden changes flag account takeover"},
    {"feature": "dist1",          "mean_abs_shap": 0.188, "interpretation": "Purchaser-recipient distance; large gaps → card-not-present fraud"},
]

DEMO_TRANSACTION = {
    "TransactionAmt": 68.5, "TransactionDT": 86400.0,
    "card1": 9500.0, "card2": 325.0, "card3": 150.0, "card5": 226.0,
    "addr1": 299.0, "addr2": 87.0,
    "C1": 1.0, "C2": 1.0, "C3": 0.0, "C4": 0.0, "C5": 0.0,
    "C6": 1.0, "C7": 0.0, "C8": 0.0, "C9": 1.0, "C10": 0.0,
    "C11": 1.0, "C12": 0.0, "C13": 1.0, "C14": 1.0,
    "D1": 14.0,
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _api_available() -> bool:
    return bool(API_URL and API_KEY)


def _call_predict(payload: dict) -> dict | None:
    if not _api_available():
        return None
    try:
        r = requests.post(f"{API_URL}/predict", json=payload, headers=_HEADERS, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        st.error(f"API call failed: {exc}")
        return None


def _call_health() -> dict | None:
    if not _api_available():
        return None
    try:
        r = requests.get(f"{API_URL}/health", headers=_HEADERS, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def _load_drift_logs(days: int = 30) -> pd.DataFrame:
    try:
        from src.utils.config import load_config
        cfg = load_config(os.getenv("ENV", "dev"))
        from google.cloud import bigquery
        project = cfg["gcp"]["project_id"]
        dataset = cfg["bigquery"]["dataset"]
        table = cfg["monitoring"]["drift_log_table"]
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).date().isoformat()
        query = f"""
            SELECT timestamp, feature_name, psi_score, alert_fired, baseline_mean, current_mean
            FROM `{project}.{dataset}.{table}`
            WHERE DATE(timestamp) >= '{cutoff}'
            ORDER BY timestamp
        """
        return bigquery.Client(project=project).query(query).to_dataframe()
    except Exception:
        return pd.DataFrame()


def _load_shap_csv() -> pd.DataFrame | None:
    try:
        from src.utils.config import load_config
        from google.cloud import storage as gcs
        cfg = load_config(os.getenv("ENV", "dev"))
        bucket = cfg["storage"]["artifacts_bucket"]
        buf = io.BytesIO()
        gcs.Client(project=cfg["gcp"]["project_id"]).bucket(bucket).blob(
            "explainability/shap_feature_importance.csv"
        ).download_to_file(buf)
        buf.seek(0)
        return pd.read_csv(buf)
    except Exception:
        return None


def _gauge_chart(prob: float) -> go.Figure:
    color = "#e74c3c" if prob >= 0.5 else ("#f39c12" if prob >= 0.25 else "#27ae60")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob * 100, 1),
        number={"suffix": "%", "font": {"size": 40}},
        title={"text": "Fraud Probability", "font": {"size": 18}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 25],  "color": "#d5f5e3"},
                {"range": [25, 50], "color": "#fdebd0"},
                {"range": [50, 100],"color": "#fadbd8"},
            ],
            "threshold": {"line": {"color": "black", "width": 3}, "value": 24.58},
        },
    ))
    fig.update_layout(height=280, margin=dict(t=40, b=10, l=20, r=20))
    return fig


def _shap_bar(explanation: list[dict]) -> go.Figure:
    feats = [e["feature"] for e in explanation]
    vals  = [e["shap_value"] for e in explanation]
    colors = ["#e74c3c" if v > 0 else "#3498db" for v in vals]
    fig = go.Figure(go.Bar(
        x=vals, y=feats, orientation="h",
        marker_color=colors,
        text=[f"{v:+.3f}" for v in vals],
        textposition="auto",
    ))
    fig.update_layout(
        title="Top 3 SHAP Features — This Prediction",
        xaxis_title="SHAP value (positive = increases fraud risk)",
        height=220,
        margin=dict(t=40, b=10, l=10, r=10),
    )
    return fig


# ── Sidebar navigation ────────────────────────────────────────────────────────

st.sidebar.title("Fraud Detection MLOps")
st.sidebar.caption("Portfolio demo — IEEE-CIS dataset, GCP stack")
page = st.sidebar.radio(
    "Navigate",
    ["Live Fraud Scorer", "Model Performance", "Drift Monitoring"],
    label_visibility="collapsed",
)

if _api_available():
    health = _call_health()
    if health:
        st.sidebar.success(f"API online — {health.get('model_version', '?')}")
        st.sidebar.metric("Predictions today", health.get("predictions_today", "—"))
    else:
        st.sidebar.warning("API unreachable")
else:
    st.sidebar.info("No API_URL configured — live scoring disabled")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Live Fraud Scorer
# ══════════════════════════════════════════════════════════════════════════════

if page == "Live Fraud Scorer":
    st.title("Live Fraud Scorer")
    st.caption(
        "Enter transaction details and click **Score Transaction** to get a real-time fraud "
        "probability from the production model. Demo-safe defaults are pre-filled."
    )

    with st.form("score_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            amt = st.number_input("TransactionAmt ($)", value=68.5, min_value=0.01, step=0.01)
            card1 = st.number_input("card1", value=9500, step=1)
            c13 = st.number_input("C13 (card tx count)", value=1, step=1, min_value=0)
        with col2:
            card2 = st.number_input("card2", value=325.0, step=1.0)
            addr1 = st.number_input("addr1 (billing zip)", value=299.0, step=1.0)
            d1 = st.number_input("D1 (days since last tx)", value=14.0, step=1.0)
        with col3:
            txn_dt = st.number_input("TransactionDT (seconds)", value=86400, step=3600)
            c1 = st.number_input("C1 (addr count on card)", value=1, step=1, min_value=0)
            c14 = st.number_input("C14", value=1, step=1, min_value=0)

        submitted = st.form_submit_button("Score Transaction", use_container_width=True)

    if submitted:
        payload = dict(DEMO_TRANSACTION)
        payload.update({
            "TransactionAmt": float(amt),
            "card1": float(card1),
            "card2": float(card2),
            "addr1": float(addr1),
            "C13": float(c13),
            "C1": float(c1),
            "C14": float(c14),
            "D1": float(d1),
            "TransactionDT": float(txn_dt),
        })

        if _api_available():
            with st.spinner("Scoring…"):
                result = _call_predict(payload)
        else:
            # Offline demo — deterministic mock based on amount
            import hashlib, struct
            h = struct.unpack("f", hashlib.md5(str(amt).encode()).digest()[:4])[0]
            mock_prob = abs(h) % 1.0
            result = {
                "prediction_id": "demo-offline",
                "fraud_probability": round(mock_prob, 4),
                "prediction": int(mock_prob >= 0.2458),
                "threshold_used": 0.2458,
                "latency_ms": 0.0,
                "model_version": "lgb-v8-no-txnid (offline demo)",
                "explanation": [
                    {"feature": "TransactionAmt", "shap_value": 0.12, "direction": "increases_fraud_risk"},
                    {"feature": "C13",             "shap_value": -0.08, "direction": "decreases_fraud_risk"},
                    {"feature": "card1",           "shap_value": 0.05, "direction": "increases_fraud_risk"},
                ],
            }

        if result:
            prob = result["fraud_probability"]
            pred = result["prediction"]
            threshold = result["threshold_used"]

            col_gauge, col_verdict, col_shap = st.columns([1.2, 1, 1.5])
            with col_gauge:
                st.plotly_chart(_gauge_chart(prob), use_container_width=True)
            with col_verdict:
                st.metric("Fraud Probability", f"{prob:.1%}")
                if pred == 1:
                    st.error("**FRAUD ALERT** 🚨")
                else:
                    st.success("**Legitimate** ✓")
                st.caption(f"Threshold: {threshold:.4f} (Platt-calibrated)")
                st.caption(f"Latency: {result['latency_ms']:.1f} ms")
                st.caption(f"Model: {result['model_version']}")
            with col_shap:
                if result.get("explanation"):
                    st.plotly_chart(_shap_bar(result["explanation"]), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Model Performance Dashboard
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Model Performance":
    st.title("Model Performance Dashboard")

    # AUC-PR progression
    st.subheader("AUC-PR Progression — All Versions")
    df_prog = pd.DataFrame(MODEL_PROGRESSION)

    fig_prog = go.Figure()
    fig_prog.add_trace(go.Scatter(
        x=df_prog["version"], y=df_prog["auc_pr"],
        mode="lines+markers+text",
        text=[f"{v:.4f}" for v in df_prog["auc_pr"]],
        textposition="top center",
        line=dict(color="#3498db", width=2),
        marker=dict(size=8, color=[
            "#e74c3c" if "Champion" in r["model"] else "#3498db"
            for _, r in df_prog.iterrows()
        ]),
        name="AUC-PR",
    ))
    fig_prog.add_hline(y=0.48, line_dash="dash", line_color="orange",
                       annotation_text="CI/CD gate (0.48)")
    fig_prog.update_layout(
        xaxis_title="Model Version", yaxis_title="AUC-PR",
        yaxis=dict(range=[0.0, 0.65]),
        height=380, margin=dict(t=20, b=60),
        xaxis_tickangle=-35,
    )
    st.plotly_chart(fig_prog, use_container_width=True)

    # SHAP top 10
    st.subheader("SHAP Feature Importance — lgb-v8-no-txnid (147k test rows)")
    shap_df = _load_shap_csv()
    if shap_df is not None:
        top20 = shap_df.head(20)
    else:
        top20 = pd.DataFrame(SHAP_TOP10)
        top20 = top20.rename(columns={"mean_abs_shap": "mean_abs_shap"})

    fig_shap = go.Figure(go.Bar(
        x=top20["mean_abs_shap"][::-1] if "mean_abs_shap" in top20.columns else top20.iloc[:, 1][::-1],
        y=top20["feature"][::-1] if "feature" in top20.columns else top20.iloc[:, 0][::-1],
        orientation="h",
        marker_color="#e74c3c",
    ))
    fig_shap.update_layout(
        xaxis_title="Mean |SHAP value|",
        height=420, margin=dict(t=10, b=10, l=10, r=10),
    )
    st.plotly_chart(fig_shap, use_container_width=True)

    # Business interpretation table (top 10)
    st.subheader("Feature Business Interpretation")
    shap_interp = pd.DataFrame(SHAP_TOP10)
    shap_interp["Rank"] = range(1, len(shap_interp) + 1)
    shap_interp = shap_interp[["Rank", "feature", "mean_abs_shap", "interpretation"]]
    shap_interp.columns = ["Rank", "Feature", "Mean |SHAP|", "Business Interpretation"]
    st.dataframe(shap_interp, use_container_width=True, hide_index=True)

    # All runs table
    st.subheader("All Experiment Runs")
    df_all = pd.DataFrame(MODEL_PROGRESSION)
    df_all.columns = ["Version", "Model", "AUC-PR", "AUC-ROC", "Key Decision"]
    st.dataframe(
        df_all.style.highlight_max(subset=["AUC-PR"], color="#d5f5e3"),
        use_container_width=True,
        hide_index=True,
    )

    # Champion model card
    st.subheader("Champion Model — lgb-v8-no-txnid")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("AUC-PR", "0.5263")
    col2.metric("AUC-ROC", "0.8923")
    col3.metric("Precision", "0.610")
    col4.metric("Recall", "0.466")
    col1.metric("F1", "0.5285")
    col2.metric("KS Statistic", "0.6310")
    col3.metric("Threshold (calibrated)", "0.2458")
    col4.metric("Features", "387")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Drift Monitoring
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Drift Monitoring":
    st.title("Drift Monitoring")
    st.caption("PSI and fraud rate trends from BigQuery drift_logs · Last 30 days")

    days = st.slider("Look-back window (days)", min_value=7, max_value=90, value=30)
    df_drift = _load_drift_logs(days=days)

    if df_drift.empty:
        st.info("No drift log data yet. The daily Cloud Scheduler job will populate this table after deployment.")
        st.markdown("""
        **Monitoring setup:**
        - Cloud Scheduler fires daily at **6:00 AM IST** → POST /drift-check
        - PSI computed for **TransactionAmt** and **C13** vs training baseline
        - Alert fires when **PSI > 0.2** for either feature
        - Fraud rate monitored vs training rate (3.51%) — alert at **±2 std deviations**
        - Results written to `fraud_detection.drift_logs` (BigQuery)
        """)

        # Show demo chart with synthetic data
        demo_dates = pd.date_range(end=datetime.now(), periods=30, freq="D")
        import numpy as np
        rng = np.random.default_rng(42)
        demo_psi = pd.DataFrame({
            "date": demo_dates,
            "TransactionAmt": rng.uniform(0.01, 0.15, 30),
            "C13": rng.uniform(0.01, 0.12, 30),
        })
        st.caption("⚠️ Demo data — actual drift logs will appear after first scheduler run")
        fig_demo = go.Figure()
        fig_demo.add_trace(go.Scatter(x=demo_psi["date"], y=demo_psi["TransactionAmt"],
                                      name="TransactionAmt PSI", line=dict(color="#3498db")))
        fig_demo.add_trace(go.Scatter(x=demo_psi["date"], y=demo_psi["C13"],
                                      name="C13 PSI", line=dict(color="#e74c3c")))
        fig_demo.add_hline(y=0.2, line_dash="dash", line_color="orange",
                           annotation_text="Alert threshold (PSI=0.2)")
        fig_demo.update_layout(yaxis_title="PSI Score", height=320, margin=dict(t=10, b=10))
        st.plotly_chart(fig_demo, use_container_width=True)

    else:
        df_drift["timestamp"] = pd.to_datetime(df_drift["timestamp"])

        feat_df = df_drift[df_drift["feature_name"] != "fraud_rate"]
        perf_df = df_drift[df_drift["feature_name"] == "fraud_rate"]

        # PSI chart
        st.subheader("Feature Drift (PSI)")
        fig_psi = go.Figure()
        for feat in feat_df["feature_name"].unique():
            sub = feat_df[feat_df["feature_name"] == feat]
            fig_psi.add_trace(go.Scatter(
                x=sub["timestamp"], y=sub["psi_score"],
                name=feat, mode="lines+markers",
            ))
        fig_psi.add_hline(y=0.2, line_dash="dash", line_color="orange",
                          annotation_text="Alert (PSI=0.2)")
        fig_psi.update_layout(yaxis_title="PSI Score", height=320, margin=dict(t=10, b=10))
        st.plotly_chart(fig_psi, use_container_width=True)

        # Fraud rate trend
        if not perf_df.empty:
            st.subheader("Fraud Rate Trend (7-day rolling)")
            fig_fr = go.Figure()
            fig_fr.add_trace(go.Scatter(
                x=perf_df["timestamp"], y=perf_df["current_mean"],
                name="Observed fraud rate", line=dict(color="#e74c3c"),
            ))
            fig_fr.add_hline(y=0.0351, line_dash="dash", line_color="green",
                             annotation_text="Training rate (3.51%)")
            fig_fr.update_layout(yaxis_title="Fraud Rate", height=280, margin=dict(t=10, b=10))
            st.plotly_chart(fig_fr, use_container_width=True)

        # Alert history
        st.subheader("Alert History")
        alerts = df_drift[df_drift["alert_fired"]].copy()
        if alerts.empty:
            st.success("No alerts fired in the selected window.")
        else:
            st.warning(f"{len(alerts)} alert(s) fired")
            st.dataframe(
                alerts[["timestamp", "feature_name", "psi_score", "baseline_mean", "current_mean"]],
                use_container_width=True,
                hide_index=True,
            )
