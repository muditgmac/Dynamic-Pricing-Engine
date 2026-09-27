"""Streamlit dashboard for the assumption-based pricing system.

Run with:
    streamlit run dashboard/app.py --server.port 8501

The dashboard deliberately separates:
- predictive calendar-unavailability modeling;
- explicit price-sensitivity scenario assumptions;
- proxy-based pricing decision support.

It does not present the scenario outputs as causal elasticity estimates or
realized-revenue forecasts.
"""

import os
import sys
import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# Add project root to path.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.scenario_engine import compute_scenario_curve  # noqa: E402
from src.utils.config import load_config  # noqa: E402

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Dynamic Pricing Engine",
    page_icon="💲",
    layout="wide",
    initial_sidebar_state="expanded",
)

config = load_config()

# Local development defaults to localhost. Docker Compose overrides this with
# PRICING_API_URL=http://api:8000 so the dashboard uses Compose service DNS.
API_URL = os.getenv(
    "PRICING_API_URL",
    f"http://localhost:{config['api']['port']}",
).rstrip("/")


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------
def _api_get(path: str, timeout: float = 5.0) -> dict | None:
    """Return JSON from an API GET endpoint, or None when unavailable."""
    try:
        import httpx

        response = httpx.get(
            f"{API_URL}{path}",
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json()
    except Exception:
        return None


def get_pricing_recommendation(
    params: dict,
) -> tuple[dict | None, str | None]:
    """Call the production pricing API without inventing a local fallback."""
    try:
        import httpx

        response = httpx.post(
            f"{API_URL}/predict",
            json=params,
            timeout=8.0,
        )
        response.raise_for_status()
        return response.json(), None
    except Exception as exc:
        return None, str(exc)


# ---------------------------------------------------------------------------
# Sidebar: listing and scenario inputs
# ---------------------------------------------------------------------------
st.sidebar.title("📋 Listing & Scenario")

room_type = st.sidebar.selectbox(
    "Room Type",
    [
        "Entire home/apt",
        "Private room",
        "Hotel room",
        "Shared room",
    ],
)
beds = st.sidebar.slider(
    "Beds",
    1,
    10,
    2,
)
bathrooms = st.sidebar.slider(
    "Bathrooms",
    0.5,
    5.0,
    1.0,
    0.5,
)

# Load neighborhoods from processed data if available.
neighborhoods = [
    "Williamsburg",
    "Harlem",
    "SoHo",
    "Chelsea",
    "Astoria",
    "East Village",
    "Upper West Side",
    "Bushwick",
    "Midtown",
    "Lower East Side",
    "Hell's Kitchen",
    "Bedford-Stuyvesant",
]

features_path = (
    PROJECT_ROOT
    / config["data"]["processed_dir"]
    / "listings_features.parquet"
)

if features_path.exists():
    try:
        listings_df = pd.read_parquet(
            features_path,
            columns=["neighbourhood_cleansed"],
        )
        neighborhoods = sorted(
            listings_df[
                "neighbourhood_cleansed"
            ]
            .dropna()
            .unique()
            .tolist()
        )
    except Exception:
        pass

neighborhood = st.sidebar.selectbox(
    "Neighborhood",
    neighborhoods,
)
checkin_date = st.sidebar.date_input(
    "Check-in Date",
    pd.Timestamp("2024-07-15"),
)
checkout_date = st.sidebar.date_input(
    "Check-out Date",
    pd.Timestamp("2024-07-18"),
)
amenity_score = st.sidebar.slider(
    "Amenity Score",
    0.0,
    1.0,
    0.5,
    0.05,
)
review_score = st.sidebar.slider(
    "Review Score",
    0.0,
    5.0,
    4.0,
    0.1,
)

st.sidebar.markdown("---")
st.sidebar.subheader("Pricing Assumptions")

sensitivity_scenario = st.sidebar.selectbox(
    "Price sensitivity",
    ["low", "moderate", "high"],
    index=1,
    help=(
        "This is an explicit scenario assumption, "
        "not an elasticity estimate learned from Airbnb data."
    ),
)

use_custom_reference = st.sidebar.checkbox(
    "Use custom reference price",
    value=False,
)

reference_price = None
if use_custom_reference:
    reference_price = st.sidebar.number_input(
        "Reference nightly price ($)",
        min_value=1.0,
        value=150.0,
        step=5.0,
        help=(
            "Baseline price at which the demand model is evaluated."
        ),
    )

st.sidebar.markdown("---")
auto_refresh = st.sidebar.checkbox(
    "Auto-refresh",
    value=False,
)

request_params = {
    "room_type": room_type,
    "beds": beds,
    "bathrooms": bathrooms,
    "neighborhood": neighborhood,
    "checkin_date": str(checkin_date),
    "checkout_date": str(checkout_date),
    "amenity_score": amenity_score,
    "review_score": review_score,
    "sensitivity_scenario": sensitivity_scenario,
}

if reference_price is not None:
    request_params["reference_price"] = reference_price


# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------
st.title("💲 Dynamic Pricing Decision Support")
st.markdown(
    "Leakage-audited XGBoost unavailability prediction + "
    "explicit price-sensitivity scenarios."
)

result, api_error = get_pricing_recommendation(
    request_params
)

if result is None:
    st.error(
        "The pricing API is unavailable, so the dashboard will not "
        "fabricate a local recommendation."
    )
    st.caption(
        f"API target: {API_URL}. "
        f"Last error: {api_error or 'unknown'}"
    )
    st.stop()

st.info(result["methodology_note"])

# ---------------------------------------------------------------------------
# Row 1: key outputs
# ---------------------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)

with col1:
    price_delta = (
        result["recommended_price"]
        - result["reference_price"]
    )
    st.metric(
        label="Scenario Price",
        value=f"${result['recommended_price']:.2f}",
        delta=(
            f"{price_delta:+.2f} vs reference"
        ),
    )

with col2:
    st.metric(
        label="Reference Price",
        value=f"${result['reference_price']:.2f}",
    )
    st.caption(
        result[
            "reference_price_source"
        ].replace("_", " ").title()
    )

with col3:
    st.metric(
        label="Baseline Unavailability",
        value=(
            f"{result['unavailability_probability']:.1%}"
        ),
    )
    st.caption(
        "Calendar-unavailability proxy; not confirmed bookings."
    )

with col4:
    st.metric(
        label="Revenue-Proxy Change",
        value=(
            f"{result['revenue_proxy_change_pct']:+.1f}%"
        ),
    )
    st.caption(
        f"{result['selected_scenario'].title()} "
        f"sensitivity assumption"
    )

if result.get("is_anomaly"):
    st.warning(
        "⚠️ Input is unusual relative to the anomaly detector's "
        "training distribution. Interpret the recommendation cautiously."
    )

st.markdown("---")

# ---------------------------------------------------------------------------
# Row 2: selected scenario curve + cross-scenario comparison
# ---------------------------------------------------------------------------
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.subheader("Selected Scenario Curve")

    curve = compute_scenario_curve(
        base_demand_proxy=(
            result["unavailability_probability"]
        ),
        reference_price=result["reference_price"],
        sensitivity=result["assumed_sensitivity"],
        floor_price=result["price_bounds"][0],
        ceiling_price=result["price_bounds"][1],
        n_points=161,
        config=config,
    )

    fig = make_subplots(
        specs=[[{"secondary_y": True}]]
    )

    fig.add_trace(
        go.Scatter(
            x=curve["prices"],
            y=curve["revenue_proxies"],
            name="Revenue proxy",
            line=dict(
                color="#2563eb",
                width=3,
            ),
            fill="tozeroy",
            fillcolor="rgba(37, 99, 235, 0.1)",
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=curve["prices"],
            y=curve["demand_proxies"],
            name="Demand proxy",
            line=dict(
                color="#16a34a",
                width=2,
                dash="dash",
            ),
        ),
        secondary_y=True,
    )

    fig.add_vline(
        x=result["recommended_price"],
        line_dash="dot",
        line_color="red",
        annotation_text=(
            f"Scenario: ${result['recommended_price']:.0f}"
        ),
    )
    fig.add_vline(
        x=result["reference_price"],
        line_dash="dot",
        line_color="gray",
        annotation_text=(
            f"Reference: ${result['reference_price']:.0f}"
        ),
    )

    fig.update_layout(
        height=420,
        margin=dict(
            t=30,
            b=30,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
        ),
    )
    fig.update_yaxes(
        title_text="Revenue proxy",
        secondary_y=False,
    )
    fig.update_yaxes(
        title_text="Demand proxy",
        secondary_y=True,
        range=[0, 1],
    )
    fig.update_xaxes(
        title_text="Candidate nightly price ($)"
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

with chart_col2:
    st.subheader("Sensitivity Comparison")

    scenario_df = pd.DataFrame(
        result["scenario_results"]
    )

    fig2 = px.bar(
        scenario_df,
        x="scenario_name",
        y="recommended_price",
        text="recommended_price",
        labels={
            "scenario_name": "Sensitivity assumption",
            "recommended_price": "Recommended price ($)",
        },
    )
    fig2.update_traces(
        texttemplate="$%{text:.0f}",
        textposition="outside",
    )
    fig2.update_layout(
        height=360,
        margin=dict(
            t=30,
            b=30,
        ),
        showlegend=False,
    )
    st.plotly_chart(
        fig2,
        use_container_width=True,
    )

    comparison = scenario_df[
        [
            "scenario_name",
            "sensitivity",
            "demand_proxy_at_recommended",
            "revenue_proxy_change_pct",
        ]
    ].copy()
    comparison.columns = [
        "Scenario",
        "Sensitivity",
        "Demand proxy",
        "Revenue-proxy Δ %",
    ]
    st.dataframe(
        comparison,
        use_container_width=True,
        hide_index=True,
    )

# ---------------------------------------------------------------------------
# Row 3: TreeSHAP + price positioning
# ---------------------------------------------------------------------------
detail_col1, detail_col2 = st.columns(2)

with detail_col1:
    st.subheader(
        "What Drives Baseline Unavailability?"
    )

    shap_features = result.get(
        "shap_top_features",
        [],
    )

    if shap_features:
        shap_df = pd.DataFrame(
            shap_features
        ).sort_values(
            "contribution",
            ascending=True,
        )

        fig3 = px.bar(
            shap_df,
            x="contribution",
            y="feature",
            orientation="h",
            color="contribution",
            color_continuous_scale="RdBu_r",
        )
        fig3.update_layout(
            height=350,
            margin=dict(
                t=10,
                b=10,
            ),
            yaxis_title="",
            xaxis_title=(
                "TreeSHAP contribution to raw model margin"
            ),
            showlegend=False,
        )
        st.plotly_chart(
            fig3,
            use_container_width=True,
        )
        st.caption(
            "Positive TreeSHAP values push the XGBoost model "
            "toward higher predicted calendar unavailability."
        )
    else:
        st.info(
            "TreeSHAP explanations are unavailable for this run."
        )

with detail_col2:
    st.subheader("Price Positioning")

    recommended = result[
        "recommended_price"
    ]
    reference = result[
        "reference_price"
    ]
    low, high = result[
        "price_bounds"
    ]

    fig4 = go.Figure()
    fig4.add_trace(
        go.Bar(
            x=[
                "Floor",
                "Scenario Price",
                "Reference",
                "Ceiling",
            ],
            y=[
                low,
                recommended,
                reference,
                high,
            ],
            marker_color=[
                "#94a3b8",
                "#2563eb",
                "#64748b",
                "#94a3b8",
            ],
            text=[
                f"${low:.0f}",
                f"${recommended:.0f}",
                f"${reference:.0f}",
                f"${high:.0f}",
            ],
            textposition="outside",
        )
    )

    fig4.update_layout(
        height=350,
        margin=dict(
            t=10,
            b=10,
        ),
        yaxis_title="Nightly price ($)",
        showlegend=False,
    )
    st.plotly_chart(
        fig4,
        use_container_width=True,
    )

# ---------------------------------------------------------------------------
# Row 4: model metrics + drift monitor
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("Model Validation & Monitoring")

metrics = _api_get(
    "/metrics",
    timeout=3.0,
)

if metrics:
    metric_cols = st.columns(5)
    metric_cols[0].metric(
        "Holdout ROC-AUC",
        f"{metrics['holdout_auc']:.3f}",
    )
    metric_cols[1].metric(
        "Holdout PR-AUC",
        f"{metrics['holdout_pr_auc']:.3f}",
    )
    metric_cols[2].metric(
        "Log Loss",
        f"{metrics['holdout_log_loss']:.3f}",
    )
    metric_cols[3].metric(
        "Brier",
        f"{metrics['holdout_brier']:.3f}",
    )
    metric_cols[4].metric(
        "CV ROC-AUC",
        f"{metrics['cv_auc_mean']:.3f}",
    )

st.markdown("#### Drift Monitor")

drift_status = "unavailable"
drift_score = 0.0
features_drifted = []

drift_data = _api_get(
    "/drift-report",
    timeout=5.0,
)

if drift_data:
    drift_status = drift_data.get(
        "status",
        "unavailable",
    )
    drift_score = drift_data.get(
        "drift_score",
        0.0,
    )
    features_drifted = drift_data.get(
        "features_drifted",
        [],
    )

drift_col1, drift_col2, drift_col3 = st.columns(3)

with drift_col1:
    status_color = {
        "healthy": "🟢",
        "degraded": "🟡",
        "critical": "🔴",
    }.get(
        drift_status,
        "⚪",
    )
    st.metric(
        "Model Status",
        f"{status_color} {drift_status.title()}",
    )

with drift_col2:
    st.metric(
        "Drift Score",
        f"{drift_score:.3f}",
    )

with drift_col3:
    st.metric(
        "Features Drifted",
        len(features_drifted),
    )
    if features_drifted:
        st.caption(
            ", ".join(features_drifted)
        )

fig5 = go.Figure(
    go.Indicator(
        mode="gauge+number",
        value=drift_score,
        domain={
            "x": [0, 1],
            "y": [0, 1],
        },
        gauge={
            "axis": {
                "range": [0, 1]
            },
            "bar": {
                "color": "#2563eb"
            },
            "steps": [
                {
                    "range": [0, 0.3],
                    "color": "#dcfce7",
                },
                {
                    "range": [0.3, 0.6],
                    "color": "#fef9c3",
                },
                {
                    "range": [0.6, 1.0],
                    "color": "#fecaca",
                },
            ],
            "threshold": {
                "line": {
                    "color": "red",
                    "width": 4,
                },
                "thickness": 0.75,
                "value": 0.6,
            },
        },
        title={
            "text": "Overall Drift Score"
        },
    )
)
fig5.update_layout(
    height=250,
    margin=dict(
        t=50,
        b=10,
    ),
)
st.plotly_chart(
    fig5,
    use_container_width=True,
)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.caption(
    f"Model Version: {result.get('model_version', 'N/A')} | "
    f"Scenario: {result['selected_scenario']} "
    f"(s={result['assumed_sensitivity']:.2f}) | "
    "XGBoost + TreeSHAP + assumption-based scenario analysis + FastAPI"
)

if auto_refresh:
    time.sleep(
        config["dashboard"][
            "refresh_interval_seconds"
        ]
    )
    st.rerun()
