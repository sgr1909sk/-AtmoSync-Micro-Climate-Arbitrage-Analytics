from __future__ import annotations

from datetime import datetime, timedelta
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# -----------------------------------------------------------------------------
# AtmoSync | Micro-Climate Arbitrage Analytics
# Demo values are synthetic. Upload a CSV to work with your own shipment data.
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="AtmoSync | Climate Intelligence",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
      :root { --ink:#e8f0f6; --muted:#93a7b8; --teal:#42dfc2; --blue:#67a9ff; --panel:#101d2a; --line:rgba(148,175,196,.14); }
      html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
      .stApp { background: radial-gradient(ellipse at 12% 0%, rgba(31,116,126,.18), transparent 34%), linear-gradient(145deg,#08121d 0%,#0b1723 55%,#0b1b28 100%); color:var(--ink); }
      [data-testid="stHeader"] { background:rgba(8,18,29,.78); }
      [data-testid="stSidebar"] { background:linear-gradient(180deg,#0d1b28 0%,#0a1520 100%); border-right:1px solid var(--line); }
      [data-testid="stSidebar"] * { color:#dce8f0; }
      h1,h2,h3 { font-family:'Manrope',sans-serif !important; letter-spacing:-.035em; }
      h1 { font-size:2.05rem !important; }
      h2 { font-size:1.25rem !important; }
      h3 { font-size:1.02rem !important; }
      p, label, .stCaption { color:#a9bbc9; }
      .block-container { padding-top:1.55rem; padding-bottom:3rem; max-width:1500px; }
      div[data-testid="stMetric"] { background:linear-gradient(145deg,rgba(20,39,54,.95),rgba(13,28,41,.94)); border:1px solid var(--line); border-radius:16px; padding:17px 18px; box-shadow:0 10px 30px rgba(0,0,0,.12); }
      div[data-testid="stMetricLabel"] { color:#9eb2c1; font-size:.78rem; }
      div[data-testid="stMetricValue"] { color:#f3f8fb; font-family:'Manrope',sans-serif; font-weight:800; }
      div[data-testid="stMetricDelta"] { font-size:.76rem; }
      div[data-testid="stTabs"] button { color:#a9bbc9; }
      div[data-testid="stTabs"] button[aria-selected="true"] { color:#53e5cc; border-bottom-color:#53e5cc; }
      div[data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:13px; overflow:hidden; }
      .hero { background:linear-gradient(110deg,rgba(20,52,67,.97),rgba(15,37,54,.92) 57%,rgba(18,49,62,.9)); border:1px solid rgba(75,219,194,.23); border-radius:22px; padding:24px 28px; margin:0 0 22px 0; position:relative; overflow:hidden; }
      .hero:after { content:' '; position:absolute; width:260px; height:260px; border-radius:50%; right:-55px; top:-135px; background:rgba(66,223,194,.12); filter:blur(2px); }
      .eyebrow { color:#5ce6ce; text-transform:uppercase; letter-spacing:.18em; font-size:.68rem; font-weight:700; margin-bottom:9px; }
      .hero-title { color:#f3fafc; font-family:'Manrope',sans-serif; font-size:2rem; line-height:1.13; font-weight:800; margin:0 0 8px 0; }
      .hero-copy { color:#aac1ce; font-size:.94rem; max-width:780px; margin:0; }
      .status-pill { display:inline-block; padding:5px 10px; border-radius:30px; font-size:.7rem; letter-spacing:.04em; font-weight:700; background:rgba(66,223,194,.12); color:#6cebd5; border:1px solid rgba(66,223,194,.25); }
      .section-kicker { color:#62dec9; font-size:.68rem; font-weight:700; letter-spacing:.15em; text-transform:uppercase; margin:2px 0 7px 0; }
      .panel { background:linear-gradient(145deg,rgba(17,34,49,.95),rgba(13,27,40,.96)); border:1px solid var(--line); border-radius:17px; padding:17px 19px; }
      .small-muted { color:#91a7b7; font-size:.78rem; }
      .alert-card { background:rgba(28,47,60,.74); border:1px solid rgba(246,180,91,.22); border-left:3px solid #f2b35d; border-radius:12px; padding:12px 14px; margin-bottom:9px; }
      .alert-high { border-left-color:#ff747f; border-color:rgba(255,116,127,.25); }
      .alert-low { border-left-color:#49d9bf; border-color:rgba(73,217,191,.22); }
      .alert-title { color:#edf5f8; font-weight:700; font-size:.84rem; }
      .alert-body { color:#9fb3c1; font-size:.75rem; margin-top:3px; }
      .footer { color:#718899; font-size:.72rem; border-top:1px solid var(--line); padding-top:14px; margin-top:22px; }
      .stButton>button, .stDownloadButton>button { border-radius:10px; border:1px solid rgba(75,219,194,.35); background:rgba(48,177,158,.12); color:#b7f8eb; font-weight:700; }
      .stButton>button:hover, .stDownloadButton>button:hover { border-color:#54e6ce; color:white; background:rgba(48,177,158,.22); }
      .stSelectbox div[data-baseweb="select"] > div, .stMultiSelect div[data-baseweb="select"] > div { background:#122435; border-color:#294052; }
      .stTextInput input, .stNumberInput input { background:#122435; color:#edf5f8; }
      hr { border-color:var(--line); }
      @media(max-width:800px){ .hero-title{font-size:1.55rem;} .hero{padding:19px;} }
    </style>
    """,
    unsafe_allow_html=True,
)


REQUIRED_COLUMNS = [
    "shipment_id", "origin", "destination", "product", "status",
    "temperature_c", "humidity_pct", "rainfall_mm", "transit_hours",
    "baseline_hours", "distance_km", "cargo_value_inr", "threshold_temp_c",
    "humidity_threshold_pct", "latitude", "longitude",
]

CITY_COORDS = {
    "Delhi": (28.6139, 77.2090), "Mumbai": (19.0760, 72.8777),
    "Pune": (18.5204, 73.8567), "Jaipur": (26.9124, 75.7873),
    "Ahmedabad": (23.0225, 72.5714), "Lucknow": (26.8467, 80.9462),
    "Bengaluru": (12.9716, 77.5946), "Hyderabad": (17.3850, 78.4867),
    "Kolkata": (22.5726, 88.3639), "Chennai": (13.0827, 80.2707),
    "Indore": (22.7196, 75.8577), "Nagpur": (21.1458, 79.0882),
    "Haridwar": (29.9457, 78.1642), "Chandigarh": (30.7333, 76.7794),
}


def make_demo_data() -> pd.DataFrame:
    rng = np.random.default_rng(17)
    lanes = [
        ("Delhi", "Mumbai"), ("Pune", "Bengaluru"), ("Jaipur", "Ahmedabad"),
        ("Lucknow", "Kolkata"), ("Hyderabad", "Chennai"), ("Haridwar", "Delhi"),
        ("Mumbai", "Indore"), ("Nagpur", "Pune"), ("Chandigarh", "Jaipur"),
        ("Bengaluru", "Hyderabad"), ("Kolkata", "Lucknow"), ("Ahmedabad", "Mumbai"),
    ]
    products = ["Fresh produce", "Pharmaceuticals", "Dairy", "Floral", "Specialty chemicals"]
    statuses = ["In transit", "In transit", "At hub", "Dispatched"]
    rows = []
    now = datetime.now().replace(minute=0, second=0, microsecond=0)
    for i in range(1, 37):
        origin, dest = lanes[(i - 1) % len(lanes)]
        product = products[(i - 1) % len(products)]
        if product == "Pharmaceuticals":
            temp_threshold, humidity_threshold = 8.0, 65.0
            temp = rng.normal(7.2, 2.1)
        elif product == "Dairy":
            temp_threshold, humidity_threshold = 6.0, 70.0
            temp = rng.normal(5.4, 2.0)
        elif product == "Floral":
            temp_threshold, humidity_threshold = 18.0, 78.0
            temp = rng.normal(17.6, 3.8)
        elif product == "Fresh produce":
            temp_threshold, humidity_threshold = 12.0, 82.0
            temp = rng.normal(11.7, 3.5)
        else:
            temp_threshold, humidity_threshold = 25.0, 60.0
            temp = rng.normal(24.0, 4.0)
        humidity = rng.normal(70 if product != "Specialty chemicals" else 55, 14)
        rainfall = max(0, rng.gamma(1.3, 5.0) - 2)
        baseline = float(rng.integers(8, 42))
        transit = baseline * rng.uniform(.78, 1.35)
        # Add a small number of deliberate anomaly cases so the demo makes
        # the high-risk workflow visible during a presentation.
        if i in {4, 11, 18, 25, 32}:
            temp = max(temp, temp_threshold + 5.5)
            humidity = max(humidity, humidity_threshold + 20)
            rainfall = max(rainfall, 27.0)
            transit = baseline * 1.42
        lat, lon = CITY_COORDS[dest]
        rows.append({
            "shipment_id": f"ATM-{1000+i}", "origin": origin, "destination": dest,
            "product": product, "status": rng.choice(statuses),
            "temperature_c": round(float(np.clip(temp, -2, 42)), 1),
            "humidity_pct": round(float(np.clip(humidity, 20, 100)), 1),
            "rainfall_mm": round(float(rainfall), 1),
            "wind_kmh": round(float(rng.uniform(4, 48)), 1),
            "transit_hours": round(float(transit), 1),
            "baseline_hours": round(float(baseline), 1),
            "distance_km": int(rng.integers(180, 1800)),
            "cargo_value_inr": int(rng.integers(250000, 4200000)),
            "threshold_temp_c": temp_threshold,
            "humidity_threshold_pct": humidity_threshold,
            "latitude": lat + float(rng.normal(0, .08)),
            "longitude": lon + float(rng.normal(0, .08)),
            "updated_at": (now - timedelta(minutes=int(rng.integers(2, 95)))).isoformat(),
            "alternate_route": f"{origin}–{dest} via alternate corridor",
            "alt_temp_delta_c": round(float(rng.uniform(-4.2, 1.2)), 1),
            "alt_delay_delta_h": round(float(rng.uniform(-1.5, 3.0)), 1),
        })
    return pd.DataFrame(rows)


def normalise_upload(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = [str(c).strip().lower().replace(" ", "_") for c in frame.columns]
    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))

    numeric_defaults = {
        "wind_kmh": 0, "alt_temp_delta_c": -1.5, "alt_delay_delta_h": 0,
        "rainfall_mm": 0, "latitude": 20.5937, "longitude": 78.9629,
    }
    for col, default in numeric_defaults.items():
        if col not in frame.columns:
            frame[col] = default
    if "alternate_route" not in frame.columns:
        frame["alternate_route"] = frame["origin"].astype(str) + "–" + frame["destination"].astype(str) + " alternate"
    for col in [c for c in REQUIRED_COLUMNS if c not in {"shipment_id", "origin", "destination", "product", "status"}]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    for col in ["shipment_id", "origin", "destination", "product", "status"]:
        frame[col] = frame[col].fillna("Unknown").astype(str)
    for col, default in numeric_defaults.items():
        frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(default)
    frame["updated_at"] = frame.get("updated_at", pd.Series([datetime.now().isoformat()] * len(frame))).astype(str)
    return frame


def score_risk(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    temp_excess = (df["temperature_c"] - df["threshold_temp_c"]).clip(lower=0)
    humidity_excess = (df["humidity_pct"] - df["humidity_threshold_pct"]).clip(lower=0)
    temp_risk = ((temp_excess + 1.5) / 10 * 100).clip(0, 100)
    humidity_risk = ((humidity_excess + 5) / 30 * 100).clip(0, 100)
    rain_risk = (df["rainfall_mm"] / 35 * 100).clip(0, 100)
    delay_pct = ((df["transit_hours"] - df["baseline_hours"]) / df["baseline_hours"].clip(lower=1) * 100).clip(0, 100)
    df["risk_score"] = (0.45 * temp_risk + 0.30 * humidity_risk + 0.15 * rain_risk + 0.10 * delay_pct).round(0).astype(int)
    df["risk_level"] = np.select(
        [df["risk_score"] >= 65, df["risk_score"] >= 35],
        ["High", "Moderate"], default="Low"
    )
    df["delay_hours"] = (df["transit_hours"] - df["baseline_hours"]).clip(lower=0).round(1)
    df["exposure_value_inr"] = (df["cargo_value_inr"] * df["risk_score"] / 100).round(0)
    df["estimated_protected_inr"] = (df["exposure_value_inr"] * 0.35).round(0)
    df["route"] = df["origin"].astype(str) + " → " + df["destination"].astype(str)
    return df


def make_telemetry(shipment: pd.Series) -> pd.DataFrame:
    # Synthetic illustrative 24-hour trace anchored to the latest shipment reading.
    rng = np.random.default_rng(sum(ord(c) for c in str(shipment["shipment_id"])))
    times = pd.date_range(end=pd.Timestamp.now().floor("h"), periods=24, freq="h")
    t_base = float(shipment["temperature_c"])
    h_base = float(shipment["humidity_pct"])
    phase = np.linspace(-1.8, 1.8, 24)
    return pd.DataFrame({
        "time": times,
        "temperature_c": np.clip(t_base + np.sin(np.linspace(0, 3.5*np.pi, 24))*1.1 + rng.normal(0, .32, 24), -5, 50),
        "humidity_pct": np.clip(h_base + np.cos(np.linspace(0, 2.4*np.pi, 24))*5 + rng.normal(0, 1.8, 24), 10, 100),
    })


def inr(value: float) -> str:
    value = float(value)
    if value >= 10_000_000:
        return f"₹{value/10_000_000:.2f} Cr"
    if value >= 100_000:
        return f"₹{value/100_000:.2f} L"
    return f"₹{value:,.0f}"


def chart_layout(fig: go.Figure, height: int = 300) -> go.Figure:
    fig.update_layout(
        height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans", color="#b7c8d4", size=11),
        margin=dict(l=8, r=8, t=24, b=8),
        legend=dict(orientation="h", y=1.12, x=0, font=dict(size=10)),
        hoverlabel=dict(bgcolor="#142a3b", bordercolor="#355267", font_color="#ecf5f8"),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor="rgba(150,180,200,.15)")
    fig.update_yaxes(gridcolor="rgba(150,180,200,.10)", zeroline=False)
    return fig


# Sidebar: data source and controls
with st.sidebar:
    st.markdown("## 🌦️ AtmoSync")
    st.caption("MICRO-CLIMATE INTELLIGENCE")
    st.markdown("---")
    st.markdown("**Data source**")
    data_mode = st.radio("Choose data", ["Demo environment", "Upload shipment CSV"], label_visibility="collapsed")
    demo = make_demo_data()
    st.download_button(
        "⬇ Download CSV template",
        data=demo.head(5).to_csv(index=False).encode("utf-8"),
        file_name="atmosync_shipment_template.csv",
        mime="text/csv",
        width="stretch",
    )
    uploaded = None
    if data_mode == "Upload shipment CSV":
        uploaded = st.file_uploader("Upload your shipment data", type=["csv"])
        st.caption("Use the template to see the required column names.")
    st.markdown("---")

    raw = demo
    source_label = "SYNTHETIC DEMO DATA"
    if data_mode == "Upload shipment CSV" and uploaded is not None:
        try:
            raw = normalise_upload(pd.read_csv(uploaded))
            source_label = "UPLOADED DATA"
            st.success(f"Loaded {len(raw)} shipments")
        except Exception as exc:
            st.error(f"Could not load CSV: {exc}")
            st.info("Showing demo data until a valid CSV is uploaded.")
    elif data_mode == "Upload shipment CSV":
        st.info("Upload a CSV to replace the demo data. The dashboard is currently showing the sample environment.")

    data = score_risk(raw)
    st.markdown("**Filters**")
    all_products = sorted(data["product"].dropna().astype(str).unique().tolist())
    all_origins = sorted(data["origin"].dropna().astype(str).unique().tolist())
    all_destinations = sorted(data["destination"].dropna().astype(str).unique().tolist())
    products = st.multiselect("Cargo type", all_products, default=all_products)
    origins = st.multiselect("Origin", all_origins, default=all_origins)
    destinations = st.multiselect("Destination", all_destinations, default=all_destinations)
    risk_filter = st.multiselect("Risk level", ["High", "Moderate", "Low"], default=["High", "Moderate", "Low"])
    st.markdown("---")
    st.caption("AtmoSync prototype • Decision support, not a safety certification")

filtered = data[
    data["product"].isin(products)
    & data["origin"].isin(origins)
    & data["destination"].isin(destinations)
    & data["risk_level"].isin(risk_filter)
].copy()

# Header
st.markdown(
    f"""
    <div class="hero">
      <div class="eyebrow">Climate-aware supply chain control tower</div>
      <div class="hero-title">See the weather risk <span style="color:#54e4cc">inside the journey.</span></div>
      <p class="hero-copy">Monitor shipment-level temperature, humidity and weather exposure. Spot vulnerable lanes early and compare alternate-route conditions before cargo quality is affected.</p>
      <div style="display:flex;gap:10px;align-items:center;margin-top:16px;flex-wrap:wrap">
        <span class="status-pill">● SYSTEM ONLINE</span>
        <span style="color:#91a9b8;font-size:.72rem">{source_label}</span>
        <span style="color:#91a9b8;font-size:.72rem">{datetime.now().strftime('%d %b %Y · %H:%M')} local dashboard time</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if filtered.empty:
    st.warning("No shipments match the selected filters. Adjust the filters in the sidebar.")
    st.stop()

# KPI strip
n_shipments = len(filtered)
high_count = int((filtered["risk_level"] == "High").sum())
moderate_count = int((filtered["risk_level"] == "Moderate").sum())
avg_risk = float(filtered["risk_score"].mean())
exposure = float(filtered["exposure_value_inr"].sum())
protected = float(filtered["estimated_protected_inr"].sum())
avg_delay = float(filtered["delay_hours"].mean())

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Shipments monitored", f"{n_shipments:,}", "Across selected lanes")
k2.metric("High-risk shipments", f"{high_count:,}", f"{moderate_count} moderate", delta_color="inverse")
k3.metric("Average climate risk", f"{avg_risk:.0f}/100", "Composite exposure score")
k4.metric("Cargo exposure value", inr(exposure), "Risk-weighted cargo value")
k5.metric("Potential value protected", inr(protected), "Illustrative 35% mitigation scenario")

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
tab_overview, tab_shipments, tab_intelligence = st.tabs(["◈  Control tower", "▤  Shipment explorer", "✧  Risk intelligence"])

with tab_overview:
    left, right = st.columns([1.45, 1], gap="large")
    with left:
        st.markdown("<div class='section-kicker'>Network overview</div>", unsafe_allow_html=True)
        st.subheader("Where shipments are exposed")
        map_df = filtered.copy()
        map_fig = px.scatter_geo(
            map_df, lat="latitude", lon="longitude", color="risk_level", size="risk_score",
            hover_name="shipment_id", hover_data={"origin": True, "destination": True, "product": True, "risk_score": True, "latitude": False, "longitude": False},
            color_discrete_map={"High": "#ff737e", "Moderate": "#f4bd66", "Low": "#4bd8be"},
            projection="mercator",
        )
        map_fig.update_geos(
            showcountries=True, countrycolor="rgba(139,169,190,.28)", showland=True,
            landcolor="#142a39", showocean=True, oceancolor="#0d1c29",
            showcoastlines=True, coastlinecolor="rgba(139,169,190,.22)",
            lataxis_range=[5, 36], lonaxis_range=[67, 98],
            bgcolor="rgba(0,0,0,0)",
        )
        map_fig.update_layout(
            height=385, paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=0,r=0,t=8,b=0),
            font=dict(color="#b7c8d4"), legend=dict(orientation="h", y=0, x=0),
        )
        st.plotly_chart(map_fig, width="stretch", config={"displayModeBar": False})
        st.caption("Map pins show destination-side shipment positions. Coordinates in demo mode are approximate city locations.")

    with right:
        st.markdown("<div class='section-kicker'>Live prioritisation</div>", unsafe_allow_html=True)
        st.subheader("Attention required")
        alerts = filtered.sort_values(["risk_score", "exposure_value_inr"], ascending=False).head(5)
        if alerts.empty:
            st.info("No shipments to prioritise.")
        for _, item in alerts.iterrows():
            klass = "alert-high" if item["risk_level"] == "High" else ("" if item["risk_level"] == "Moderate" else "alert-low")
            signal = "Temperature / humidity excursion" if item["temperature_c"] > item["threshold_temp_c"] or item["humidity_pct"] > item["humidity_threshold_pct"] else "Weather / transit exposure"
            st.markdown(
                f"<div class='alert-card {klass}'><div style='display:flex;justify-content:space-between;gap:8px'><span class='alert-title'>{item['shipment_id']} · {item['risk_level']} risk</span><span style='color:#d8e6ed;font-size:.78rem;font-weight:700'>{int(item['risk_score'])}/100</span></div><div class='alert-body'>{item['route']} · {item['product']}<br>{signal} · Exposure {inr(item['exposure_value_inr'])}</div></div>",
                unsafe_allow_html=True,
            )
        st.markdown("<div class='small-muted'>Suggested workflow: verify sensor reading → contact carrier → assess alternate corridor → record intervention.</div>", unsafe_allow_html=True)

    st.markdown("---")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("<div class='section-kicker'>Lane analytics</div>", unsafe_allow_html=True)
        st.subheader("Highest-risk transport lanes")
        lane = filtered.groupby("route", as_index=False).agg(avg_risk=("risk_score", "mean"), shipments=("shipment_id", "count"), exposure=("exposure_value_inr", "sum")).sort_values("avg_risk", ascending=True).tail(8)
        lane_fig = px.bar(lane, x="avg_risk", y="route", orientation="h", text=lane["avg_risk"].round(0).astype(int), color="avg_risk", color_continuous_scale=["#39cbb3", "#e9ba68", "#f16f7a"])
        lane_fig.update_traces(textposition="outside", cliponaxis=False, marker_line_width=0)
        lane_fig.update_layout(coloraxis_showscale=False, xaxis_title="Average risk score", yaxis_title="", height=315)
        chart_layout(lane_fig, 315)
        st.plotly_chart(lane_fig, width="stretch", config={"displayModeBar": False})
    with c2:
        st.markdown("<div class='section-kicker'>Risk composition</div>", unsafe_allow_html=True)
        st.subheader("Shipment risk distribution")
        risk_counts = filtered["risk_level"].value_counts().reindex(["High", "Moderate", "Low"], fill_value=0).reset_index()
        risk_counts.columns = ["risk_level", "shipments"]
        donut = px.pie(risk_counts, names="risk_level", values="shipments", hole=.68,
                       color="risk_level", color_discrete_map={"High":"#ff737e", "Moderate":"#f4bd66", "Low":"#4bd8be"})
        donut.update_traces(textinfo="percent+label", textfont_size=11, marker=dict(line=dict(color="#102130", width=3)))
        donut.update_layout(height=315, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            font=dict(color="#c5d4dd"), margin=dict(l=8,r=8,t=24,b=8), showlegend=False)
        st.plotly_chart(donut, width="stretch", config={"displayModeBar": False})

with tab_shipments:
    st.markdown("<div class='section-kicker'>Operational detail</div>", unsafe_allow_html=True)
    st.subheader("Shipment register")
    search = st.text_input("Search shipment, route or cargo", placeholder="e.g. ATM-1004, Delhi, Pharmaceuticals")
    view = filtered.copy()
    if search.strip():
        term = search.strip().lower()
        mask = view[["shipment_id", "origin", "destination", "product", "status"]].astype(str).apply(lambda col: col.str.lower().str.contains(term)).any(axis=1)
        view = view[mask]
    show = view[["shipment_id", "route", "product", "status", "temperature_c", "humidity_pct", "rainfall_mm", "transit_hours", "baseline_hours", "risk_score", "risk_level", "exposure_value_inr"]].copy()
    show = show.rename(columns={
        "shipment_id":"Shipment ID", "route":"Route", "product":"Cargo", "status":"Status",
        "temperature_c":"Temp °C", "humidity_pct":"Humidity %", "rainfall_mm":"Rain mm",
        "transit_hours":"Transit h", "baseline_hours":"Baseline h", "risk_score":"Risk /100",
        "risk_level":"Risk band", "exposure_value_inr":"Exposure ₹",
    })
    st.dataframe(
        show.style.format({"Temp °C":"{:.1f}", "Humidity %":"{:.1f}", "Rain mm":"{:.1f}", "Transit h":"{:.1f}", "Baseline h":"{:.1f}", "Exposure ₹":"₹{:,.0f}"}),
        width="stretch", hide_index=True, height=420,
    )
    st.download_button("⬇ Export filtered shipments", data=view.to_csv(index=False).encode("utf-8"), file_name="atmosync_filtered_shipments.csv", mime="text/csv")

    st.markdown("---")
    st.markdown("<div class='section-kicker'>Shipment deep dive</div>", unsafe_allow_html=True)
    selected_id = st.selectbox("Choose a shipment", view["shipment_id"].astype(str).tolist() if not view.empty else filtered["shipment_id"].astype(str).tolist())
    selected = data[data["shipment_id"].astype(str) == str(selected_id)].iloc[0]
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Climate risk", f"{int(selected['risk_score'])}/100", selected["risk_level"])
    d2.metric("Temperature", f"{selected['temperature_c']:.1f} °C", f"Limit {selected['threshold_temp_c']:.1f} °C")
    d3.metric("Humidity", f"{selected['humidity_pct']:.1f}%", f"Limit {selected['humidity_threshold_pct']:.0f}%")
    d4.metric("Transit variance", f"{selected['delay_hours']:.1f} h", "Above baseline" if selected["delay_hours"] > 0 else "Within baseline")

    telem = make_telemetry(selected)
    trend = go.Figure()
    trend.add_trace(go.Scatter(x=telem["time"], y=telem["temperature_c"], name="Temperature", mode="lines", line=dict(color="#58c8ff", width=2.5), fill="tozeroy", fillcolor="rgba(88,200,255,.06)"))
    trend.add_hline(y=float(selected["threshold_temp_c"]), line_dash="dot", line_color="#f4bd66", annotation_text="Temperature limit", annotation_position="top left")
    trend.update_layout(yaxis_title="Temperature (°C)", xaxis_title="", height=290)
    chart_layout(trend, 290)
    st.plotly_chart(trend, width="stretch", config={"displayModeBar": False})
    st.caption("Illustrative 24-hour trace generated around the latest reading. Connect actual IoT telemetry to replace this demo series.")

with tab_intelligence:
    st.markdown("<div class='section-kicker'>Micro-climate arbitrage</div>", unsafe_allow_html=True)
    st.subheader("Find the safer corridor, not just the shortest route")
    st.write("Compare an indicative alternate corridor against the current route. A negative temperature delta suggests cooler conditions; a positive delay delta means the alternative may take longer.")
    compare = filtered.copy()
    compare["alt_risk_proxy"] = (compare["risk_score"] + compare["alt_temp_delta_c"] * 5 + compare["alt_delay_delta_h"] * 1.5).clip(0, 100)
    compare["risk_improvement"] = (compare["risk_score"] - compare["alt_risk_proxy"]).round(0)
    compare = compare.sort_values("risk_improvement", ascending=False)
    winners = compare.head(8)
    l, r = st.columns([1.1, 1], gap="large")
    with l:
        st.markdown("**Alternate-route opportunities**")
        opp = px.bar(winners.sort_values("risk_improvement"), x="risk_improvement", y="shipment_id", orientation="h",
                     color="risk_improvement", color_continuous_scale=["#ed7a83", "#e8bd6d", "#46d9bf"],
                     hover_data={"route":True, "alt_temp_delta_c":True, "alt_delay_delta_h":True})
        opp.update_layout(xaxis_title="Estimated risk points improved", yaxis_title="Shipment", coloraxis_showscale=False, height=340)
        chart_layout(opp, 340)
        st.plotly_chart(opp, width="stretch", config={"displayModeBar":False})
    with r:
        st.markdown("**Decision candidates**")
        for _, row in winners.head(5).iterrows():
            beneficial = row["risk_improvement"] > 0
            accent = "alert-low" if beneficial else ""
            verdict = "Potentially safer" if beneficial else "Validate before rerouting"
            st.markdown(
                f"<div class='alert-card {accent}'><div class='alert-title'>{row['shipment_id']} · {verdict}</div><div class='alert-body'>{row['route']}<br>Alt. temperature {row['alt_temp_delta_c']:+.1f} °C · Transit impact {row['alt_delay_delta_h']:+.1f} h · Estimated risk change {row['risk_improvement']:+.0f} pts</div></div>",
                unsafe_allow_html=True,
            )
    st.markdown("---")
    st.markdown("<div class='section-kicker'>How the prototype scores risk</div>", unsafe_allow_html=True)
    a, b, c, d = st.columns(4)
    a.metric("Temperature", "45%", "Weight in composite score")
    b.metric("Humidity", "30%", "Weight in composite score")
    c.metric("Rainfall", "15%", "Weight in composite score")
    d.metric("Transit delay", "10%", "Weight in composite score")
    st.info("Prototype methodology: the 0–100 score combines threshold exceedance and transit/weather indicators. Thresholds and weights are illustrative defaults; calibrate them by cargo type, packaging, sensor placement and historical quality outcomes before operational use.")

st.markdown(
    "<div class='footer'>ATMoSYNC · Micro-climate supply chain intelligence &nbsp;|&nbsp; Demo data is synthetic and should not be interpreted as live sensor or weather data. Potential protected value is an illustrative scenario, not a guaranteed saving.</div>",
    unsafe_allow_html=True,
)
