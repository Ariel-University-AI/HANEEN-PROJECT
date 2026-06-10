"""
TaxiWise — AI Transportation Intelligence Platform
Driver-first redesign: Live · My Shift · Analytics · Model
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime as _dt

st.set_page_config(
    page_title="TaxiWise",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.data_loader import load_trips, load_zones, compute_demand, compute_kpis
from src.model       import load_xgb_model, load_regression_model, predict_regression
import src.charts     as charts
import src.clustering as clust
import src.regression as reg

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
*{font-family:'Inter',sans-serif!important;box-sizing:border-box}

.main{background:#0E1117}
.block-container{padding:1rem 1.8rem;max-width:1440px}
section[data-testid="stSidebar"]{background:#080B12;border-right:1px solid rgba(255,255,255,.05)}

/* ── titles ── */
.page-title{font-size:1.75rem;font-weight:900;line-height:1.1;
  background:linear-gradient(90deg,#F7C948,#F97316);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
  margin-bottom:.15rem}
.page-sub{color:#6B7280;font-size:.82rem;margin-bottom:1rem}
.sec{font-size:.95rem;font-weight:700;color:#FAFAFA;
  border-left:3px solid #F7C948;padding-left:9px;margin:1.1rem 0 .7rem}

/* ── hero card (Live page) ── */
.hero-card{background:linear-gradient(135deg,#1A1D27,#1E2233);
  border:2px solid rgba(247,201,72,.30);border-radius:20px;padding:24px 28px;
  animation:pulse-gold 3s infinite}
.hero-label{font-size:.72rem;font-weight:700;color:#9CA3AF;
  text-transform:uppercase;letter-spacing:.08em;margin-bottom:6px}
.hero-zone{font-size:2rem;font-weight:900;color:#FAFAFA;line-height:1.15}
.hero-boro{font-size:.85rem;color:#9CA3AF;margin-bottom:12px}
.hero-demand{font-size:2.8rem;font-weight:900;color:#F7C948;line-height:1}
.hero-unit{font-size:.78rem;color:#9CA3AF;margin-bottom:10px}
.hero-rev{font-size:1.4rem;font-weight:800;color:#10B981;margin-bottom:10px}

/* ── demand badge ── */
.badge{display:inline-block;font-size:.8rem;font-weight:700;
  padding:5px 16px;border-radius:20px}
.b-extreme{background:rgba(239,68,68,.15);color:#FCA5A5;border:1px solid #EF4444}
.b-high{background:rgba(249,115,22,.15);color:#FDBA74;border:1px solid #F97316}
.b-medium{background:rgba(247,201,72,.12);color:#FDE68A;border:1px solid rgba(247,201,72,.4)}
.b-low{background:rgba(107,114,128,.12);color:#D1D5DB;border:1px solid #4B5563}

/* ── zone quick-card (top 5 list) ── */
.zone-quick{background:#1A1D27;border:1px solid rgba(255,255,255,.06);
  border-radius:12px;padding:12px 14px;margin-bottom:7px}
.zone-quick.r1{border-left:4px solid #EF4444}
.zone-quick.r2{border-left:4px solid #F97316}
.zone-quick.r3{border-left:4px solid #F7C948}
.zone-quick.r4{border-left:4px solid #3B82F6}
.zone-quick.r5{border-left:4px solid #8B5CF6}
.zq-name{font-size:.92rem;font-weight:700;color:#FAFAFA}
.zq-boro{font-size:.68rem;color:#6B7280;margin-bottom:3px}
.zq-stats{font-size:.76rem;color:#9CA3AF}
.zq-rev{font-size:.82rem;font-weight:700;color:#10B981;margin-top:2px}

/* ── KPI grid ── */
.kpi-grid{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:.8rem}
.kpi-card{background:#1A1D27;border:1px solid rgba(247,201,72,.10);
  border-radius:14px;padding:14px 18px;flex:1;min-width:120px;
  transition:transform .15s,border-color .15s}
.kpi-card:hover{transform:translateY(-2px);border-color:rgba(247,201,72,.28)}
.kpi-card.top{animation:pulse-gold 3s infinite}
.kpi-icon{font-size:1.3rem;margin-bottom:4px}
.kpi-value{font-size:1.4rem;font-weight:800;color:#F7C948;line-height:1}
.kpi-label{font-size:.68rem;color:#9CA3AF;text-transform:uppercase;letter-spacing:.05em;margin-top:4px}
.kpi-sub{font-size:.64rem;color:#6B7280;margin-top:2px}

/* ── prediction card ── */
.pred-card{background:linear-gradient(135deg,#1A1D27,#252836);
  border:2px solid rgba(247,201,72,.25);border-radius:18px;
  padding:20px 24px;text-align:center;margin-bottom:10px}
.pred-number{font-size:3.6rem;font-weight:900;color:#F7C948;line-height:1}
.pred-unit{color:#9CA3AF;font-size:.8rem;margin-top:4px}

/* ── revenue card ── */
.rev-card{background:#1A1D27;border:1px solid rgba(16,185,129,.20);
  border-radius:14px;padding:14px 18px;margin-bottom:10px}

/* ── alerts ── */
.alert-extreme{background:rgba(239,68,68,.08);border-left:4px solid #EF4444;
  border-radius:8px;padding:11px 15px;margin:6px 0;color:#FCA5A5;font-size:.84rem;font-weight:600}
.alert-high{background:rgba(249,115,22,.08);border-left:4px solid #F97316;
  border-radius:8px;padding:11px 15px;margin:6px 0;color:#FDBA74;font-size:.84rem;font-weight:600}
.alert-ok{background:rgba(16,185,129,.07);border-left:4px solid #10B981;
  border-radius:8px;padding:11px 15px;margin:6px 0;color:#6EE7B7;font-size:.84rem;font-weight:600}

/* ── insight card ── */
.insight{background:#1E2130;border-left:3px solid #F7C948;border-radius:6px;
  padding:9px 13px;margin-bottom:7px}
.insight b{color:#FAFAFA}
.insight-detail{color:#9CA3AF;font-size:.75rem;margin-top:2px}

/* ── relocation card ── */
.reloc-card{background:linear-gradient(135deg,#1A1D27,#252836);
  border-radius:14px;padding:18px 22px}

/* ── info banners ── */
.banner{background:rgba(59,130,246,.07);border:1px solid rgba(59,130,246,.20);
  border-radius:10px;padding:8px 14px;color:#93C5FD;font-size:.79rem;margin:.5rem 0}
.warn-banner{background:rgba(249,115,22,.07);border:1px solid rgba(249,115,22,.22);
  border-radius:10px;padding:8px 14px;color:#FDBA74;font-size:.79rem;margin:.5rem 0}

/* ── tabs ── */
[data-testid="stTabs"] button{font-weight:600;color:#9CA3AF!important}
[data-testid="stTabs"] button[aria-selected="true"]{color:#F7C948!important}

/* ── sidebar radio highlight ── */
[data-testid="stRadio"] label{font-size:.88rem;padding:7px 10px}

@keyframes pulse-gold{
  0%{box-shadow:0 0 0 rgba(247,201,72,0)}
  50%{box-shadow:0 0 20px rgba(247,201,72,.15)}
  100%{box-shadow:0 0 0 rgba(247,201,72,0)}}
::-webkit-scrollbar{width:5px;height:5px}
::-webkit-scrollbar-track{background:#080B12}
::-webkit-scrollbar-thumb{background:#2D3044;border-radius:3px}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Data bootstrap
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _bootstrap():
    return load_trips(), load_zones(), compute_demand()


with st.spinner("Loading NYC Taxi data …"):
    try:
        df_all, zones, demand = _bootstrap()
        if df_all.empty:
            st.error("Data not loaded. Run `python prepare_data.py`.")
            st.stop()
    except Exception as _e:
        st.error(f"Data error: {_e}")
        st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    _now_ts = _dt.now()
    st.markdown(f"""
    <div style="padding:10px 0 14px">
      <div style="font-size:1.5rem;font-weight:900;color:#F7C948">🚕 TaxiWise</div>
      <div style="color:#6B7280;font-size:.71rem;margin-top:2px">
        {_now_ts.strftime("%a %b %d  ·  %H:%M")}
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="height:1px;background:rgba(255,255,255,.06);margin-bottom:12px"></div>',
                unsafe_allow_html=True)

    PAGES = {
        "🚕  Live":       "live",
        "📋  My Shift":   "shift",
        "📈  Analytics":  "analytics",
        "🔬  Model":      "model",
    }
    page_key = PAGES[st.radio("nav", list(PAGES.keys()), label_visibility="collapsed")]

    st.markdown('<div style="height:1px;background:rgba(255,255,255,.06);margin:12px 0"></div>',
                unsafe_allow_html=True)
    st.markdown(f"""
    <div style="color:#4B5563;font-size:.68rem;line-height:1.85">
      <b style="color:#6B7280">Data</b><br>
      NYC Yellow Taxi 2023–2026<br>
      {len(df_all):,} trips loaded<br><br>
      <b style="color:#6B7280">Models</b><br>
      Random Forest · XGBoost
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Constants + today defaults
# ─────────────────────────────────────────────────────────────────────────────
_DOW   = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
_MON   = ["January","February","March","April","May","June",
          "July","August","September","October","November","December"]
_MONS  = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
_DRK   = dict(template="plotly_dark", paper_bgcolor="#1A1D27",
              plot_bgcolor="#1A1D27", font=dict(color="#FAFAFA"), height=340)

_today     = _dt.now()
_now_year  = _today.year
_now_mon   = _today.month
_now_dow   = _today.weekday()
_now_hour  = _today.hour
_YEAR_LIST = list(range(2023, 2036))
_year_idx  = _YEAR_LIST.index(_now_year) if _now_year in _YEAR_LIST else 3


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _zone_data() -> tuple[list[str], dict[str, int]]:
    """Clean zone labels (no IDs) + lookup dict → LocationID."""
    labels: list[str] = []
    lut: dict[str, int] = {}
    for _, r in zones.iterrows():
        lbl = f"{r['Zone']} — {r['Borough']}"
        labels.append(lbl)
        lut[lbl] = int(r["LocationID"])
    return labels, lut


def _zone_defaults(loc_id: int) -> dict:
    z = demand[demand["PULocationID"] == loc_id]
    s = z if not z.empty else demand
    return {
        "hist": float(s["zone_total_trips"].iloc[0]) if not z.empty
                else float(demand["zone_total_trips"].median()),
        "fare": max(1.0,  min(500.0, float(s["avg_fare"].mean()))),
        "dist": max(0.1,  min(100.0, float(s["avg_distance"].mean()))),
        "dur":  max(1.0,  min(300.0, float(s["avg_duration"].mean()))),
    }


def _demand_level(val: float, ref: pd.Series) -> tuple[str, str]:
    """Return (level_name, css_class) based on percentile rank in ref."""
    p25 = float(ref.quantile(0.25))
    p75 = float(ref.quantile(0.75))
    p90 = float(ref.quantile(0.90))
    if   val >= p90: return "Very High", "b-extreme"
    elif val >= p75: return "High",      "b-high"
    elif val >= p25: return "Medium",    "b-medium"
    else:            return "Low",       "b-low"


def _badge(level: str, cls: str) -> str:
    emoji = {"Very High":"🔴","High":"🟠","Medium":"🟡","Low":"🟢"}.get(level,"⚪")
    return f'<span class="badge {cls}">{emoji} {level} Demand</span>'


def _kpi_row(items: list, top_idx: int = 0):
    html = '<div class="kpi-grid">'
    for i, (icon, val, label, sub) in enumerate(items):
        cls = "kpi-card top" if i == top_idx else "kpi-card"
        html += (f'<div class="{cls}"><div class="kpi-icon">{icon}</div>'
                 f'<div class="kpi-value">{val}</div>'
                 f'<div class="kpi-label">{label}</div>'
                 + (f'<div class="kpi-sub">{sub}</div>' if sub else "")
                 + "</div>")
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def _section(t: str):
    st.markdown(f'<div class="sec">{t}</div>', unsafe_allow_html=True)


def _pchart(fig, h: int | None = None, **kw):
    if h:
        fig.update_layout(height=h)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, **kw)


# ─────────────────────────────────────────────────────────────────────────────
# Demand map helpers (XGBoost-based, no year feature)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _zone_preds(hour: int, dow: int, month: int) -> pd.DataFrame:
    from src.model import FEATURE_COLS
    from src.zone_coords import get_zone_coord
    model, *_ = load_xgb_model()

    zs = (demand.groupby("PULocationID")
          .agg(avg_fare        =("avg_fare",         "mean"),
               avg_distance    =("avg_distance",     "mean"),
               avg_duration    =("avg_duration",     "mean"),
               zone_total_trips=("zone_total_trips", "first"),
               hist_demand     =("trip_count",       "mean"))
          .reset_index())
    zs["hour"]  = hour
    zs["dow"]   = dow
    zs["month"] = month
    for col in ["avg_fare","avg_distance","avg_duration","zone_total_trips"]:
        zs[col] = zs[col].fillna(float(demand[col].median()))
    zs = zs.dropna(subset=FEATURE_COLS)
    if zs.empty:
        return pd.DataFrame()
    zs["predicted_demand"] = np.maximum(model.predict(zs[FEATURE_COLS].values), 0)

    merged = zs.merge(zones[["LocationID","Zone","Borough"]],
                      left_on="PULocationID", right_on="LocationID", how="left")
    merged["Zone"]    = merged["Zone"].fillna("Zone "+merged["PULocationID"].astype(str))
    merged["Borough"] = merged["Borough"].fillna("Unknown")
    coords = [get_zone_coord(int(r["PULocationID"]), str(r["Borough"]))
              for _, r in merged.iterrows()]
    merged["lat"] = [c[0] for c in coords]
    merged["lon"] = [c[1] for c in coords]

    # 4-level demand classification using percentile thresholds
    p25 = float(np.percentile(merged["predicted_demand"], 25))
    p75 = float(np.percentile(merged["predicted_demand"], 75))
    p90 = float(np.percentile(merged["predicted_demand"], 90))
    merged["Demand Level"] = merged["predicted_demand"].apply(
        lambda x: "Very High" if x >= p90 else
                  ("High"     if x >= p75 else
                  ("Medium"   if x >= p25 else "Low")))

    merged["Predicted Trips/hr"] = merged["predicted_demand"].round(1)
    merged["Avg Fare ($)"]       = merged["avg_fare"].round(2)
    merged["Revenue est ($/hr)"] = (merged["predicted_demand"] * merged["avg_fare"] * 0.7).round(2)

    # Driver Opportunity Score 0–100: 65% demand rank + 35% fare rank
    d_min, d_max = float(merged["predicted_demand"].min()), float(merged["predicted_demand"].max())
    f_min, f_max = float(merged["avg_fare"].min()),         float(merged["avg_fare"].max())
    d_norm = (merged["predicted_demand"] - d_min) / max(d_max - d_min, 0.001)
    f_norm = (merged["avg_fare"]         - f_min) / max(f_max - f_min, 0.001)
    merged["Opportunity Score"] = (
        (d_norm * 0.65 + f_norm * 0.35) * 100
    ).clip(0, 100).round(0).astype(int)
    return merged


# Green → Yellow → Orange → Red  (Low → Medium → High → Very High)
_MAP_COLORS = [
    [0.00, "#10B981"],
    [0.33, "#FACC15"],
    [0.66, "#F97316"],
    [1.00, "#EF4444"],
]

_MAP_CENTER = {"lat": 40.730, "lon": -73.985}


def _build_map(
    merged: pd.DataFrame,
    sel_id: int | None = None,
    height: int = 400,
    mode: str = "scatter",
) -> "go.Figure":
    import plotly.express as px
    import plotly.graph_objects as go

    if merged.empty:
        return go.Figure()

    # ── Heatmap (density) mode ───────────────────────────────────────────────
    if mode == "heatmap":
        fig = px.density_mapbox(
            merged, lat="lat", lon="lon", z="predicted_demand",
            radius=26,
            center=_MAP_CENTER, zoom=10,
            mapbox_style="carto-darkmatter",
            color_continuous_scale=_MAP_COLORS,
            opacity=0.82,
        )
        fig.update_layout(
            height=height, paper_bgcolor="#1A1D27",
            font=dict(color="#FAFAFA"),
            coloraxis_colorbar=dict(
                title="Trips/hr", tickfont=dict(color="#9CA3AF"),
                thickness=12, len=0.55,
            ),
            margin={"r":0,"t":0,"l":0,"b":0},
        )
        # Overlay best zone marker
        if sel_id is not None:
            row = merged[merged["PULocationID"] == sel_id]
            if not row.empty:
                zname = str(row["Zone"].iloc[0])
                fig.add_trace(go.Scattermapbox(
                    lat=row["lat"], lon=row["lon"], mode="markers+text",
                    marker=dict(size=22, color="#F7C948", opacity=1.0),
                    text=[f"⭐ {zname}"], textposition="top center",
                    textfont=dict(color="#F7C948", size=11),
                    hoverinfo="skip", showlegend=False,
                ))
        return fig

    # ── Scatter mode (default) ───────────────────────────────────────────────
    merged = merged.copy()
    has_opp = "Opportunity Score" in merged.columns

    # Rich hover tooltip
    merged["_hover"] = merged.apply(lambda r: (
        f"<b>{r['Zone']}</b><br>"
        f"<span>{r['Borough']}</span><br>"
        f"───────────────────<br>"
        f"🔮 Demand: <b>{float(r['Predicted Trips/hr']):.0f} trips/hr</b><br>"
        + (f"⭐ Opportunity: <b>{int(r['Opportunity Score'])}/100</b><br>" if has_opp else "")
        + f"💰 Avg Fare: <b>${float(r['Avg Fare ($)']):.2f}</b><br>"
        f"💵 Revenue Est: <b>${float(r['Revenue est ($/hr)']):.2f}/hr</b><br>"
        f"📊 Level: <b>{r['Demand Level']}</b>"
    ), axis=1)

    # Marker size: demand-proportional, clamped to readable range
    d_max = float(merged["predicted_demand"].max())
    merged["_size"] = ((merged["predicted_demand"] / max(d_max, 1)) * 22 + 7).clip(7, 29)

    fig = go.Figure(go.Scattermapbox(
        lat=merged["lat"],
        lon=merged["lon"],
        mode="markers",
        marker=go.scattermapbox.Marker(
            size=merged["_size"],
            color=merged["predicted_demand"],
            colorscale=_MAP_COLORS,
            cmin=float(merged["predicted_demand"].min()),
            cmax=float(merged["predicted_demand"].max()),
            colorbar=dict(
                title="Trips/hr",
                tickfont=dict(color="#9CA3AF", size=10),
                titlefont=dict(color="#9CA3AF", size=11),
                thickness=12, len=0.55, x=1.01,
            ),
            opacity=0.90,
        ),
        text=merged["_hover"],
        hoverinfo="text",
        hoverlabel=dict(
            bgcolor="#1A1D27",
            bordercolor="rgba(247,201,72,.45)",
            font=dict(color="#FAFAFA", size=12),
        ),
    ))

    # Best zone: outer glow ring + inner gold dot + label
    if sel_id is not None:
        row = merged[merged["PULocationID"] == sel_id]
        if not row.empty:
            zname = str(row["Zone"].iloc[0])
            fig.add_trace(go.Scattermapbox(           # glow
                lat=row["lat"], lon=row["lon"], mode="markers",
                marker=dict(size=42, color="#F7C948", opacity=0.18),
                hoverinfo="skip", showlegend=False,
            ))
            fig.add_trace(go.Scattermapbox(           # ring
                lat=row["lat"], lon=row["lon"], mode="markers",
                marker=dict(size=30, color="#F7C948", opacity=0.45),
                hoverinfo="skip", showlegend=False,
            ))
            fig.add_trace(go.Scattermapbox(           # dot + label
                lat=row["lat"], lon=row["lon"], mode="markers+text",
                marker=dict(size=16, color="#F7C948", opacity=1.0),
                text=[f"⭐ {zname}"], textposition="top center",
                textfont=dict(color="#F7C948", size=11),
                hoverinfo="skip", showlegend=False,
            ))

    fig.update_layout(
        height=height, paper_bgcolor="#1A1D27",
        font=dict(color="#FAFAFA"),
        margin={"r":0,"t":0,"l":0,"b":0},
        mapbox=dict(style="carto-darkmatter", zoom=10, center=_MAP_CENTER),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Auto-insights (Analytics page)
# ─────────────────────────────────────────────────────────────────────────────
def _auto_insights(df_in: pd.DataFrame, kpis_in: dict) -> list[tuple]:
    out: list[tuple] = []
    ph = kpis_in["peak_hour"]
    if 7 <= ph <= 9:
        out.append(("🌅","Morning Rush Dominates",
                    f"Peak demand at {ph}:00 — NYC commuters drive highest ridership","#F97316"))
    elif 17 <= ph <= 19:
        out.append(("🌆","Evening Rush Peak",
                    f"Peak demand at {ph}:00 — evening commute fuels trip volume","#F97316"))
    else:
        out.append(("🌙","Off-Peak Pattern",
                    f"Unusual peak at {ph}:00 — possible event or nightlife cluster","#8B5CF6"))

    tz = str(kpis_in.get("top_zone",""))
    if any(k in tz for k in ("Airport","JFK","LaGuardia","EWR")):
        out.append(("✈️","Airport Demand Spike",
                    f"{tz[:30]} leads — airport travel drives volume","#3B82F6"))
    elif "Midtown" in tz:
        out.append(("🏙️","Midtown Hotspot",
                    f"{tz[:30]} is #1 — business district, steady demand","#F7C948"))
    else:
        out.append(("📍","Top Zone",
                    f"{tz[:30]} leads in total trip volume","#10B981"))

    cp = kpis_in.get("credit_pct", 0)
    if cp > 70:
        out.append(("💳","Digital-First City",
                    f"{cp:.0f}% of trips paid by card — NYC trending cashless","#10B981"))

    if "dow" in df_in.columns and len(df_in) > 100:
        we = df_in[df_in["dow"] >= 5]
        wd = df_in[df_in["dow"] <  5]
        if len(we) > 0 and len(wd) > 0:
            we_avg = len(we) / max(we["dow"].nunique(), 1)
            wd_avg = len(wd) / max(wd["dow"].nunique(), 1)
            if we_avg > wd_avg * 1.08:
                out.append(("🎉","Weekend Surge",
                            "Weekends outpace weekdays — leisure trips boost revenue","#8B5CF6"))
            else:
                out.append(("💼","Weekday Dominance",
                            "Weekdays drive more trips — commuter demand leads","#3B82F6"))
    return out[:4]


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Live  (driver home: map + instant recommendation)
# ═════════════════════════════════════════════════════════════════════════════
def page_live():
    st.markdown('<div class="page-title">Live Demand</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Where to go right now · Drag the hour slider to preview upcoming demand</div>',
                unsafe_allow_html=True)

    # ── Time controls ────────────────────────────────────────────────────────
    tc1, tc2, tc3 = st.columns([2, 1.8, 1.2])
    with tc1:
        live_hour = st.slider("⏰ Hour", 0, 23, _now_hour, key="lv_hour",
                              help="Drag to preview demand at any hour of the day")
    with tc2:
        live_date    = st.date_input("📅 Date", value=_today.date(),
                                     format="DD/MM/YYYY", key="lv_date")
        live_dow     = live_date.weekday()
        live_dow_lbl = _DOW[live_dow]
        live_mon     = live_date.month
    with tc3:
        map_mode = st.radio("🗺️ Map View", ["🔵 Scatter", "🌡️ Heatmap"],
                            horizontal=True, key="lv_mapmode")

    with st.spinner("Computing live demand …"):
        zp = _zone_preds(live_hour, live_dow, live_mon)

    if zp.empty:
        st.error("Could not compute demand predictions.")
        return

    best_row  = zp.nlargest(1, "predicted_demand").iloc[0]
    best_id   = int(best_row["PULocationID"])
    best_name = str(best_row.get("Zone", f"Zone {best_id}"))
    best_boro = str(best_row.get("Borough", ""))
    best_dem  = float(best_row["predicted_demand"])
    best_fare = float(best_row.get("avg_fare", 15.0))
    best_rev  = float(best_row.get("Revenue est ($/hr)", best_dem * best_fare * 0.7))
    best_opp  = int(best_row.get("Opportunity Score", 0))
    level, lcls = _demand_level(best_dem, zp["predicted_demand"])

    # ── Layout: hero + map ───────────────────────────────────────────────────
    col_hero, col_map = st.columns([1, 1.65], gap="large")

    with col_hero:
        st.markdown(f"""
        <div class="hero-card">
          <div class="hero-label">🤖 Best Zone Right Now</div>
          <div class="hero-zone">{best_name}</div>
          <div class="hero-boro">{best_boro} · {live_dow_lbl[:3]} {live_hour:02d}:00</div>
          <div class="hero-demand">{best_dem:.0f}</div>
          <div class="hero-unit">predicted trips / hour</div>
          <div class="hero-rev">💵 ~${best_rev:.2f}/hr revenue est.</div>
          <div style="display:flex;align-items:center;gap:10px;margin-top:8px;flex-wrap:wrap">
            {_badge(level, lcls)}
            <span style="background:rgba(247,201,72,.12);color:#F7C948;
              border:1px solid rgba(247,201,72,.35);font-size:.8rem;font-weight:700;
              padding:5px 14px;border-radius:20px">⭐ {best_opp}/100 Score</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        _section("Top 5 Zones")
        top5   = zp.nlargest(5, "predicted_demand").reset_index(drop=True)
        colors = ["r1","r2","r3","r4","r5"]
        emojis = ["🥇","🥈","🥉","4.","5."]
        cards  = ""
        for i, (_, row) in enumerate(top5.iterrows()):
            zn   = str(row.get("Zone",""))
            bo   = str(row.get("Borough",""))
            pd_  = float(row["predicted_demand"])
            af   = float(row.get("avg_fare", 0))
            rv   = float(row.get("Revenue est ($/hr)", pd_ * af * 0.7))
            opp  = int(row.get("Opportunity Score", 0))
            cards += f"""
            <div class="zone-quick {colors[i]}">
              <div class="zq-name">{emojis[i]} {zn}</div>
              <div class="zq-boro">{bo}</div>
              <div class="zq-stats">🔮 <b>{pd_:.0f}</b> trips/hr &nbsp;·&nbsp; 💰 ${af:.2f} avg fare</div>
              <div style="display:flex;justify-content:space-between;align-items:center;margin-top:3px">
                <div class="zq-rev">💵 ~${rv:.2f}/hr</div>
                <div style="font-size:.72rem;color:#F7C948;font-weight:700">⭐ {opp}/100</div>
              </div>
            </div>"""
        st.markdown(cards, unsafe_allow_html=True)

        # Demand distribution
        _section("Demand Distribution")
        _lv_colors = {"Very High":"#EF4444","High":"#F97316","Medium":"#FACC15","Low":"#10B981"}
        dist_html = '<div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:4px">'
        for lv in ["Very High","High","Medium","Low"]:
            cnt = int((zp["Demand Level"] == lv).sum())
            clr = _lv_colors[lv]
            dist_html += (
                f'<div style="background:{clr}18;border:1px solid {clr}55;'
                f'border-radius:10px;padding:6px 12px;text-align:center">'
                f'<div style="font-size:1.1rem;font-weight:800;color:{clr}">{cnt}</div>'
                f'<div style="font-size:.65rem;color:#9CA3AF;margin-top:1px">{lv}</div>'
                f'</div>'
            )
        dist_html += '</div>'
        st.markdown(dist_html, unsafe_allow_html=True)

    with col_map:
        _section("🌡️ NYC Demand Map")
        _mode   = "heatmap" if "Heatmap" in map_mode else "scatter"
        fig_map = _build_map(zp, sel_id=best_id, height=500, mode=_mode)
        st.plotly_chart(fig_map, use_container_width=True,
                        config={"displayModeBar": True}, key="lv_map")

        # Color legend
        st.markdown("""
        <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;
             padding:8px 14px;background:rgba(16,17,23,.6);border-radius:10px;margin-top:6px">
          <span style="font-size:.72rem;color:#6B7280;font-weight:600;margin-right:2px">Demand:</span>
          <span style="background:#10B98118;border:1px solid #10B98155;border-radius:6px;
            padding:3px 10px;font-size:.74rem;color:#10B981;font-weight:600">🟢 Low</span>
          <span style="background:#FACC1518;border:1px solid #FACC1555;border-radius:6px;
            padding:3px 10px;font-size:.74rem;color:#FACC15;font-weight:600">🟡 Medium</span>
          <span style="background:#F9731618;border:1px solid #F9731655;border-radius:6px;
            padding:3px 10px;font-size:.74rem;color:#F97316;font-weight:600">🟠 High</span>
          <span style="background:#EF444418;border:1px solid #EF444455;border-radius:6px;
            padding:3px 10px;font-size:.74rem;color:#EF4444;font-weight:600">🔴 Very High</span>
          <span style="background:rgba(247,201,72,.12);border:1px solid rgba(247,201,72,.35);
            border-radius:6px;padding:3px 10px;font-size:.74rem;color:#F7C948;font-weight:600">
            ⭐ Best Zone</span>
          <span style="font-size:.70rem;color:#4B5563;margin-left:auto">XGBoost · 2023–2026</span>
        </div>
        """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2 — My Shift  (prediction + forecast for a chosen zone/time)
# ═════════════════════════════════════════════════════════════════════════════
def page_shift():
    st.markdown('<div class="page-title">My Shift Planner</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Predict demand · Estimate earnings · Plan your day · Compare zones</div>',
                unsafe_allow_html=True)

    with st.spinner("Loading AI model …"):
        payload    = load_regression_model()
    model_obj  = payload["model"]
    feat_cols  = payload["feature_cols"]
    y_test     = payload["y_test"]
    model_name = payload["model_name"]

    labels, lut = _zone_data()

    # ── Form (left) | Results (right) ───────────────────────────────────────
    col_form, col_res = st.columns([1, 1.2], gap="large")

    with col_form:
        _section("⚙️ Inputs")
        zone_lbl = st.selectbox("Pickup Zone", labels, index=0, key="sh_zone")
        loc_id   = lut[zone_lbl]
        defs     = _zone_defaults(loc_id)

        r1, r2 = st.columns(2)
        with r1:
            hour = st.slider("Hour", 0, 23, _now_hour, key="sh_hour",
                             help="0=midnight · 8=morning rush · 18=evening rush")
        with r2:
            shift_date = st.date_input("📅 Date", value=_today.date(),
                                       format="DD/MM/YYYY", key="sh_date")
            dow_num  = shift_date.weekday()
            dow_sel  = _DOW[dow_num]
            mon_num  = shift_date.month
            mon_sel  = _MON[mon_num - 1]
            year_sel = shift_date.year

        driver_share = st.slider("Driver Share %", 50, 100, 70, key="sh_share",
                                 help="Your cut of the fare (typical range 60–80%)") / 100.0

        with st.expander("🔧 Advanced — trip statistics"):
            ac1, ac2 = st.columns(2)
            with ac1:
                avg_fare = st.number_input("Avg Fare ($)",   1.0, 500.0, float(round(defs["fare"],2)), 0.5, key="sh_fare")
                avg_dist = st.number_input("Distance (mi)",  0.1, 100.0, float(round(defs["dist"],2)), 0.1, key="sh_dist")
            with ac2:
                avg_dur  = st.number_input("Duration (min)", 1.0, 300.0, float(round(defs["dur"],1)),  1.0, key="sh_dur")
                hist_cnt = st.number_input("Historical Demand", 0, 500000, int(defs["hist"]), 100, key="sh_hist")
            pax = st.slider("Passengers", 1, 6, 1, key="sh_pax")

        # AI Driver Assistant (quick best-zone hint)
        _zp = _zone_preds(hour, dow_num, mon_num)
        if not _zp.empty:
            best = _zp.nlargest(1, "predicted_demand").iloc[0]
            bname = str(best.get("Zone", f"Zone {best['PULocationID']}"))
            bdem  = float(best["predicted_demand"])
            brev  = float(best.get("Revenue est ($/hr)", 0))
            same  = best["PULocationID"] == loc_id
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,rgba(59,130,246,.08),rgba(247,201,72,.06));
                 border:1px solid rgba(247,201,72,.22);border-radius:14px;padding:14px 18px;margin-top:12px">
              <div style="font-size:.82rem;font-weight:700;color:#F7C948;margin-bottom:8px">
                🤖 AI Driver Assistant
              </div>
              <div style="font-size:1.05rem;font-weight:800;color:#FAFAFA;margin-bottom:4px">
                {"✅ You're already in the hottest zone!" if same else f"→ Consider: {bname}"}
              </div>
              <div style="color:#9CA3AF;font-size:.76rem">
                📈 {bdem:.0f} trips/hr &nbsp;·&nbsp; 💵 ${brev:.2f}/hr est. (70% share)
              </div>
            </div>
            """, unsafe_allow_html=True)

    with col_res:
        # Compute prediction
        features = {
            "pickup_location_id":    float(loc_id),
            "pickup_hour":           float(hour),
            "pickup_day_of_week":    float(dow_num),
            "pickup_month":          float(mon_num),
            "historical_trip_count": float(hist_cnt),
            "avg_fare_amount":       float(avg_fare),
            "avg_trip_distance":     float(avg_dist),
            "avg_trip_duration":     float(avg_dur),
            "year":                  float(year_sel),
        }
        pred = predict_regression(payload, features)

        # Confidence interval (Random Forest only)
        ci_lo = ci_hi = None
        if hasattr(model_obj, "estimators_"):
            X_raw  = np.array([[features[f] for f in feat_cols]], dtype=float)
            tpreds = np.maximum([e.predict(X_raw)[0] for e in model_obj.estimators_], 0)
            ci_lo  = float(np.percentile(tpreds, 10))
            ci_hi  = float(np.percentile(tpreds, 90))

        level, lcls = _demand_level(pred, pd.Series(y_test))

        # Historical comparison
        hist_rows = demand[(demand["PULocationID"] == loc_id) & (demand["hour"] == hour)]
        hist_avg  = float(hist_rows["trip_count"].mean()) if len(hist_rows) > 0 else 0.0
        diff_pct  = ((pred - hist_avg) / max(hist_avg, 1)) * 100
        arr  = "▲" if diff_pct >= 0 else "▼"
        clrd = "#10B981" if diff_pct >= 0 else "#EF4444"

        if ci_lo is not None:
            conf = int(max(0, 100 - (ci_hi - ci_lo) / max(pred, 1) * 50))
            ci_html = (f'<div style="font-size:.74rem;color:#9CA3AF;margin-top:5px">'
                       f'Range: <b style="color:#FAFAFA">{ci_lo:.0f}–{ci_hi:.0f}</b> trips'
                       f' &nbsp;·&nbsp; Confidence: {conf}%</div>')
        else:
            ci_html = ('<div class="banner" style="margin-top:6px;font-size:.74rem">'
                       'ℹ️ Linear Regression active — no confidence interval.</div>')

        extrap_html = ""
        if year_sel > 2026:
            extrap_html = ('<div class="warn-banner" style="margin-bottom:8px;font-size:.76rem">'
                           f'⚠️ {year_sel} is outside training range (2023–2026) — extrapolation.</div>')

        st.markdown(f"""
        {extrap_html}
        <div class="pred-card">
          <div class="pred-number">{pred:.0f}</div>
          <div class="pred-unit">predicted trips / hour</div>
          <div style="margin-top:10px">{_badge(level, lcls)}</div>
          <div style="margin-top:8px;font-size:.76rem;color:#9CA3AF">
            Historical avg {hour:02d}:00 → <b style="color:#FAFAFA">{hist_avg:.0f} trips</b>
            &nbsp;<span style="color:{clrd}">{arr} {abs(diff_pct):.1f}%</span>
          </div>
          {ci_html}
        </div>
        """, unsafe_allow_html=True)

        # Revenue card
        rev_hr  = pred * float(avg_fare) * driver_share
        rev_day = rev_hr * 8
        st.markdown(f"""
        <div class="rev-card">
          <div style="font-size:.72rem;color:#9CA3AF;font-weight:600;margin-bottom:8px">
            💵 REVENUE ESTIMATE &nbsp;({int(driver_share*100)}% driver share)
          </div>
          <div style="display:flex;gap:22px;flex-wrap:wrap">
            <div>
              <div style="font-size:1.6rem;font-weight:800;color:#10B981">${rev_hr:.2f}</div>
              <div style="font-size:.68rem;color:#6B7280">per hour</div>
            </div>
            <div>
              <div style="font-size:1.6rem;font-weight:800;color:#10B981">${rev_day:.2f}</div>
              <div style="font-size:.68rem;color:#6B7280">8-hour shift</div>
            </div>
            <div>
              <div style="font-size:1.6rem;font-weight:800;color:#F7C948">${float(avg_fare):.2f}</div>
              <div style="font-size:.68rem;color:#6B7280">avg fare/trip</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Smart alert
        if level == "Very High":
            st.markdown('<div class="alert-extreme">🔴 Extreme Demand — Surge pricing likely. Head here for maximum earnings.</div>',
                        unsafe_allow_html=True)
        elif level == "High":
            st.markdown('<div class="alert-high">🟠 High Demand — Strong pickup opportunities. Good time to work this zone.</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<div class="alert-ok">🟢 Normal Conditions — Stable demand expected for this zone and time.</div>',
                        unsafe_allow_html=True)

        # Explainability
        _section("🧠 Why This Prediction?")
        zone_info = zones[zones["LocationID"] == loc_id]
        zname = zone_info["Zone"].iloc[0] if not zone_info.empty else f"Zone {loc_id}"

        def _ins(icon, title, detail):
            return (f'<div class="insight"><span>{icon}</span> <b>{title}</b>'
                    f'<div class="insight-detail">{detail}</div></div>')

        ins = ""
        if   hour in range(7, 10):  ins += _ins("🌅","Morning Rush",  f"{hour}:00 — peak commuter demand (7–9)")
        elif hour in range(17, 20): ins += _ins("🌆","Evening Rush",  f"{hour}:00 — peak evening commute (17–19)")
        elif hour >= 22 or hour < 3:ins += _ins("🌙","Night Hours",   f"{hour}:00 — lower demand, longer trips")
        else:                       ins += _ins("☀️","Standard Hours",f"{hour}:00 — average activity")

        ins += _ins("📅", "Weekday" if dow_num < 5 else "Weekend",
            f"{dow_sel} — {'commuter patterns dominate' if dow_num<5 else 'leisure & nightlife increase'}")

        zone_pct = float(np.mean(demand["zone_total_trips"] <= defs["hist"]) * 100)
        if   zone_pct >= 80: ins += _ins("📍","High-Demand Zone", f"{zname} — Top {100-int(zone_pct)}% by historical volume")
        elif zone_pct <= 20: ins += _ins("📍","Low-Demand Zone",  f"{zname} — Below average historical activity")

        seasons = {
            (12,1,2): ("❄️","Winter",  "Cold months — weather may suppress demand"),
            (3,4,5):  ("🌸","Spring",  "Balanced demand — pleasant conditions"),
            (6,7,8):  ("☀️","Summer",  "Tourism & leisure boost trip volume"),
            (9,10,11):("🍂","Autumn",  "Steady demand — average seasonal pattern"),
        }
        for mgrp, (ic, nm, det) in seasons.items():
            if mon_num in mgrp:
                ins += _ins(ic, f"{nm} — {_MONS[mon_num-1]}", det); break

        if pax >= 3:
            ins += _ins("👥","Group Trip", f"{pax} passengers — group rides often mean shorter distances")

        st.markdown(ins, unsafe_allow_html=True)

    # ── Forecast tabs ────────────────────────────────────────────────────────
    st.markdown("---")
    _section("🔮 Shift Forecast")
    st.markdown(
        '<div class="banner">Forecast for the selected zone · day · month · year above.</div>',
        unsafe_allow_html=True)

    tab_24h, tab_dow, tab_mon = st.tabs(["⏰  24-Hour", "📅  Day-of-Week", "🗓️  Monthly"])

    fd = _zone_defaults(loc_id)

    def _fp(h, d, m):
        return predict_regression(payload, {
            "pickup_location_id":    float(loc_id),
            "pickup_hour":           float(h),
            "pickup_day_of_week":    float(d),
            "pickup_month":          float(m),
            "historical_trip_count": fd["hist"],
            "avg_fare_amount":       fd["fare"],
            "avg_trip_distance":     fd["dist"],
            "avg_trip_duration":     fd["dur"],
            "year":                  float(year_sel),
        })

    import plotly.graph_objects as go

    with tab_24h:
        hrs     = list(range(24))
        preds_h = [_fp(h, dow_num, mon_num) for h in hrs]
        pk_h    = hrs[int(np.argmax(preds_h))]
        fig_h   = go.Figure()
        fig_h.add_trace(go.Bar(
            x=hrs, y=preds_h,
            marker_color=["#EF4444" if h==pk_h else "#F7C948" for h in hrs],
            text=[f"{p:.0f}" if h==pk_h else "" for h,p in zip(hrs,preds_h)],
            textposition="outside",
        ))
        fig_h.add_trace(go.Scatter(x=hrs, y=preds_h, mode="lines",
            line=dict(color="rgba(247,201,72,.35)", width=2, dash="dot"), showlegend=False))
        fig_h.update_layout(
            title=f"24h — {zone_lbl.split(' — ')[0]} · {dow_sel} · {mon_sel} {year_sel}",
            xaxis_title="Hour", yaxis_title="Trips/hr",
            xaxis=dict(tickmode="linear",tick0=0,dtick=2), **_DRK)
        _pchart(fig_h)
        _kpi_row([
            ("⚡", f"{max(preds_h):.0f}",      f"Peak ({pk_h}:00)",  "trips/hr"),
            ("📉", f"{min(preds_h):.0f}",      "Lowest",             "trips/hr"),
            ("📊", f"{np.mean(preds_h):.0f}",  "Daily Average",      "trips/hr"),
            ("💵", f"${max(preds_h)*fd['fare']*driver_share:.2f}", "Peak Revenue est", "per hour"),
        ])

    with tab_dow:
        preds_d = [_fp(hour, d, mon_num) for d in range(7)]
        pk_d    = int(np.argmax(preds_d))
        fig_d   = go.Figure(go.Bar(
            x=_DOW, y=preds_d,
            marker_color=["#EF4444" if i==pk_d else "#3B82F6" for i in range(7)],
            text=[f"{p:.0f}" for p in preds_d], textposition="outside",
        ))
        fig_d.update_layout(
            title=f"Day-of-Week — {zone_lbl.split(' — ')[0]} · {hour:02d}:00 · {mon_sel} {year_sel}",
            xaxis_title="Day", yaxis_title="Trips/hr", **_DRK)
        _pchart(fig_d)
        _kpi_row([
            ("🏆", _DOW[pk_d],             "Busiest Day",  f"at {hour:02d}:00"),
            ("⚡", f"{max(preds_d):.0f}",  "Peak Demand",  "trips/hr"),
            ("📊", f"{np.mean(preds_d):.0f}","Weekly Avg", "trips/hr"),
            ("📉", f"{min(preds_d):.0f}",  "Quietest",     "trips/hr"),
        ])

    with tab_mon:
        preds_m = [_fp(hour, dow_num, mn) for mn in range(1, 13)]
        pk_m    = int(np.argmax(preds_m))
        fig_m   = go.Figure()
        fig_m.add_trace(go.Scatter(
            x=_MONS, y=preds_m, mode="lines+markers",
            line=dict(color="#3B82F6", width=2.5),
            marker=dict(color=["#EF4444" if i==pk_m else "#3B82F6" for i in range(12)],
                        size=[13 if i==pk_m else 7 for i in range(12)]),
            fill="tozeroy", fillcolor="rgba(59,130,246,.07)",
        ))
        fig_m.update_layout(
            title=f"Monthly — {zone_lbl.split(' — ')[0]} · {dow_sel} · {hour:02d}:00 · {year_sel}",
            xaxis_title="Month", yaxis_title="Trips/hr", **_DRK)
        _pchart(fig_m)
        _kpi_row([
            ("🏆", _MONS[pk_m],            "Busiest Month", f"at {hour:02d}:00"),
            ("⚡", f"{max(preds_m):.0f}",  "Peak Demand",   "trips/hr"),
            ("📊", f"{np.mean(preds_m):.0f}","Annual Avg",  "trips/hr"),
            ("📉", f"{min(preds_m):.0f}",  "Quietest",      "trips/hr"),
        ])

    st.markdown("---")

    # ── Relocation Simulator ─────────────────────────────────────────────────
    with st.expander("🚗  Relocation Simulator — Should I move zones?"):
        tgt_lbl = st.selectbox("Target Zone", labels, index=min(1, len(labels)-1), key="rs_tgt")
        tgt_id  = lut[tgt_lbl]

        td = _zone_defaults(tgt_id)
        tgt_pred   = predict_regression(payload, {
            "pickup_location_id":    float(tgt_id),
            "pickup_hour":           float(hour),
            "pickup_day_of_week":    float(dow_num),
            "pickup_month":          float(mon_num),
            "historical_trip_count": td["hist"],
            "avg_fare_amount":       td["fare"],
            "avg_trip_distance":     td["dist"],
            "avg_trip_duration":     td["dur"],
            "year":                  float(year_sel),
        })
        d_abs      = tgt_pred - pred
        d_pct      = (d_abs / max(pred, 1)) * 100
        cur_rev_rs = pred * float(avg_fare) * driver_share
        tgt_rev_rs = tgt_pred * td["fare"] * driver_share

        if   d_pct > 20:  rc, rt = "#10B981", "✅ Strongly Recommended"
        elif d_pct > 5:   rc, rt = "#F7C948", "⚡ Recommended"
        elif d_pct > -5:  rc, rt = "#3B82F6", "ℹ️ Neutral"
        else:             rc, rt = "#EF4444", "⚠️ Not Recommended"

        tgt_z   = zones[zones["LocationID"] == tgt_id]
        tgt_zn  = tgt_z["Zone"].iloc[0] if not tgt_z.empty else f"Zone {tgt_id}"

        st.markdown(f"""
        <div class="reloc-card" style="border:1.5px solid {rc}">
          <b style="color:#FAFAFA">Moving → {tgt_zn}</b>
          <div style="display:flex;gap:24px;margin-top:10px;flex-wrap:wrap">
            <div>
              <div style="color:#9CA3AF;font-size:.72rem">Demand Δ</div>
              <div style="font-size:1.35rem;font-weight:800;color:#F7C948">
                {"▲" if d_abs>=0 else "▼"} {abs(d_abs):.0f} trips/hr ({d_pct:+.1f}%)
              </div>
            </div>
            <div>
              <div style="color:#9CA3AF;font-size:.72rem">Revenue Δ/hr</div>
              <div style="font-size:1.35rem;font-weight:800;
                   color:{"#10B981" if tgt_rev_rs>=cur_rev_rs else "#EF4444"}">
                {"▲" if tgt_rev_rs>=cur_rev_rs else "▼"} ${abs(tgt_rev_rs-cur_rev_rs):.2f}
              </div>
            </div>
            <div>
              <div style="color:#9CA3AF;font-size:.72rem">Recommendation</div>
              <div style="font-size:1rem;font-weight:700;color:{rc}">{rt}</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)
        _kpi_row([
            ("📍", f"{pred:.0f}",      f"Current · {zname[:16]}",   "trips/hr"),
            ("🎯", f"{tgt_pred:.0f}",  f"Target · {tgt_zn[:16]}",   "trips/hr"),
            ("💰", f"${cur_rev_rs:.2f}", "Current revenue/hr",       f"{int(driver_share*100)}% share"),
            ("💵", f"${tgt_rev_rs:.2f}", "Target revenue/hr",        f"{int(driver_share*100)}% share"),
        ])

    # ── What If? ──────────────────────────────────────────────────────────────
    with st.expander("🎯  What If? Scenario Simulator"):
        wc1, wc2 = st.columns(2)
        with wc1:
            wi_hour = st.slider("What if Hour?", 0, 23, hour, key="wi_hour")
        with wc2:
            wi_date    = st.date_input("📅 What if Date?", value=shift_date,
                                       format="DD/MM/YYYY", key="wi_date")
            wi_dow     = wi_date.weekday()
            wi_dow_lbl = _DOW[wi_dow]
            wi_year    = wi_date.year

        wi_pred = predict_regression(payload, {**features,
                    "pickup_hour":float(wi_hour), "pickup_day_of_week":float(wi_dow),
                    "year":float(wi_year)})
        wi_d = wi_pred - pred
        wi_p = (wi_d / max(pred, 1)) * 100

        wca, wcb = st.columns(2)
        with wca:
            st.markdown(f"""
            <div style="background:#1A1D27;border:1px solid rgba(255,255,255,.07);
                 border-radius:12px;padding:16px;text-align:center">
              <div style="color:#9CA3AF;font-size:.72rem;margin-bottom:6px">CURRENT SCENARIO</div>
              <div style="font-size:2.2rem;font-weight:800;color:#F7C948">{pred:.0f}</div>
              <div style="color:#9CA3AF;font-size:.75rem">trips/hr · {dow_sel[:3]} {hour:02d}:00 · {year_sel}</div>
            </div>""", unsafe_allow_html=True)
        with wcb:
            cw = "#10B981" if wi_d >= 0 else "#EF4444"
            st.markdown(f"""
            <div style="background:#1A1D27;border:1px solid {cw}40;
                 border-radius:12px;padding:16px;text-align:center">
              <div style="color:#9CA3AF;font-size:.72rem;margin-bottom:6px">WHAT IF SCENARIO</div>
              <div style="font-size:2.2rem;font-weight:800;color:{cw}">{wi_pred:.0f}</div>
              <div style="color:#9CA3AF;font-size:.75rem">trips/hr · {wi_dow_lbl[:3]} {wi_hour:02d}:00 · {wi_year}</div>
              <div style="font-size:.82rem;color:{cw};margin-top:4px">
                {"▲" if wi_d>=0 else "▼"} {abs(wi_d):.0f} trips ({wi_p:+.1f}%)
              </div>
            </div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Analytics  (historical dashboard with always-visible filters)
# ═════════════════════════════════════════════════════════════════════════════
def page_analytics():
    st.markdown('<div class="page-title">Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Historical patterns · Demand trends · Year-over-year comparison</div>',
                unsafe_allow_html=True)

    # ── Filters (always visible, not collapsed) ──────────────────────────────
    _section("🎛️ Filters")
    fa, fb, fc, fd = st.columns(4)
    with fa:
        sel_years = st.multiselect("Years", [2023,2024,2025,2026],
                                   default=[2023,2024,2025,2026], key="an_years")
    with fb:
        boros    = ["All"] + sorted(df_all["pickup_borough"].dropna().unique().tolist()) \
                   if "pickup_borough" in df_all.columns else ["All"]
        sel_boro = st.selectbox("Borough", boros, key="an_boro")
    with fc:
        avail_m   = sorted(int(m) for m in df_all["month"].dropna().unique() if 1<=int(m)<=12)
        avail_mlb = [_MONS[m-1] for m in avail_m]
        sel_mlb   = st.multiselect("Months", avail_mlb, default=avail_mlb, key="an_mon")
        sel_months = [avail_m[avail_mlb.index(l)] for l in sel_mlb] if sel_mlb else avail_m
    with fd:
        hr_range = st.slider("Hour Range", 0, 23, (0, 23), key="an_hr")

    active_years = sorted(sel_years) if sel_years else [2023,2024,2025,2026]
    fdf = df_all[df_all["year"].isin(active_years)].copy()
    if sel_boro != "All" and "pickup_borough" in fdf.columns:
        fdf = fdf[fdf["pickup_borough"] == sel_boro]
    fdf = fdf[fdf["month"].isin(sel_months)]
    fdf = fdf[(fdf["hour"] >= hr_range[0]) & (fdf["hour"] <= hr_range[1])].reset_index(drop=True)

    if fdf.empty:
        st.warning("No trips match the current filters.")
        return

    fkpis   = compute_kpis(fdf)
    yrs_str = " · ".join(str(y) for y in active_years)

    # KPIs
    _section(f"📊 Overview — {yrs_str} · {fkpis['total_trips']:,} trips")
    _kpi_row([
        ("🚖", f"{fkpis['total_trips']:,}",       "Total Trips",   ""),
        ("💰", f"${fkpis['avg_fare']:.2f}",        "Avg Fare",      "per trip"),
        ("📍", f"{fkpis['avg_distance']:.1f} mi",  "Avg Distance",  "per trip"),
        ("⏱️", f"{fkpis['avg_duration']:.1f} min", "Avg Duration",  "per trip"),
        ("⚡", f"{fkpis['peak_hour']}:00",          "Peak Hour",     "most demand"),
        ("🗺️", f"{fkpis['active_zones']}",         "Active Zones",  "pickup areas"),
    ], top_idx=0)
    _kpi_row([
        ("💳", f"{fkpis['credit_pct']:.1f}%",    "Credit Card",   "of payments"),
        ("🏆", fkpis["top_zone"][:26],            "Busiest Zone",  ""),
        ("💵", f"${fkpis['total_revenue']:,.0f}", "Total Revenue", "gross fares"),
    ], top_idx=2)

    # Auto-Insights
    _section("🧠 AI Auto-Insights")
    insights   = _auto_insights(fdf, fkpis)
    cols_ins   = st.columns(len(insights))
    for col, (icon, title, detail, color) in zip(cols_ins, insights):
        with col:
            st.markdown(
                f'<div style="background:rgba(26,29,39,.8);border-top:3px solid {color};'
                f'border-radius:12px;padding:14px 16px;height:100%">'
                f'<div style="font-size:1.3rem;margin-bottom:6px">{icon}</div>'
                f'<div style="font-weight:700;color:#FAFAFA;font-size:.86rem;margin-bottom:4px">{title}</div>'
                f'<div style="color:#9CA3AF;font-size:.74rem;line-height:1.5">{detail}</div>'
                f'</div>', unsafe_allow_html=True)

    st.markdown("---")

    # Charts
    _section("Where & When Are Trips Happening?")
    c1, c2 = st.columns(2)
    with c1: _pchart(charts.trips_by_hour(fdf), h=300)
    with c2: _pchart(charts.trips_by_dow(fdf),  h=300)

    _section("Trends & Top Zones")
    c3, c4 = st.columns(2)
    with c3: _pchart(charts.monthly_trend(fdf),       h=300)
    with c4: _pchart(charts.top_zones(fdf, top_n=10), h=300)

    _section("Demand Heatmap (Hour × Day)")
    _pchart(charts.demand_heatmap(fdf), h=280)

    if "pickup_borough" in fdf.columns:
        _section("Borough Distribution")
        _pchart(charts.borough_flow(fdf), h=300)

    _section("Year-over-Year Summary")
    yoy = df_all.groupby("year").agg(
        Trips   =("fare_amount",       "count"),
        Fare    =("fare_amount",       "mean"),
        Distance=("trip_distance",     "mean"),
        Duration=("trip_duration_min", "mean"),
    ).reset_index().rename(columns={"year":"Year"})
    yoy["Year"]     = yoy["Year"].astype(str)
    yoy["Trips"]    = yoy["Trips"].apply(lambda x: f"{x:,}")
    yoy["Fare"]     = yoy["Fare"].apply(lambda x: f"${x:.2f}")
    yoy["Distance"] = yoy["Distance"].apply(lambda x: f"{x:.1f} mi")
    yoy["Duration"] = yoy["Duration"].apply(lambda x: f"{x:.1f} min")
    st.dataframe(yoy, use_container_width=True, hide_index=True)

    c5, c6 = st.columns(2)
    with c5: _pchart(charts.yearly_trip_comparison(df_all), h=280)
    with c6: _pchart(charts.yearly_fare_trend(df_all),      h=280)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Model  (technical AI insights: performance · clustering · regression)
# ═════════════════════════════════════════════════════════════════════════════
def page_model():
    import plotly.graph_objects as go

    st.markdown('<div class="page-title">Model Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Model performance · Feature importance · Clustering · Regression comparison</div>',
                unsafe_allow_html=True)

    tab_perf, tab_clust, tab_regr = st.tabs(
        ["⚙️  Model Performance", "🔵  Clustering", "📉  Regression"]
    )

    # ── Model Performance ─────────────────────────────────────────────────────
    with tab_perf:
        with st.spinner("Loading regression model …"):
            payload = load_regression_model()
        with st.spinner("Loading XGBoost model …"):
            _, xgb_met, xgb_fi, xgb_yte, xgb_ypred = load_xgb_model()

        mn = payload["model_name"]
        m  = payload["metrics"]
        am = payload.get("all_metrics", {})

        _section("Best Model Metrics")
        _kpi_row([
            ("🤖", mn,                  "Active Model",   "best by R²"),
            ("📉", f"{m['mae']:.2f}",   "MAE",            "avg trips error"),
            ("📊", f"{m['rmse']:.2f}",  "RMSE",           "root mean sq. error"),
            ("📈", f"{m['r2']:.3f}",    "R² Score",       "variance explained"),
            ("🏋️",f"{payload['n_train']:,}", "Train rows","all years"),
        ], top_idx=3)

        r2v = m["r2"]
        if   r2v > 0.85: badge = "🟢 Excellent — explains >85% of demand variance"
        elif r2v > 0.70: badge = "🟡 Good — solid predictive power"
        else:            badge = "🔴 Fair — more data would help"
        st.info(badge)

        if am:
            _section("All Models Comparison")
            all_m = {**am}
            if xgb_met:
                all_m["XGBoost"] = xgb_met
            mcols = st.columns(len(all_m))
            for col, (mname, mmet) in zip(mcols, all_m.items()):
                is_best = mname == mn
                border  = "rgba(247,201,72,.35)" if is_best else "rgba(255,255,255,.06)"
                with col:
                    st.markdown(f"""
                    <div style="background:#1A1D27;border:1.5px solid {border};
                         border-radius:14px;padding:16px;text-align:center">
                      <div style="font-size:.82rem;font-weight:700;color:#FAFAFA;margin-bottom:10px">
                        {mname} {"✅" if is_best else ""}
                      </div>
                      <div style="font-size:1.2rem;font-weight:800;color:#F7C948">{mmet["r2"]:.3f}</div>
                      <div style="font-size:.68rem;color:#9CA3AF">R²</div>
                      <div style="margin-top:8px;font-size:.84rem;color:#9CA3AF">
                        MAE {mmet["mae"]:.2f} · RMSE {mmet["rmse"]:.2f}
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

            pc1, pc2 = st.columns(2)
            with pc1: _pchart(reg.chart_metrics_bar(am), h=300)
            with pc2: _pchart(reg.chart_r2_bar(am),      h=300)

        _section("Actual vs Predicted")
        dc1, dc2 = st.columns(2)
        with dc1:
            _pchart(reg.chart_actual_vs_pred(payload["y_test"], payload["y_pred"], mn), h=320)
        with dc2:
            if payload.get("feature_importance") is not None:
                _pchart(reg.chart_feature_importance(payload["feature_importance"], mn), h=320)
            else:
                st.info("Feature importance not available for Linear Regression.")

    # ── Clustering ────────────────────────────────────────────────────────────
    with tab_clust:
        feat_map = clust.available_features(demand)
        if not feat_map:
            st.error("No numeric features for clustering.")
        else:
            cc1, cc2, cc3 = st.columns([2.5, 1, 1])
            with cc1:
                sel_f = st.multiselect("Features (select 2–5)", list(feat_map.keys()),
                                       default=["trip_count","avg_fare","avg_distance"],
                                       format_func=lambda x: feat_map[x], key="cl_feats")
            with cc2:
                k = st.slider("Clusters K", 2, 8, 3, key="cl_k")
            with cc3:
                normalize = st.checkbox("Normalize", True, key="cl_norm")

            if len(sel_f) < 2:
                st.warning("Select at least 2 features.")
            else:
                with st.spinner("Running KMeans …"):
                    labels_k, X_proc, inertia = clust.run_kmeans(demand, sel_f, k, normalize)

                use_pca = len(sel_f) > 2
                if use_pca:
                    X_2d, vr = clust.apply_pca(X_proc)
                    xl = f"PC1 ({vr[0]*100:.1f}% var)"
                    yl = f"PC2 ({vr[1]*100:.1f}% var)"
                    title = f"KMeans K={k} — PCA 2D"
                else:
                    X_2d = X_proc
                    xl, yl = feat_map[sel_f[0]], feat_map[sel_f[1]]
                    title = f"KMeans K={k}"

                kc1, kc2 = st.columns([2, 1])
                with kc1:
                    _pchart(clust.chart_scatter(X_2d, labels_k, xl, yl, title), h=380)
                with kc2:
                    _section("Elbow Method")
                    with st.spinner("Computing elbow …"):
                        ks_n, in_n = clust.compute_elbow(demand, sel_f, normalize=True)
                        ks_r, in_r = clust.compute_elbow(demand, sel_f, normalize=False)
                    _pchart(clust.chart_elbow(ks_n, in_n, in_r), h=280)

                st.markdown(
                    f'<div class="banner">K={k} · Inertia: {inertia:,.0f} · '
                    f'{"Normalized ✅" if normalize else "Raw ⚠️"}</div>',
                    unsafe_allow_html=True)

                _section("Cluster Statistics")
                ds = demand[sel_f].dropna().copy()
                ds["Cluster"] = labels_k
                st.dataframe(
                    ds.groupby("Cluster")[sel_f].mean().round(2)
                    .rename(index=lambda i: f"Cluster {i}")
                    .rename(columns=feat_map),
                    use_container_width=True,
                )

    # ── Regression comparison ─────────────────────────────────────────────────
    with tab_regr:
        _section("Feature Selection")
        rc1, rc2 = st.columns([3, 1])
        with rc1:
            sel_reg = st.multiselect(
                "Features", list(reg.REGRESSION_FEATURES.keys()),
                default=list(reg.REGRESSION_FEATURES.keys()),
                format_func=lambda x: reg.REGRESSION_FEATURES[x], key="rg_feats")
        with rc2:
            st.markdown('<div style="margin-top:2rem;color:#9CA3AF;font-size:.8rem">'
                        '<b>Target:</b> trip_count</div>', unsafe_allow_html=True)

        if not sel_reg:
            st.warning("Select at least one feature.")
        else:
            with st.spinner("Training LR + RF — this may take a moment …"):
                out     = reg.get_regression_results(tuple(sorted(sel_reg)))
            results = out["results"]
            y_te    = out["y_te"]
            best_n  = max(results, key=lambda x: results[x]["r2"])
            best_r  = results[best_n]

            _kpi_row([
                ("🏆", best_n,                    "Best Model",  "by R²"),
                ("📉", f"{best_r['mae']:.2f}",    "Best MAE",    "trips"),
                ("📈", f"{best_r['r2']:.3f}",     "Best R²",     ""),
                ("🏋️",f"{out['n_train']:,}",      "Train rows",  ""),
            ], top_idx=2)

            rc3, rc4 = st.columns(2)
            with rc3: _pchart(reg.chart_metrics_bar(results), h=300)
            with rc4: _pchart(reg.chart_r2_bar(results),      h=300)

            _section("Actual vs Predicted")
            tabs_m = st.tabs([f"📊 {mn}" for mn in results])
            for tab, (mname, res) in zip(tabs_m, results.items()):
                with tab:
                    _pchart(reg.chart_actual_vs_pred(y_te, res["y_pred"], mname), h=320)

            fi_list = [(mn, r) for mn, r in results.items() if "feature_importance" in r]
            if fi_list:
                _section("Feature Importance")
                fi_cols = st.columns(len(fi_list))
                for col, (mname, res) in zip(fi_cols, fi_list):
                    with col:
                        _pchart(reg.chart_feature_importance(res["feature_importance"], mname), h=300)


# ─────────────────────────────────────────────────────────────────────────────
# Router
# ─────────────────────────────────────────────────────────────────────────────
_ROUTES = {
    "live":      page_live,
    "shift":     page_shift,
    "analytics": page_analytics,
    "model":     page_model,
}
_ROUTES[page_key]()
