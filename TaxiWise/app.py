"""
TaxiWise — AI Transportation Intelligence Platform
Driver-first redesign: Live · My Shift · Analytics · Model
Bilingual: English / Hebrew (RTL supported)
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

# ── Language state — must be initialised before any t() call ─────────────────
if "lang" not in st.session_state:
    st.session_state["lang"] = "en"

from src.i18n        import TRANSLATIONS
from src.data_loader import load_trips, load_zones, compute_demand, compute_kpis
from src.model       import load_xgb_model, load_regression_model, predict_regression
import src.charts     as charts
import src.clustering as clust
import src.regression as reg


# ── Translation helpers ───────────────────────────────────────────────────────
def t(key: str, **kwargs) -> str:
    """Return the UI string for *key* in the active language."""
    lang = st.session_state.get("lang", "en")
    text = (TRANSLATIONS.get(lang, {}).get(key)
            or TRANSLATIONS["en"].get(key, key))
    if kwargs:
        try:
            text = text.format(**kwargs)
        except Exception:
            try:
                text = TRANSLATIONS["en"].get(key, key).format(**kwargs)
            except Exception:
                pass
    return text


def tl(key: str) -> list:
    """Return a translated list (days, months, …)."""
    lang = st.session_state.get("lang", "en")
    return (TRANSLATIONS.get(lang, {}).get(key)
            or TRANSLATIONS["en"].get(key, []))

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
/* ── What Changed Today panel ── */
.wc-panel{background:linear-gradient(160deg,#0D0F1A,#131726);
  border:1px solid rgba(255,255,255,.07);border-radius:18px;padding:22px 26px;margin-top:.8rem}
.wc-kpis{display:grid;grid-template-columns:repeat(5,1fr);gap:8px;margin:14px 0 20px}
.wc-kpi{background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.06);
  border-radius:11px;padding:13px 14px;text-align:center}
.wc-kpi-val{font-size:1.4rem;font-weight:900;line-height:1}
.wc-kpi-lbl{font-size:.6rem;color:#6B7280;text-transform:uppercase;letter-spacing:.05em;margin-top:5px}
.wc-cols{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:18px}
.wc-half{background:rgba(255,255,255,.02);border:1px solid rgba(255,255,255,.05);
  border-radius:12px;padding:14px 16px}
.wc-half-title{font-size:.78rem;font-weight:700;margin-bottom:10px;letter-spacing:.02em}
.wc-row{display:flex;justify-content:space-between;align-items:center;
  padding:5px 0;border-bottom:1px solid rgba(255,255,255,.04)}
.wc-row:last-child{border-bottom:none}
.wc-name{font-size:.82rem;font-weight:600;color:#FAFAFA}
.wc-boro{font-size:.64rem;color:#6B7280}
.wc-up{color:#10B981;font-size:.86rem;font-weight:800}
.wc-dn{color:#EF4444;font-size:.86rem;font-weight:800}
.wc-pk-row{display:flex;justify-content:space-between;align-items:center;
  padding:6px 0;border-bottom:1px solid rgba(255,255,255,.04);font-size:.8rem}
.wc-pk-row:last-child{border-bottom:none}

/* ── Natural language insight cards ── */
.nl-insight{background:#161924;border-left:3px solid #6B7280;border-radius:10px;
  padding:14px 18px;font-size:.85rem;color:#D1D5DB;line-height:1.55}
.nl-insight.surge{border-left-color:#F7C948;background:rgba(247,201,72,.035)}
.nl-insight.warn {border-left-color:#F97316;background:rgba(249,115,22,.035)}
.nl-insight.info {border-left-color:#3B82F6;background:rgba(59,130,246,.035)}
.nl-ni-icon{font-size:1.15rem;margin-bottom:6px;display:block}

/* ── AI Recommendation Card ── */
.ai-rec-card{background:linear-gradient(135deg,#0F111A,#16192A,#1E2438);
  border:2px solid rgba(247,201,72,.50);border-radius:22px;padding:24px 30px;
  margin-bottom:1.2rem;position:relative;overflow:hidden}
.ai-rec-card::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;
  background:linear-gradient(90deg,#F7C948,#F97316,#EF4444)}
.ai-rec-tag{font-size:.7rem;font-weight:700;color:#9CA3AF;text-transform:uppercase;
  letter-spacing:.09em;margin-bottom:8px;display:flex;align-items:center;
  justify-content:space-between;flex-wrap:wrap;gap:8px}
.ai-rec-zone{font-size:2.2rem;font-weight:900;color:#FAFAFA;line-height:1.1;margin-bottom:3px}
.ai-rec-boro{font-size:.82rem;color:#9CA3AF;margin-bottom:18px}
.ai-rec-row{display:flex;gap:10px;flex-wrap:wrap}
.ai-rec-chip{background:rgba(255,255,255,.045);border:1px solid rgba(255,255,255,.09);
  border-radius:13px;padding:13px 18px;flex:1;min-width:118px}
.ai-rec-chip-val{font-size:1.42rem;font-weight:900;line-height:1}
.ai-rec-chip-lbl{font-size:.63rem;color:#6B7280;text-transform:uppercase;
  letter-spacing:.06em;margin-top:5px}

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
    # ── Language switcher (top of sidebar) ───────────────────────────────────
    _lang_opts  = [t("lang_en"), t("lang_he")]
    _lang_codes = ["en", "he"]
    _lang_cur   = _lang_codes.index(st.session_state.get("lang", "en"))
    _lang_sel   = st.radio(
        t("lang_label"),
        _lang_opts,
        index=_lang_cur,
        horizontal=True,
        key="lang_radio",
    )
    if _lang_codes[_lang_opts.index(_lang_sel)] != st.session_state["lang"]:
        st.session_state["lang"] = _lang_codes[_lang_opts.index(_lang_sel)]
        st.rerun()

    st.markdown('<div style="height:1px;background:rgba(255,255,255,.06);margin:8px 0 14px"></div>',
                unsafe_allow_html=True)

    _now_ts = _dt.now()
    st.markdown(f"""
    <div style="padding:4px 0 14px">
      <div style="font-size:1.5rem;font-weight:900;color:#F7C948">🚕 TaxiWise</div>
      <div style="color:#6B7280;font-size:.71rem;margin-top:2px">
        {_now_ts.strftime("%a %b %d  ·  %H:%M")}
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="height:1px;background:rgba(255,255,255,.06);margin-bottom:12px"></div>',
                unsafe_allow_html=True)

    PAGES = {
        t("nav_live"):      "live",
        t("nav_shift"):     "shift",
        t("nav_analytics"): "analytics",
        t("nav_model"):     "model",
        t("nav_intel"):     "intel",
        t("nav_future"):    "future",
    }
    page_key = PAGES[st.radio("nav", list(PAGES.keys()), label_visibility="collapsed")]

    st.markdown('<div style="height:1px;background:rgba(255,255,255,.06);margin:12px 0"></div>',
                unsafe_allow_html=True)
    st.markdown(f"""
    <div style="color:#4B5563;font-size:.68rem;line-height:1.85">
      <b style="color:#6B7280">{t("sidebar_data")}</b><br>
      {t("sidebar_data_desc")}<br>
      {t("sidebar_trips_loaded", n=len(df_all))}<br><br>
      <b style="color:#6B7280">{t("sidebar_models")}</b><br>
      {t("sidebar_models_desc")}
    </div>
    """, unsafe_allow_html=True)

# ── RTL CSS — injected when Hebrew is active ──────────────────────────────────
if st.session_state.get("lang") == "he":
    st.markdown("""
    <style>
    .block-container { direction: rtl !important; }
    .sec {
        border-left:  none       !important;
        border-right: 3px solid #F7C948 !important;
        padding-left: 0          !important;
        padding-right: 9px       !important;
    }
    .hero-card, .zone-quick, .kpi-card, .pred-card, .rev-card,
    .alert-extreme, .alert-high, .alert-ok, .insight,
    .reloc-card, .banner, .warn-banner,
    .ai-rec-card, .nl-insight { text-align: right !important; }
    .nl-insight { border-left: none !important; border-right: 3px solid #6B7280 !important; }
    .nl-insight.surge { border-right-color: #F7C948 !important; }
    .nl-insight.warn  { border-right-color: #F97316 !important; }
    .nl-insight.info  { border-right-color: #3B82F6 !important; }
    .zone-quick.r1 { border-left:none!important; border-right:4px solid #EF4444!important; }
    .zone-quick.r2 { border-left:none!important; border-right:4px solid #F97316!important; }
    .zone-quick.r3 { border-left:none!important; border-right:4px solid #F7C948!important; }
    .zone-quick.r4 { border-left:none!important; border-right:4px solid #3B82F6!important; }
    .zone-quick.r5 { border-left:none!important; border-right:4px solid #8B5CF6!important; }
    section[data-testid="stSidebar"] { direction: rtl !important; }
    </style>
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
    key = {"Very High":"badge_vh","High":"badge_h","Medium":"badge_m","Low":"badge_l"}.get(level,"badge_m")
    return f'<span class="badge {cls}">{t(key)}</span>'


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
    zs["hour"]      = hour
    zs["dow"]       = dow
    zs["month"]     = month
    zs["hour_sin"]  = np.sin(2 * np.pi * hour  / 24)
    zs["hour_cos"]  = np.cos(2 * np.pi * hour  / 24)
    zs["dow_sin"]   = np.sin(2 * np.pi * dow   / 7)
    zs["dow_cos"]   = np.cos(2 * np.pi * dow   / 7)
    zs["month_sin"] = np.sin(2 * np.pi * month / 12)
    zs["month_cos"] = np.cos(2 * np.pi * month / 12)
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

    # Historical actual demand for this exact (hour, dow, month) slot — ground truth
    _slot = demand[
        (demand["hour"] == hour) &
        (demand["dow"]  == dow)  &
        (demand["month"] == month)
    ]
    _slot_avg = _slot.groupby("PULocationID")["trip_count"].mean().reset_index()
    _slot_avg.columns = ["PULocationID", "hist_demand_slot"]
    merged = merged.merge(_slot_avg, on="PULocationID", how="left")
    merged["hist_demand_slot"] = merged["hist_demand_slot"].fillna(0.0)

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


@st.cache_data(show_spinner=False)
def _hour_curve(dow: int, month: int) -> pd.DataFrame:
    """Max predicted demand for every hour (0-23) for the given dow/month."""
    from src.model import FEATURE_COLS
    model, *_ = load_xgb_model()
    _static = ["PULocationID", "zone_total_trips", "avg_fare", "avg_distance", "avg_duration"]
    zs_base = (demand.groupby("PULocationID")
               .agg(avg_fare        =("avg_fare",         "mean"),
                    avg_distance    =("avg_distance",     "mean"),
                    avg_duration    =("avg_duration",     "mean"),
                    zone_total_trips=("zone_total_trips", "first"))
               .reset_index())
    for col in ["avg_fare", "avg_distance", "avg_duration", "zone_total_trips"]:
        zs_base[col] = zs_base[col].fillna(float(demand[col].median()))
    zs_base = zs_base.dropna(subset=_static)

    rows = []
    for hour in range(24):
        zs              = zs_base.copy()
        zs["hour"]      = hour
        zs["dow"]       = dow
        zs["month"]     = month
        zs["hour_sin"]  = np.sin(2 * np.pi * hour  / 24)
        zs["hour_cos"]  = np.cos(2 * np.pi * hour  / 24)
        zs["dow_sin"]   = np.sin(2 * np.pi * dow   / 7)
        zs["dow_cos"]   = np.cos(2 * np.pi * dow   / 7)
        zs["month_sin"] = np.sin(2 * np.pi * month / 12)
        zs["month_cos"] = np.cos(2 * np.pi * month / 12)
        preds           = np.maximum(model.predict(zs[FEATURE_COLS].values), 0)
        rows.append({"hour": hour, "max_demand": float(preds.max()),
                     "avg_demand": float(preds.mean())})
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def _dow_curve(hour: int, month: int) -> pd.DataFrame:
    """Max predicted demand for every day-of-week (0-6) for the given hour/month."""
    from src.model import FEATURE_COLS
    model, *_ = load_xgb_model()
    _static = ["PULocationID", "zone_total_trips", "avg_fare", "avg_distance", "avg_duration"]
    zs_base = (demand.groupby("PULocationID")
               .agg(avg_fare        =("avg_fare",         "mean"),
                    avg_distance    =("avg_distance",     "mean"),
                    avg_duration    =("avg_duration",     "mean"),
                    zone_total_trips=("zone_total_trips", "first"))
               .reset_index())
    for col in ["avg_fare", "avg_distance", "avg_duration", "zone_total_trips"]:
        zs_base[col] = zs_base[col].fillna(float(demand[col].median()))
    zs_base = zs_base.dropna(subset=_static)

    rows = []
    for d in range(7):
        zs              = zs_base.copy()
        zs["hour"]      = hour
        zs["dow"]       = d
        zs["month"]     = month
        zs["hour_sin"]  = np.sin(2 * np.pi * hour / 24)
        zs["hour_cos"]  = np.cos(2 * np.pi * hour / 24)
        zs["dow_sin"]   = np.sin(2 * np.pi * d    / 7)
        zs["dow_cos"]   = np.cos(2 * np.pi * d    / 7)
        zs["month_sin"] = np.sin(2 * np.pi * month / 12)
        zs["month_cos"] = np.cos(2 * np.pi * month / 12)
        preds           = np.maximum(model.predict(zs[FEATURE_COLS].values), 0)
        rows.append({"dow": d, "max_demand": float(preds.max()),
                     "avg_demand": float(preds.mean())})
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def _daily_zone_summary(dow: int, month: int) -> pd.DataFrame:
    """Avg/peak predicted demand per zone across all 24 hours for a given dow+month."""
    from src.model import FEATURE_COLS
    model, *_ = load_xgb_model()
    zs_base = (demand.groupby("PULocationID")
               .agg(avg_fare        =("avg_fare",         "mean"),
                    avg_distance    =("avg_distance",     "mean"),
                    avg_duration    =("avg_duration",     "mean"),
                    zone_total_trips=("zone_total_trips", "first"))
               .reset_index())
    for col in ["avg_fare", "avg_distance", "avg_duration", "zone_total_trips"]:
        zs_base[col] = zs_base[col].fillna(float(demand[col].median()))
    _static_cols = ["PULocationID", "zone_total_trips", "avg_fare", "avg_distance", "avg_duration"]
    zs_base = zs_base.dropna(subset=_static_cols)
    if zs_base.empty:
        return pd.DataFrame()

    hour_preds = []
    for hour in range(24):
        zs              = zs_base.copy()
        zs["hour"]      = hour
        zs["dow"]       = dow
        zs["month"]     = month
        zs["hour_sin"]  = np.sin(2 * np.pi * hour  / 24)
        zs["hour_cos"]  = np.cos(2 * np.pi * hour  / 24)
        zs["dow_sin"]   = np.sin(2 * np.pi * dow   / 7)
        zs["dow_cos"]   = np.cos(2 * np.pi * dow   / 7)
        zs["month_sin"] = np.sin(2 * np.pi * month / 12)
        zs["month_cos"] = np.cos(2 * np.pi * month / 12)
        hour_preds.append(np.maximum(model.predict(zs[FEATURE_COLS].values), 0))

    arr = np.stack(hour_preds)   # (24, n_zones)
    out = zs_base[["PULocationID"]].copy()
    out["avg_demand"]   = arr.mean(axis=0)
    out["peak_demand"]  = arr.max(axis=0)
    out["peak_hour"]    = arr.argmax(axis=0).astype(int)
    out["total_demand"] = arr.sum(axis=0)

    return out.merge(
        zones[["LocationID", "Zone", "Borough"]],
        left_on="PULocationID", right_on="LocationID", how="left",
    ).fillna({"Zone": "Unknown", "Borough": "Unknown"})


@st.cache_data(show_spinner=False)
def _build_zone_feat_table() -> pd.DataFrame:
    """Per-zone static features for the regression model (no time fields)."""
    zft = (demand.groupby("PULocationID")
           .agg(
               historical_trip_count=("zone_total_trips", "first"),
               avg_fare_amount      =("avg_fare",         "mean"),
               avg_trip_distance    =("avg_distance",     "mean"),
               avg_trip_duration    =("avg_duration",     "mean"),
           )
           .reset_index()
           .rename(columns={"PULocationID": "pickup_location_id"}))
    zft["historical_trip_count"] = zft["historical_trip_count"].fillna(
        float(demand["zone_total_trips"].median())).clip(lower=0)
    zft["avg_fare_amount"]   = zft["avg_fare_amount"].fillna(
        float(demand["avg_fare"].median())).clip(1.0, 500.0)
    zft["avg_trip_distance"] = zft["avg_trip_distance"].fillna(
        float(demand["avg_distance"].median())).clip(0.1, 100.0)
    zft["avg_trip_duration"] = zft["avg_trip_duration"].fillna(
        float(demand["avg_duration"].median())).clip(1.0, 300.0)
    return zft


@st.cache_data(show_spinner=False)
def _future_full_forecast(month: int, hour: int, dow: int) -> pd.DataFrame:
    """Regression-model predictions for every (zone × year 2023-2035) at a fixed time slot."""
    payload   = load_regression_model()
    feat_cols = payload["feature_cols"]
    zft       = _build_zone_feat_table()
    frames    = []
    for year in range(2023, 2036):
        zf                       = zft.copy()
        zf["pickup_hour"]        = float(hour)
        zf["pickup_day_of_week"] = float(dow)
        zf["pickup_month"]       = float(month)
        zf["year"]               = float(year)
        preds = np.maximum(payload["model"].predict(zf[feat_cols].values.astype(float)), 0)
        frames.append(pd.DataFrame({
            "PULocationID": zft["pickup_location_id"].values,
            "year":         year,
            "pred":         preds,
        }))
    return (pd.concat(frames, ignore_index=True)
            .merge(zones[["LocationID","Zone","Borough"]],
                   left_on="PULocationID", right_on="LocationID", how="left")
            .fillna({"Zone": "Unknown", "Borough": "Unknown"}))


@st.cache_data(show_spinner=False)
def _future_monthly_profile(year: int, hour: int, dow: int) -> pd.DataFrame:
    """Total predicted demand (all zones) per month for a given year/hour/dow."""
    payload   = load_regression_model()
    feat_cols = payload["feature_cols"]
    zft       = _build_zone_feat_table()
    rows = []
    for month in range(1, 13):
        zf                       = zft.copy()
        zf["pickup_hour"]        = float(hour)
        zf["pickup_day_of_week"] = float(dow)
        zf["pickup_month"]       = float(month)
        zf["year"]               = float(year)
        preds = np.maximum(payload["model"].predict(zf[feat_cols].values.astype(float)), 0)
        rows.append({"month": month, "total": float(preds.sum()), "avg": float(preds.mean())})
    return pd.DataFrame(rows)


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

    # Rich hover tooltip (translated)
    _h_demand  = t("hover_demand")
    _h_opp     = t("hover_opp")
    _h_fare    = t("hover_avg_fare")
    _h_rev     = t("hover_revenue")
    _h_level   = t("hover_level")
    _h_thr     = t("hover_trips_hr")
    _lv_map    = {
        "Very High": t("very_high"), "High": t("high"),
        "Medium":    t("medium"),    "Low":  t("low"),
    }
    merged["_hover"] = merged.apply(lambda r: (
        f"<b>{r['Zone']}</b><br>"
        f"<span>{r['Borough']}</span><br>"
        f"───────────────────<br>"
        f"🔮 {_h_demand}: <b>{float(r['Predicted Trips/hr']):.0f} {_h_thr}</b><br>"
        + (f"⭐ {_h_opp}: <b>{int(r['Opportunity Score'])}/100</b><br>" if has_opp else "")
        + f"💰 {_h_fare}: <b>${float(r['Avg Fare ($)']):.2f}</b><br>"
        f"💵 {_h_rev}: <b>${float(r['Revenue est ($/hr)']):.2f}/hr</b><br>"
        f"📊 {_h_level}: <b>{_lv_map.get(r['Demand Level'], r['Demand Level'])}</b>"
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
        out.append(("🌅", t("ins_morning_title"), t("ins_morning_det", h=ph), "#F97316"))
    elif 17 <= ph <= 19:
        out.append(("🌆", t("ins_evening_title"), t("ins_evening_det", h=ph), "#F97316"))
    else:
        out.append(("🌙", t("ins_offpeak_title"), t("ins_offpeak_det", h=ph), "#8B5CF6"))

    tz = str(kpis_in.get("top_zone", ""))
    if any(k in tz for k in ("Airport","JFK","LaGuardia","EWR")):
        out.append(("✈️", t("ins_airport_title"), t("ins_airport_det", zone=tz[:30]), "#3B82F6"))
    elif "Midtown" in tz:
        out.append(("🏙️", t("ins_midtown_title"), t("ins_midtown_det", zone=tz[:30]), "#F7C948"))
    else:
        out.append(("📍", t("ins_topzone_title"), t("ins_topzone_det", zone=tz[:30]), "#10B981"))

    cp = kpis_in.get("credit_pct", 0)
    if cp > 70:
        out.append(("💳", t("ins_digital_title"), t("ins_digital_det", pct=cp), "#10B981"))

    if "dow" in df_in.columns and len(df_in) > 100:
        we = df_in[df_in["dow"] >= 5]
        wd = df_in[df_in["dow"] <  5]
        if len(we) > 0 and len(wd) > 0:
            we_avg = len(we) / max(we["dow"].nunique(), 1)
            wd_avg = len(wd) / max(wd["dow"].nunique(), 1)
            if we_avg > wd_avg * 1.08:
                out.append(("🎉", t("ins_wkend_title"), t("ins_wkend_det"), "#8B5CF6"))
            else:
                out.append(("💼", t("ins_wkday_title"), t("ins_wkday_det"), "#3B82F6"))
    return out[:4]


def _generate_nl_insights(
    zp: pd.DataFrame, hcur: pd.DataFrame,
    live_hour: int, live_dow: int, live_mon: int,
) -> list[tuple]:
    """Return up to 4 (emoji, text, css_mod) NL insight tuples from live forecast data."""
    if zp.empty:
        return []

    insights: list[tuple] = []
    best_row  = zp.nlargest(1, "predicted_demand").iloc[0]
    best_zone = str(best_row.get("Zone", ""))
    best_dem  = float(best_row["predicted_demand"])

    # ── 1. Peak-hour window ──────────────────────────────────────────────────
    if not hcur.empty:
        peak_h = int(hcur.loc[hcur["max_demand"].idxmax(), "hour"])
        diff   = peak_h - live_hour
        if diff == 0:
            insights.append(("🔥", t("ins_nl_at_peak"), "surge"))
        elif 1 <= diff <= 4:
            insights.append(("🔮", t("ins_nl_peak_coming", h=peak_h, n=diff), "surge"))
        elif -2 <= diff < 0:
            insights.append(("🔥", t("ins_nl_peak_recent", h=peak_h), "warn"))
        else:
            insights.append(("📈", t("ins_nl_peak_later", h=peak_h), "info"))

    # ── 2. Best zone demand change in 2 hours ────────────────────────────────
    future_h = min(live_hour + 2, 23)
    if future_h != live_hour:
        zp_fut  = _zone_preds(future_h, live_dow, live_mon)
        if not zp_fut.empty:
            best_id  = int(best_row["PULocationID"])
            fut_row  = zp_fut[zp_fut["PULocationID"] == best_id]
            if not fut_row.empty:
                fut_dem = float(fut_row["predicted_demand"].values[0])
                pct_chg = int((fut_dem - best_dem) / max(best_dem, 0.001) * 100)
                if pct_chg >= 8:
                    insights.append(("📊", t("ins_nl_rising",  zone=best_zone, pct=pct_chg,      h=future_h), "surge"))
                elif pct_chg <= -8:
                    insights.append(("📉", t("ins_nl_falling", zone=best_zone, pct=abs(pct_chg), h=future_h), "warn"))

    # ── 3. Borough leadership ────────────────────────────────────────────────
    if "Borough" in zp.columns:
        hi = zp[zp["Demand Level"].isin(["Very High", "High"])]
        if not hi.empty:
            bcount   = hi.groupby("Borough").size()
            top_boro = str(bcount.idxmax())
            boro_pct = int(bcount.max() / max(int(bcount.sum()), 1) * 100)
            if boro_pct >= 30:
                insights.append(("🏙️", t("ins_nl_boro_lead", boro=top_boro, pct=boro_pct), "info"))

    # ── 4. Revenue vs demand split ───────────────────────────────────────────
    if "Revenue est ($/hr)" in zp.columns:
        top_d = str(zp.nlargest(1, "predicted_demand").iloc[0].get("Zone", ""))
        r_row = zp.nlargest(1, "Revenue est ($/hr)").iloc[0]
        top_r = str(r_row.get("Zone", ""))
        rev   = float(r_row.get("Revenue est ($/hr)", 0))
        if top_d != top_r:
            insights.append(("💡", t("ins_nl_rev_split", d_zone=top_d, r_zone=top_r, rev=rev), "info"))
        else:
            opp = int(best_row.get("Opportunity Score", 0))
            if opp >= 70:
                insights.append(("⭐", t("ins_nl_double_top", zone=top_d, score=opp), "surge"))

    # ── 5. Demand concentration ──────────────────────────────────────────────
    top5_dem  = float(zp.nlargest(5, "predicted_demand")["predicted_demand"].sum())
    total_dem = float(zp["predicted_demand"].sum())
    conc_pct  = int(top5_dem / max(total_dem, 0.001) * 100)
    if conc_pct >= 35:
        insights.append(("🎯", t("ins_nl_concentrated", pct=conc_pct), "warn"))

    return insights[:4]


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Live  (driver home: map + instant recommendation)
# ═════════════════════════════════════════════════════════════════════════════
def page_live():
    st.markdown(f'<div class="page-title">{t("live_title")}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="page-sub">{t("live_sub")}</div>', unsafe_allow_html=True)

    # ── Time controls ────────────────────────────────────────────────────────
    tc1, tc2, tc3 = st.columns([2, 1.8, 1.2])
    with tc1:
        live_hour = st.slider(t("live_hour"), 0, 23, _now_hour, key="lv_hour",
                              help=t("live_hour_help"))
    with tc2:
        live_date    = st.date_input(t("live_date"), value=_today.date(),
                                     format="DD/MM/YYYY", key="lv_date")
        live_dow     = live_date.weekday()
        live_dow_lbl = tl("days")[live_dow]
        live_mon     = live_date.month
    with tc3:
        _scatter_lbl = t("live_scatter")
        _heatmap_lbl = t("live_heatmap")
        map_mode = st.radio(t("live_map_view"), [_scatter_lbl, _heatmap_lbl],
                            horizontal=True, key="lv_mapmode")

    with st.spinner(t("live_spinner")):
        zp   = _zone_preds(live_hour, live_dow, live_mon)
        hcur = _hour_curve(live_dow, live_mon)

    if zp.empty:
        st.error(t("live_error"))
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

    # ── Full-width AI Recommendation Card ────────────────────────────────────
    _, _xgb_met, *_ = load_xgb_model()
    _r2  = float(_xgb_met["r2"])  if _xgb_met else 0.75
    _mae = float(_xgb_met["mae"]) if _xgb_met else 5.0

    # Factor 1 — R² (40 pts): direct model quality
    _r2_contrib = _r2 * 40

    # Factor 2 — MAE factor (20 pts): lower MAE relative to mean prediction = better
    _mean_pred   = float(zp["predicted_demand"].mean()) if not zp.empty else 10.0
    _mae_factor  = max(0.0, 1.0 - _mae / max(_mean_pred, 1.0))
    _mae_contrib = _mae_factor * 20

    # Factor 3 — Data coverage (20 pts): zones with actual slot history / total zones
    _n_with_hist  = int((zp["hist_demand_slot"] > 0).sum()) if "hist_demand_slot" in zp.columns else 0
    _data_contrib = (_n_with_hist / max(len(zp), 1)) * 20

    # Factor 4 — Slot accuracy (20 pts): mean relative error of predictions vs actuals
    if "hist_demand_slot" in zp.columns:
        _valid = zp[zp["hist_demand_slot"] > 0]
        if not _valid.empty:
            _rel_err = (
                (_valid["predicted_demand"] - _valid["hist_demand_slot"]).abs()
                / _valid["hist_demand_slot"].clip(lower=1)
            ).mean()
            _acc_contrib = max(0.0, 1.0 - float(_rel_err)) * 20
        else:
            _acc_contrib = 12.0
    else:
        _acc_contrib = 12.0

    _conf       = max(20, min(99, int(_r2_contrib + _mae_contrib + _data_contrib + _acc_contrib)))
    _conf_color = "#10B981" if _conf >= 75 else ("#F7C948" if _conf >= 50 else "#EF4444")
    _opp_color  = "#EF4444" if best_opp >= 80 else ("#F97316" if best_opp >= 55 else "#F7C948")

    st.markdown(f"""
    <div class="ai-rec-card">
      <div class="ai-rec-tag">
        <span>🤖 {t("aicard_label")}</span>
        <span style="background:rgba(247,201,72,.12);color:#F7C948;
          border:1px solid rgba(247,201,72,.30);padding:3px 12px;
          border-radius:20px;font-size:.68rem;font-weight:800">
          {live_dow_lbl[:3].upper()}  {live_hour:02d}:00 · {t("aicard_now")}
        </span>
      </div>
      <div class="ai-rec-zone">{best_name}</div>
      <div class="ai-rec-boro">{best_boro}</div>
      <div class="ai-rec-row">
        <div class="ai-rec-chip">
          <div class="ai-rec-chip-val" style="color:#F7C948">{best_dem:.0f}</div>
          <div class="ai-rec-chip-lbl">{t("aicard_exp_demand")}</div>
        </div>
        <div class="ai-rec-chip">
          <div class="ai-rec-chip-val" style="color:#10B981">${best_rev:.0f}/hr</div>
          <div class="ai-rec-chip-lbl">{t("aicard_exp_rev")}</div>
        </div>
        <div class="ai-rec-chip">
          <div class="ai-rec-chip-val" style="color:{_opp_color}">{best_opp}/100</div>
          <div class="ai-rec-chip-lbl">{t("aicard_opp_score")}</div>
        </div>
        <div class="ai-rec-chip">
          <div class="ai-rec-chip-val" style="color:{_conf_color}">{_conf}%</div>
          <div class="ai-rec-chip-lbl">{t("aicard_conf_level")}</div>
        </div>
        <div class="ai-rec-chip">
          <div class="ai-rec-chip-val" style="color:#3B82F6">{_r2:.3f}</div>
          <div class="ai-rec-chip-lbl">{t("aicard_r2")}</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Natural Language AI Insights ─────────────────────────────────────────
    _section(t("ins_nl_title"))
    _nl = _generate_nl_insights(zp, hcur, live_hour, live_dow, live_mon)
    if _nl:
        ni_html = '<div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:1rem">'
        for _em, _tx, _mod in _nl:
            ni_html += (
                f'<div class="nl-insight {_mod}">'
                f'<div class="nl-ni-icon">{_em}</div>{_tx}'
                f'</div>'
            )
        ni_html += '</div>'
        st.markdown(ni_html, unsafe_allow_html=True)

    # ── Layout: hero + map ───────────────────────────────────────────────────
    col_hero, col_map = st.columns([1, 1.65], gap="large")

    with col_hero:
        _section(t("live_top5"))
        top5   = zp.nlargest(5, "Opportunity Score").reset_index(drop=True)
        colors = ["r1","r2","r3","r4","r5"]
        emojis = ["🥇","🥈","🥉","4.","5."]
        _thr   = t("live_trips_hr")
        _taf   = t("live_avg_fare")
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
              <div class="zq-stats">🔮 <b>{pd_:.0f}</b> {_thr} &nbsp;·&nbsp; 💰 ${af:.2f} {_taf}</div>
              <div style="display:flex;justify-content:space-between;align-items:center;margin-top:3px">
                <div class="zq-rev">💵 ~${rv:.2f}/hr</div>
                <div style="font-size:.72rem;color:#F7C948;font-weight:700">⭐ {opp}/100</div>
              </div>
            </div>"""
        st.markdown(cards, unsafe_allow_html=True)

        # Demand distribution
        _section(t("live_dist_title"))
        _lv_colors = {"Very High":"#EF4444","High":"#F97316","Medium":"#FACC15","Low":"#10B981"}
        _lv_labels = {
            "Very High": t("very_high"), "High": t("high"),
            "Medium":    t("medium"),    "Low":  t("low"),
        }
        dist_html = '<div style="display:flex;gap:8px;flex-wrap:wrap;margin-top:4px">'
        for lv in ["Very High","High","Medium","Low"]:
            cnt = int((zp["Demand Level"] == lv).sum())
            clr = _lv_colors[lv]
            dist_html += (
                f'<div style="background:{clr}18;border:1px solid {clr}55;'
                f'border-radius:10px;padding:6px 12px;text-align:center">'
                f'<div style="font-size:1.1rem;font-weight:800;color:{clr}">{cnt}</div>'
                f'<div style="font-size:.65rem;color:#9CA3AF;margin-top:1px">{_lv_labels[lv]}</div>'
                f'</div>'
            )
        dist_html += '</div>'
        st.markdown(dist_html, unsafe_allow_html=True)

    with col_map:
        _section(t("live_map_title"))
        _mode   = "heatmap" if _heatmap_lbl in map_mode else "scatter"
        fig_map = _build_map(zp, sel_id=best_id, height=500, mode=_mode)
        st.plotly_chart(fig_map, use_container_width=True,
                        config={"displayModeBar": True}, key="lv_map")

        # Color legend
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;
             padding:8px 14px;background:rgba(16,17,23,.6);border-radius:10px;margin-top:6px">
          <span style="font-size:.72rem;color:#6B7280;font-weight:600;margin-right:2px">{t("live_legend_hdr")}</span>
          <span style="background:#10B98118;border:1px solid #10B98155;border-radius:6px;
            padding:3px 10px;font-size:.74rem;color:#10B981;font-weight:600">{t("live_legend_low")}</span>
          <span style="background:#FACC1518;border:1px solid #FACC1555;border-radius:6px;
            padding:3px 10px;font-size:.74rem;color:#FACC15;font-weight:600">{t("live_legend_med")}</span>
          <span style="background:#F9731618;border:1px solid #F9731655;border-radius:6px;
            padding:3px 10px;font-size:.74rem;color:#F97316;font-weight:600">{t("live_legend_high")}</span>
          <span style="background:#EF444418;border:1px solid #EF444455;border-radius:6px;
            padding:3px 10px;font-size:.74rem;color:#EF4444;font-weight:600">{t("live_legend_vh")}</span>
          <span style="background:rgba(247,201,72,.12);border:1px solid rgba(247,201,72,.35);
            border-radius:6px;padding:3px 10px;font-size:.74rem;color:#F7C948;font-weight:600">
            {t("live_legend_best")}</span>
          <span style="font-size:.70rem;color:#4B5563;margin-left:auto">{t("live_legend_src")}</span>
        </div>
        """, unsafe_allow_html=True)

    # ── What Changed Today ────────────────────────────────────────────────────
    # Compares XGBoost predictions for this slot vs actual historical averages
    # (real trip counts from demand data) — not two model predictions
    _section(t("wc_title"))

    _wc = zp[zp["hist_demand_slot"] > 0].copy() if "hist_demand_slot" in zp.columns else pd.DataFrame()

    if not _wc.empty:
        _wc["delta"]     = _wc["predicted_demand"] - _wc["hist_demand_slot"]
        _wc["delta_pct"] = (_wc["delta"] / _wc["hist_demand_slot"].clip(lower=0.1)) * 100

        _wc_above   = _wc[_wc["delta_pct"] >  10]
        _wc_below   = _wc[_wc["delta_pct"] < -10]
        _pred_total = float(_wc["predicted_demand"].sum())
        _hist_total = float(_wc["hist_demand_slot"].sum())
        _overall_pct = (_pred_total - _hist_total) / max(_hist_total, 0.1) * 100
        _avg_delta   = float(_wc["delta_pct"].mean())
        _pk_today    = int(hcur.loc[hcur["max_demand"].idxmax(), "hour"])
        _slot_lbl    = f'{tl("days")[live_dow][:3]}  {live_hour:02d}:00'

        _tot_clr  = "#10B981" if _overall_pct >= 0 else "#EF4444"
        _avg_clr  = "#10B981" if _avg_delta   >= 0 else "#EF4444"
        _tot_sign = "+" if _overall_pct >= 0 else ""
        _avg_sign = "+" if _avg_delta   >= 0 else ""

        st.markdown(f"""
        <div class="wc-panel">
          <div style="font-size:.75rem;color:#6B7280;margin-bottom:4px">
            {t("wc_sub_hist", slot=_slot_lbl)}
          </div>
          <div class="wc-kpis">
            <div class="wc-kpi">
              <div class="wc-kpi-val" style="color:{_tot_clr}">{_tot_sign}{_overall_pct:.1f}%</div>
              <div class="wc-kpi-lbl">{t("wc_vs_hist")}</div>
            </div>
            <div class="wc-kpi">
              <div class="wc-kpi-val" style="color:#10B981">{len(_wc_above)}</div>
              <div class="wc-kpi-lbl">{t("wc_above_avg")}</div>
            </div>
            <div class="wc-kpi">
              <div class="wc-kpi-val" style="color:#EF4444">{len(_wc_below)}</div>
              <div class="wc-kpi-lbl">{t("wc_below_avg")}</div>
            </div>
            <div class="wc-kpi">
              <div class="wc-kpi-val" style="color:#F7C948">{_pk_today:02d}:00</div>
              <div class="wc-kpi-lbl">{t("wc_peak_today")}</div>
            </div>
            <div class="wc-kpi">
              <div class="wc-kpi-val" style="color:{_avg_clr}">{_avg_sign}{_avg_delta:.1f}%</div>
              <div class="wc-kpi-lbl">{t("wc_avg_delta")}</div>
            </div>
          </div>
        """, unsafe_allow_html=True)

        # Zones predicted above / below historical average
        _rising  = _wc.nlargest(5,  "delta_pct")
        _falling = _wc.nsmallest(5, "delta_pct")

        def _zone_rows_hist(df_r, cls):
            html = ""
            for _, r in df_r.iterrows():
                pct   = float(r["delta_pct"])
                pred  = float(r["predicted_demand"])
                hist  = float(r["hist_demand_slot"])
                sign  = "+" if pct >= 0 else ""
                color = "#10B981" if cls == "wc-up" else "#EF4444"
                html += (
                    f'<div class="wc-row">'
                    f'<div>'
                    f'<div class="wc-name">{r.get("Zone","")}</div>'
                    f'<div class="wc-boro">{r.get("Borough","")} &nbsp;·&nbsp; '
                    f'<span style="color:#9CA3AF">{t("wc_pred_label")}: {pred:.0f} / {t("wc_hist_label")}: {hist:.0f}</span></div>'
                    f'</div>'
                    f'<div class="{cls}">{sign}{pct:.1f}%</div>'
                    f'</div>'
                )
            return html

        st.markdown(f"""
          <div class="wc-cols">
            <div class="wc-half">
              <div class="wc-half-title" style="color:#10B981">{t("wc_top_rising")}</div>
              {_zone_rows_hist(_rising, "wc-up")}
            </div>
            <div class="wc-half">
              <div class="wc-half-title" style="color:#EF4444">{t("wc_top_falling")}</div>
              {_zone_rows_hist(_falling, "wc-dn")}
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown(
            f'<div class="banner">{t("wc_no_hist")}</div>',
            unsafe_allow_html=True,
        )

    # ── Model Validation Panel ────────────────────────────────────────────────
    _section(t("model_val_title"))
    _, _val_met, _, _y_te, _y_pr = load_xgb_model()
    if _val_met is not None and _y_te is not None and _y_pr is not None:
        _y_te_arr = np.array(_y_te)
        _y_pr_arr = np.array(_y_pr)
        _abs_rel   = np.abs(_y_pr_arr - _y_te_arr) / np.maximum(_y_te_arr, 1)
        _within20  = int((_abs_rel < 0.20).mean() * 100)
        _within30  = int((_abs_rel < 0.30).mean() * 100)
        _within10  = int((_abs_rel < 0.10).mean() * 100)
        _n_test    = len(_y_te_arr)

        _val_r2c   = "#10B981" if _val_met["r2"]  >= 0.85 else ("#F7C948" if _val_met["r2"]  >= 0.70 else "#EF4444")
        _val_w20c  = "#10B981" if _within20 >= 80  else ("#F7C948" if _within20 >= 60 else "#EF4444")

        _kpi_row([
            ("📈", f'{_val_met["r2"]:.3f}',  t("model_val_r2"),      t("model_val_r2_sub")),
            ("📉", f'{_val_met["mae"]:.1f}',  t("model_val_mae"),     t("model_val_trips")),
            ("📊", f'{_val_met["rmse"]:.1f}', t("model_val_rmse"),    t("model_val_trips")),
            ("✅", f'{_within10}%',            t("model_val_w10"),     t("model_val_of_test", n=_n_test)),
            ("🎯", f'{_within20}%',            t("model_val_w20"),     ""),
            ("⭐", f'{_within30}%',            t("model_val_w30"),     ""),
        ], top_idx=4)

        # Show one honest accuracy bar for visual clarity
        st.markdown(f"""
        <div style="background:#1A1D27;border:1px solid rgba(255,255,255,.06);
             border-radius:14px;padding:16px 20px;margin-top:8px">
          <div style="font-size:.74rem;color:#9CA3AF;margin-bottom:12px">
            {t("model_val_bar_label")} &nbsp;·&nbsp;
            <span style="color:#6B7280">{t("model_val_n", n=_n_test)}</span>
          </div>
          <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px">
            <div>
              <div style="font-size:.68rem;color:#10B981;font-weight:700;margin-bottom:4px">
                {t("model_val_w10")} — {_within10}%
              </div>
              <div style="background:#2D3044;border-radius:5px;height:8px;overflow:hidden">
                <div style="background:#10B981;width:{_within10}%;height:100%;border-radius:5px"></div>
              </div>
            </div>
            <div>
              <div style="font-size:.68rem;color:{_val_w20c};font-weight:700;margin-bottom:4px">
                {t("model_val_w20")} — {_within20}%
              </div>
              <div style="background:#2D3044;border-radius:5px;height:8px;overflow:hidden">
                <div style="background:{_val_w20c};width:{_within20}%;height:100%;border-radius:5px"></div>
              </div>
            </div>
            <div>
              <div style="font-size:.68rem;color:#3B82F6;font-weight:700;margin-bottom:4px">
                {t("model_val_w30")} — {_within30}%
              </div>
              <div style="background:#2D3044;border-radius:5px;height:8px;overflow:hidden">
                <div style="background:#3B82F6;width:{_within30}%;height:100%;border-radius:5px"></div>
              </div>
            </div>
          </div>
          <div style="font-size:.68rem;color:#4B5563;margin-top:10px">
            R² = {_val_met["r2"]:.3f} &nbsp;·&nbsp; MAE = {_val_met["mae"]:.1f} {t("model_val_trips")} &nbsp;·&nbsp; RMSE = {_val_met["rmse"]:.1f} {t("model_val_trips")}
          </div>
        </div>
        """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2 — My Shift  (prediction + forecast for a chosen zone/time)
# ═════════════════════════════════════════════════════════════════════════════
def page_shift():
    st.markdown(f'<div class="page-title">{t("shift_title")}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="page-sub">{t("shift_sub")}</div>', unsafe_allow_html=True)

    with st.spinner(t("shift_spinner")):
        payload    = load_regression_model()
    model_obj  = payload["model"]
    feat_cols  = payload["feature_cols"]
    y_test     = payload["y_test"]
    model_name = payload["model_name"]

    labels, lut = _zone_data()

    # ── Form (left) | Results (right) ───────────────────────────────────────
    col_form, col_res = st.columns([1, 1.2], gap="large")

    with col_form:
        _section(t("shift_inputs"))
        zone_lbl = st.selectbox(t("shift_zone"), labels, index=0, key="sh_zone")
        loc_id   = lut[zone_lbl]
        defs     = _zone_defaults(loc_id)

        r1, r2 = st.columns(2)
        with r1:
            hour = st.slider(t("shift_hour"), 0, 23, _now_hour, key="sh_hour",
                             help=t("shift_hour_help"))
        with r2:
            shift_date = st.date_input(t("shift_date"), value=_today.date(),
                                       format="DD/MM/YYYY", key="sh_date")
            dow_num  = shift_date.weekday()
            dow_sel  = tl("days")[dow_num]
            mon_num  = shift_date.month
            mon_sel  = tl("months")[mon_num - 1]
            mon_sel_s = tl("months_short")[mon_num - 1]
            year_sel = shift_date.year

        driver_share = st.slider(t("shift_share"), 50, 100, 70, key="sh_share",
                                 help=t("shift_share_help")) / 100.0

        with st.expander(t("shift_advanced")):
            ac1, ac2 = st.columns(2)
            with ac1:
                avg_fare = st.number_input(t("shift_avg_fare"), 1.0, 500.0, float(round(defs["fare"],2)), 0.5, key="sh_fare")
                avg_dist = st.number_input(t("shift_dist"),     0.1, 100.0, float(round(defs["dist"],2)), 0.1, key="sh_dist")
            with ac2:
                avg_dur  = st.number_input(t("shift_dur"),      1.0, 300.0, float(round(defs["dur"],1)),  1.0, key="sh_dur")
                hist_cnt = st.number_input(t("shift_hist"),     0, 500000, int(defs["hist"]), 100, key="sh_hist")
            pax = st.slider(t("shift_pax"), 1, 6, 1, key="sh_pax")

        # AI Driver Assistant
        _zp = _zone_preds(hour, dow_num, mon_num)
        if not _zp.empty:
            best  = _zp.nlargest(1, "predicted_demand").iloc[0]
            bname = str(best.get("Zone", f"Zone {best['PULocationID']}"))
            bdem  = float(best["predicted_demand"])
            brev  = float(best.get("Revenue est ($/hr)", 0))
            same  = best["PULocationID"] == loc_id
            _msg  = t("shift_asst_same") if same else t("shift_asst_suggest", name=bname)
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,rgba(59,130,246,.08),rgba(247,201,72,.06));
                 border:1px solid rgba(247,201,72,.22);border-radius:14px;padding:14px 18px;margin-top:12px">
              <div style="font-size:.82rem;font-weight:700;color:#F7C948;margin-bottom:8px">
                {t("shift_asst_title")}
              </div>
              <div style="font-size:1.05rem;font-weight:800;color:#FAFAFA;margin-bottom:4px">
                {_msg}
              </div>
              <div style="color:#9CA3AF;font-size:.76rem">
                {t("shift_asst_stats", dem=bdem, rev=brev)}
              </div>
            </div>
            """, unsafe_allow_html=True)

    with col_res:
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

        ci_lo = ci_hi = None
        if hasattr(model_obj, "estimators_"):
            X_raw  = np.array([[features[f] for f in feat_cols]], dtype=float)
            tpreds = np.maximum([e.predict(X_raw)[0] for e in model_obj.estimators_], 0)
            ci_lo  = float(np.percentile(tpreds, 10))
            ci_hi  = float(np.percentile(tpreds, 90))

        level, lcls = _demand_level(pred, pd.Series(y_test))

        hist_rows = demand[(demand["PULocationID"] == loc_id) & (demand["hour"] == hour)]
        hist_avg  = float(hist_rows["trip_count"].mean()) if len(hist_rows) > 0 else 0.0
        diff_pct  = ((pred - hist_avg) / max(hist_avg, 1)) * 100
        arr  = "▲" if diff_pct >= 0 else "▼"
        clrd = "#10B981" if diff_pct >= 0 else "#EF4444"

        if ci_lo is not None:
            conf    = int(max(0, 100 - (ci_hi - ci_lo) / max(pred, 1) * 50))
            ci_html = (f'<div style="font-size:.74rem;color:#9CA3AF;margin-top:5px">'
                       f'{t("shift_ci_range", lo=ci_lo, hi=ci_hi, conf=conf)}</div>')
        else:
            ci_html = (f'<div class="banner" style="margin-top:6px;font-size:.74rem">'
                       f'{t("shift_ci_lr")}</div>')

        extrap_html = ""
        if year_sel > 2026:
            extrap_html = (f'<div class="warn-banner" style="margin-bottom:8px;font-size:.76rem">'
                           f'{t("shift_extrap", year=year_sel)}</div>')

        st.markdown(f"""
        {extrap_html}
        <div class="pred-card">
          <div class="pred-number">{pred:.0f}</div>
          <div class="pred-unit">{t("shift_pred_unit")}</div>
          <div style="margin-top:10px">{_badge(level, lcls)}</div>
          <div style="margin-top:8px;font-size:.76rem;color:#9CA3AF">
            {t("shift_hist_avg", hour=hour, avg=hist_avg)}
            &nbsp;<span style="color:{clrd}">{arr} {abs(diff_pct):.1f}%</span>
          </div>
          {ci_html}
        </div>
        """, unsafe_allow_html=True)

        rev_hr  = pred * float(avg_fare) * driver_share
        rev_day = rev_hr * 8
        st.markdown(f"""
        <div class="rev-card">
          <div style="font-size:.72rem;color:#9CA3AF;font-weight:600;margin-bottom:8px">
            {t("shift_rev_title", share=int(driver_share*100))}
          </div>
          <div style="display:flex;gap:22px;flex-wrap:wrap">
            <div>
              <div style="font-size:1.6rem;font-weight:800;color:#10B981">${rev_hr:.2f}</div>
              <div style="font-size:.68rem;color:#6B7280">{t("shift_rev_hr")}</div>
            </div>
            <div>
              <div style="font-size:1.6rem;font-weight:800;color:#10B981">${rev_day:.2f}</div>
              <div style="font-size:.68rem;color:#6B7280">{t("shift_rev_shift")}</div>
            </div>
            <div>
              <div style="font-size:1.6rem;font-weight:800;color:#F7C948">${float(avg_fare):.2f}</div>
              <div style="font-size:.68rem;color:#6B7280">{t("shift_rev_fare")}</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        if level == "Very High":
            st.markdown(f'<div class="alert-extreme">{t("shift_alert_extreme")}</div>',
                        unsafe_allow_html=True)
        elif level == "High":
            st.markdown(f'<div class="alert-high">{t("shift_alert_high")}</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="alert-ok">{t("shift_alert_ok")}</div>',
                        unsafe_allow_html=True)

        _section(t("shift_why"))
        zone_info = zones[zones["LocationID"] == loc_id]
        zname = zone_info["Zone"].iloc[0] if not zone_info.empty else f"Zone {loc_id}"

        def _ins(icon, title, detail):
            return (f'<div class="insight"><span>{icon}</span> <b>{title}</b>'
                    f'<div class="insight-detail">{detail}</div></div>')

        ins = ""
        if   hour in range(7, 10):   ins += _ins("🌅", t("shift_morning_rush"), t("shift_morning_det", hour=hour))
        elif hour in range(17, 20):  ins += _ins("🌆", t("shift_evening_rush"), t("shift_evening_det", hour=hour))
        elif hour >= 22 or hour < 3: ins += _ins("🌙", t("shift_night"),        t("shift_night_det",   hour=hour))
        else:                        ins += _ins("☀️", t("shift_standard"),     t("shift_standard_det",hour=hour))

        if dow_num < 5:
            ins += _ins("📅", t("shift_weekday"), t("shift_weekday_det", day=dow_sel))
        else:
            ins += _ins("📅", t("shift_weekend"), t("shift_weekend_det", day=dow_sel))

        zone_pct = float(np.mean(demand["zone_total_trips"] <= defs["hist"]) * 100)
        if   zone_pct >= 80: ins += _ins("📍", t("shift_zone_high"), t("shift_zone_high_det", zone=zname, pct=100-int(zone_pct)))
        elif zone_pct <= 20: ins += _ins("📍", t("shift_zone_low"),  t("shift_zone_low_det",  zone=zname))

        seasons = {
            (12,1,2):  ("❄️", "shift_winter", "shift_winter_det"),
            (3,4,5):   ("🌸", "shift_spring", "shift_spring_det"),
            (6,7,8):   ("☀️", "shift_summer", "shift_summer_det"),
            (9,10,11): ("🍂", "shift_autumn", "shift_autumn_det"),
        }
        for mgrp, (ic, nk, dk) in seasons.items():
            if mon_num in mgrp:
                ins += _ins(ic, f"{t(nk)} — {mon_sel_s}", t(dk)); break

        if pax >= 3:
            ins += _ins("👥", t("shift_group"), t("shift_group_det", pax=pax))

        st.markdown(ins, unsafe_allow_html=True)

    # ── Forecast tabs ────────────────────────────────────────────────────────
    st.markdown("---")
    _section(t("shift_forecast"))
    st.markdown(f'<div class="banner">{t("shift_fc_banner")}</div>', unsafe_allow_html=True)

    tab_24h, tab_dow, tab_mon = st.tabs([t("tab_24h"), t("tab_dow"), t("tab_monthly")])

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
    _T_DAYS  = tl("days")
    _T_MONS_S = tl("months_short")

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
            xaxis_title=t("shift_hour"), yaxis_title=t("kpi_trips_hr"),
            xaxis=dict(tickmode="linear",tick0=0,dtick=2), **_DRK)
        _pchart(fig_h)
        _kpi_row([
            ("⚡", f"{max(preds_h):.0f}",     t("kpi_peak", h=pk_h),   t("kpi_trips_hr")),
            ("📉", f"{min(preds_h):.0f}",     t("kpi_lowest"),          t("kpi_trips_hr")),
            ("📊", f"{np.mean(preds_h):.0f}", t("kpi_daily_avg"),       t("kpi_trips_hr")),
            ("💵", f"${max(preds_h)*fd['fare']*driver_share:.2f}",
                   t("kpi_peak_rev"), t("shift_rev_hr")),
        ])

    with tab_dow:
        preds_d = [_fp(hour, d, mon_num) for d in range(7)]
        pk_d    = int(np.argmax(preds_d))
        fig_d   = go.Figure(go.Bar(
            x=_T_DAYS, y=preds_d,
            marker_color=["#EF4444" if i==pk_d else "#3B82F6" for i in range(7)],
            text=[f"{p:.0f}" for p in preds_d], textposition="outside",
        ))
        fig_d.update_layout(
            title=f"{t('tab_dow').strip()} — {zone_lbl.split(' — ')[0]} · {hour:02d}:00 · {mon_sel} {year_sel}",
            xaxis_title=t("tab_dow").strip(), yaxis_title=t("kpi_trips_hr"), **_DRK)
        _pchart(fig_d)
        _kpi_row([
            ("🏆", _T_DAYS[pk_d],              t("kpi_busiest_day"),  f"at {hour:02d}:00"),
            ("⚡", f"{max(preds_d):.0f}",      t("kpi_peak_demand"), t("kpi_trips_hr")),
            ("📊", f"{np.mean(preds_d):.0f}",  t("kpi_weekly_avg"),  t("kpi_trips_hr")),
            ("📉", f"{min(preds_d):.0f}",      t("kpi_quietest"),    t("kpi_trips_hr")),
        ])

    with tab_mon:
        preds_m = [_fp(hour, dow_num, mn) for mn in range(1, 13)]
        pk_m    = int(np.argmax(preds_m))
        fig_m   = go.Figure()
        fig_m.add_trace(go.Scatter(
            x=_T_MONS_S, y=preds_m, mode="lines+markers",
            line=dict(color="#3B82F6", width=2.5),
            marker=dict(color=["#EF4444" if i==pk_m else "#3B82F6" for i in range(12)],
                        size=[13 if i==pk_m else 7 for i in range(12)]),
            fill="tozeroy", fillcolor="rgba(59,130,246,.07)",
        ))
        fig_m.update_layout(
            title=f"{t('tab_monthly').strip()} — {zone_lbl.split(' — ')[0]} · {dow_sel} · {hour:02d}:00 · {year_sel}",
            xaxis_title=t("tab_monthly").strip(), yaxis_title=t("kpi_trips_hr"), **_DRK)
        _pchart(fig_m)
        _kpi_row([
            ("🏆", _T_MONS_S[pk_m],            t("kpi_busiest_month"), f"at {hour:02d}:00"),
            ("⚡", f"{max(preds_m):.0f}",      t("kpi_peak_demand"),   t("kpi_trips_hr")),
            ("📊", f"{np.mean(preds_m):.0f}",  t("kpi_annual_avg"),    t("kpi_trips_hr")),
            ("📉", f"{min(preds_m):.0f}",      t("kpi_quietest"),      t("kpi_trips_hr")),
        ])

    st.markdown("---")

    # ── Revenue Simulator ────────────────────────────────────────────────────
    _section(t("revsim_title"))
    st.markdown(
        f'<div style="color:#6B7280;font-size:.76rem;margin-bottom:14px">{t("revsim_sub")}</div>',
        unsafe_allow_html=True,
    )

    rs1, rs2, rs3 = st.columns([2.5, 1.2, 1.2])
    with rs1:
        rs_zone_lbl = st.selectbox(
            t("revsim_zone"), labels,
            index=labels.index(zone_lbl) if zone_lbl in labels else 0,
            key="rv_zone",
        )
        rs_loc = lut[rs_zone_lbl]
    with rs2:
        rs_hours = st.slider(t("revsim_hours"), 1, 12, 8, key="rv_hours")
    with rs3:
        rs_share_pct = st.slider(
            t("revsim_share"), 50, 100, int(driver_share * 100), key="rv_share"
        )
        rs_share = rs_share_pct / 100.0

    # Predict demand for the chosen zone at the current shift time
    rs_defs  = _zone_defaults(rs_loc)
    rs_feats = {
        "pickup_location_id":    float(rs_loc),
        "pickup_hour":           float(hour),
        "pickup_day_of_week":    float(dow_num),
        "pickup_month":          float(mon_num),
        "historical_trip_count": rs_defs["hist"],
        "avg_fare_amount":       rs_defs["fare"],
        "avg_trip_distance":     rs_defs["dist"],
        "avg_trip_duration":     rs_defs["dur"],
        "year":                  float(year_sel),
    }
    rs_pred_hr = predict_regression(payload, rs_feats)
    rs_trips   = rs_pred_hr * rs_hours
    rs_earn    = rs_trips * rs_defs["fare"] * rs_share
    rs_level, rs_lcls = _demand_level(rs_pred_hr, pd.Series(y_test))

    # Revenue range: RF tree-spread (P10 / P90) or ±20 % fallback
    if hasattr(model_obj, "estimators_"):
        X_rs     = np.array([[rs_feats[f] for f in feat_cols]], dtype=float)
        rs_trees = np.maximum([e.predict(X_rs)[0] for e in model_obj.estimators_], 0)
        rs_lo_hr = float(np.percentile(rs_trees, 10))
        rs_hi_hr = float(np.percentile(rs_trees, 90))
    else:
        rs_lo_hr = rs_pred_hr * 0.80
        rs_hi_hr = rs_pred_hr * 1.20

    rs_rev_lo  = rs_lo_hr   * rs_hours * rs_defs["fare"] * rs_share
    rs_rev_mid = rs_pred_hr * rs_hours * rs_defs["fare"] * rs_share
    rs_rev_hi  = rs_hi_hr   * rs_hours * rs_defs["fare"] * rs_share

    # ── 4 output KPIs ────────────────────────────────────────────────────────
    _lv_icon = {"Very High": "🔴", "High": "🟠", "Medium": "🟡", "Low": "🟢"}
    _lv_tkey = {"Very High": "very_high", "High": "high", "Medium": "medium", "Low": "low"}
    _kpi_row([
        ("🚖", f"{rs_trips:.0f}",         t("revsim_est_trips"), t("revsim_over_shift", h=rs_hours)),
        ("💵", f"${rs_earn:.2f}",          t("revsim_est_earn"),  t("revsim_at_share", share=rs_share_pct)),
        ("💰", f"${rs_defs['fare']:.2f}",  t("shift_rev_fare"),   t("kpi_per_trip")),
        (_lv_icon[rs_level], t(_lv_tkey[rs_level]), t("revsim_opp"), ""),
    ], top_idx=1)

    # ── Revenue range cards ───────────────────────────────────────────────────
    _section(t("revsim_range_title"))
    rra, rrb, rrc = st.columns(3)
    _range_data = [
        (rra, t("revsim_pessimistic"), rs_rev_lo,  rs_lo_hr * rs_hours,   "#EF4444", "rgba(239,68,68,.08)"),
        (rrb, t("revsim_expected"),    rs_rev_mid, rs_pred_hr * rs_hours, "#F7C948", "rgba(247,201,72,.08)"),
        (rrc, t("revsim_optimistic"),  rs_rev_hi,  rs_hi_hr * rs_hours,  "#10B981", "rgba(16,185,129,.08)"),
    ]
    for col, lbl, rev, trps, clr, bg in _range_data:
        is_mid = (rev == rs_rev_mid)
        delta  = None if is_mid else ((rev - rs_rev_mid) / max(rs_rev_mid, 1)) * 100
        d_html = (
            f'<div style="font-size:.72rem;color:{clr};margin-top:4px">'
            f'{"▲" if delta > 0 else "▼"} {abs(delta):.1f}%</div>'
            if delta is not None else
            '<div style="font-size:.72rem;color:#6B7280;margin-top:4px">baseline</div>'
        )
        border = f"2px solid {clr}60" if is_mid else f"1px solid {clr}35"
        with col:
            st.markdown(f"""
            <div style="background:{bg};border:{border};border-radius:14px;
                        padding:18px 14px;text-align:center">
              <div style="font-size:.68rem;color:#9CA3AF;text-transform:uppercase;
                          letter-spacing:.07em;margin-bottom:8px">{lbl}</div>
              <div style="font-size:1.9rem;font-weight:900;color:{clr}">${rev:.2f}</div>
              <div style="font-size:.72rem;color:#6B7280;margin-top:4px">
                {trps:.0f} {t("kpi_trips_hr")}
              </div>
              {d_html}
            </div>
            """, unsafe_allow_html=True)

    # ── Opportunity level alert ───────────────────────────────────────────────
    st.markdown('<div style="margin-top:10px"></div>', unsafe_allow_html=True)
    if rs_level == "Very High":
        st.markdown(f'<div class="alert-extreme">{t("shift_alert_extreme")}</div>',
                    unsafe_allow_html=True)
    elif rs_level == "High":
        st.markdown(f'<div class="alert-high">{t("shift_alert_high")}</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="alert-ok">{t("shift_alert_ok")}</div>',
                    unsafe_allow_html=True)

    st.markdown("---")

    # ── Relocation Simulator ─────────────────────────────────────────────────
    with st.expander(t("reloc_title")):
        tgt_lbl = st.selectbox(t("reloc_target"), labels, index=min(1, len(labels)-1), key="rs_tgt")
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

        if   d_pct > 20:  rc, rt = "#10B981", t("reloc_strongly")
        elif d_pct > 5:   rc, rt = "#F7C948", t("reloc_recommended")
        elif d_pct > -5:  rc, rt = "#3B82F6", t("reloc_neutral")
        else:             rc, rt = "#EF4444", t("reloc_not")

        tgt_z  = zones[zones["LocationID"] == tgt_id]
        tgt_zn = tgt_z["Zone"].iloc[0] if not tgt_z.empty else f"Zone {tgt_id}"

        st.markdown(f"""
        <div class="reloc-card" style="border:1.5px solid {rc}">
          <b style="color:#FAFAFA">{t("reloc_moving", zone=tgt_zn)}</b>
          <div style="display:flex;gap:24px;margin-top:10px;flex-wrap:wrap">
            <div>
              <div style="color:#9CA3AF;font-size:.72rem">{t("reloc_demand_delta")}</div>
              <div style="font-size:1.35rem;font-weight:800;color:#F7C948">
                {"▲" if d_abs>=0 else "▼"} {abs(d_abs):.0f} {t("kpi_trips_hr")} ({d_pct:+.1f}%)
              </div>
            </div>
            <div>
              <div style="color:#9CA3AF;font-size:.72rem">{t("reloc_rev_delta")}</div>
              <div style="font-size:1.35rem;font-weight:800;
                   color:{"#10B981" if tgt_rev_rs>=cur_rev_rs else "#EF4444"}">
                {"▲" if tgt_rev_rs>=cur_rev_rs else "▼"} ${abs(tgt_rev_rs-cur_rev_rs):.2f}
              </div>
            </div>
            <div>
              <div style="color:#9CA3AF;font-size:.72rem">{t("reloc_rec_label")}</div>
              <div style="font-size:1rem;font-weight:700;color:{rc}">{rt}</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)
        _kpi_row([
            ("📍", f"{pred:.0f}",        t("reloc_cur_kpi", zone=zname[:16]),  t("kpi_trips_hr")),
            ("🎯", f"{tgt_pred:.0f}",    t("reloc_tgt_kpi", zone=tgt_zn[:16]),t("kpi_trips_hr")),
            ("💰", f"${cur_rev_rs:.2f}", t("reloc_cur_rev"),  t("kpi_share", share=int(driver_share*100))),
            ("💵", f"${tgt_rev_rs:.2f}", t("reloc_tgt_rev"),  t("kpi_share", share=int(driver_share*100))),
        ])

    # ── What If? ──────────────────────────────────────────────────────────────
    with st.expander(t("whatif_title")):
        wc1, wc2 = st.columns(2)
        with wc1:
            wi_hour = st.slider(t("whatif_hour"), 0, 23, hour, key="wi_hour")
        with wc2:
            wi_date    = st.date_input(t("whatif_date"), value=shift_date,
                                       format="DD/MM/YYYY", key="wi_date")
            wi_dow     = wi_date.weekday()
            wi_dow_lbl = tl("days")[wi_dow]
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
              <div style="color:#9CA3AF;font-size:.72rem;margin-bottom:6px">{t("whatif_current")}</div>
              <div style="font-size:2.2rem;font-weight:800;color:#F7C948">{pred:.0f}</div>
              <div style="color:#9CA3AF;font-size:.75rem">{t("kpi_trips_hr")} · {dow_sel[:3]} {hour:02d}:00 · {year_sel}</div>
            </div>""", unsafe_allow_html=True)
        with wcb:
            cw = "#10B981" if wi_d >= 0 else "#EF4444"
            st.markdown(f"""
            <div style="background:#1A1D27;border:1px solid {cw}40;
                 border-radius:12px;padding:16px;text-align:center">
              <div style="color:#9CA3AF;font-size:.72rem;margin-bottom:6px">{t("whatif_scenario")}</div>
              <div style="font-size:2.2rem;font-weight:800;color:{cw}">{wi_pred:.0f}</div>
              <div style="color:#9CA3AF;font-size:.75rem">{t("kpi_trips_hr")} · {wi_dow_lbl[:3]} {wi_hour:02d}:00 · {wi_year}</div>
              <div style="font-size:.82rem;color:{cw};margin-top:4px">
                {"▲" if wi_d>=0 else "▼"} {abs(wi_d):.0f} ({wi_p:+.1f}%)
              </div>
            </div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Analytics  (historical dashboard with always-visible filters)
# ═════════════════════════════════════════════════════════════════════════════
def page_analytics():
    st.markdown(f'<div class="page-title">{t("analytics_title")}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="page-sub">{t("analytics_sub")}</div>',
                unsafe_allow_html=True)

    # ── Filters (always visible, not collapsed) ──────────────────────────────
    _section(t("analytics_filters"))
    fa, fb, fc, fd = st.columns(4)
    with fa:
        sel_years = st.multiselect(t("analytics_years"), [2023,2024,2025,2026],
                                   default=[2023,2024,2025,2026], key="an_years")
    with fb:
        _all_lbl = t("analytics_all")
        boros    = [_all_lbl] + sorted(df_all["pickup_borough"].dropna().unique().tolist()) \
                   if "pickup_borough" in df_all.columns else [_all_lbl]
        sel_boro = st.selectbox(t("analytics_borough"), boros, key="an_boro")
    with fc:
        _t_mons_s = tl("months_short")
        avail_m   = sorted(int(m) for m in df_all["month"].dropna().unique() if 1<=int(m)<=12)
        avail_mlb = [_t_mons_s[m-1] for m in avail_m]
        sel_mlb   = st.multiselect(t("analytics_months"), avail_mlb, default=avail_mlb, key="an_mon")
        sel_months = [avail_m[avail_mlb.index(l)] for l in sel_mlb] if sel_mlb else avail_m
    with fd:
        hr_range = st.slider(t("analytics_hours"), 0, 23, (0, 23), key="an_hr")

    active_years = sorted(sel_years) if sel_years else [2023,2024,2025,2026]
    fdf = df_all[df_all["year"].isin(active_years)].copy()
    if sel_boro != _all_lbl and "pickup_borough" in fdf.columns:
        fdf = fdf[fdf["pickup_borough"] == sel_boro]
    fdf = fdf[fdf["month"].isin(sel_months)]
    fdf = fdf[(fdf["hour"] >= hr_range[0]) & (fdf["hour"] <= hr_range[1])].reset_index(drop=True)

    if fdf.empty:
        st.warning(t("analytics_no_data"))
        return

    fkpis   = compute_kpis(fdf)
    yrs_str = " · ".join(str(y) for y in active_years)

    # KPIs
    _section(t("analytics_overview", yrs=yrs_str, n=fkpis["total_trips"]))
    _kpi_row([
        ("🚖", f"{fkpis['total_trips']:,}",       t("analytics_total_trips"), ""),
        ("💰", f"${fkpis['avg_fare']:.2f}",        t("analytics_avg_fare"),   t("kpi_per_trip")),
        ("📍", f"{fkpis['avg_distance']:.1f} mi",  t("analytics_avg_dist"),   t("kpi_per_trip")),
        ("⏱️", f"{fkpis['avg_duration']:.1f} min", t("analytics_avg_dur"),    t("kpi_per_trip")),
        ("⚡", f"{fkpis['peak_hour']}:00",          t("analytics_peak_hour"),  t("kpi_most_demand")),
        ("🗺️", f"{fkpis['active_zones']}",         t("analytics_zones"),      t("kpi_pickup_areas")),
    ], top_idx=0)
    _kpi_row([
        ("💳", f"{fkpis['credit_pct']:.1f}%",    t("analytics_credit"),   t("kpi_of_payments")),
        ("🏆", fkpis["top_zone"][:26],            t("analytics_top_zone"), ""),
        ("💵", f"${fkpis['total_revenue']:,.0f}", t("analytics_revenue"),  t("kpi_gross_fares")),
    ], top_idx=2)

    # Auto-Insights
    _section(t("analytics_insights"))
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
    _section(t("analytics_where_when"))
    c1, c2 = st.columns(2)
    with c1: _pchart(charts.trips_by_hour(fdf), h=300)
    with c2: _pchart(charts.trips_by_dow(fdf),  h=300)

    _section(t("analytics_trends"))
    c3, c4 = st.columns(2)
    with c3: _pchart(charts.monthly_trend(fdf),       h=300)
    with c4: _pchart(charts.top_zones(fdf, top_n=10), h=300)

    _section(t("analytics_heatmap"))
    _pchart(charts.demand_heatmap(fdf), h=280)

    if "pickup_borough" in fdf.columns:
        _section(t("analytics_boroughs"))
        _pchart(charts.borough_flow(fdf), h=300)

    _section(t("analytics_yoy"))
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

    # ── Borough Comparison ────────────────────────────────────────────────────
    if "pickup_borough" not in fdf.columns:
        return

    _section(t("bc_title"))
    st.markdown(f'<div class="page-sub" style="margin-bottom:1rem">{t("bc_sub")}</div>',
                unsafe_allow_html=True)

    _BORO_COLORS = {
        "Manhattan": "#F7C948", "Brooklyn": "#3B82F6",
        "Queens":    "#10B981", "Bronx":    "#8B5CF6",
    }
    _MAIN_BOROS = ["Manhattan", "Brooklyn", "Queens", "Bronx"]
    _total_trips = max(len(fdf), 1)

    # ── KPI cards (one per borough) ───────────────────────────────────────────
    bc_html = '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:1.2rem">'
    for boro in _MAIN_BOROS:
        sub = fdf[fdf["pickup_borough"] == boro]
        if sub.empty:
            bc_html += f'<div style="background:#13162080;border:1px solid rgba(255,255,255,.06);border-radius:14px;padding:16px 18px"><div style="color:#6B7280;font-size:.8rem">{boro}</div><div style="color:#4B5563;font-size:.75rem;margin-top:6px">No data</div></div>'
            continue
        n_trips  = len(sub)
        pct      = n_trips / _total_trips * 100
        avg_fare = float(sub["fare_amount"].mean())
        avg_dist = float(sub["trip_distance"].mean())
        drv_rev  = avg_fare * 0.70
        peak_h   = int(sub.groupby("hour").size().idxmax())
        color    = _BORO_COLORS[boro]
        bc_html += f"""
        <div style="background:linear-gradient(135deg,#13162090,#1A1D2780);
             border:1.5px solid {color}35;border-top:3px solid {color};
             border-radius:14px;padding:16px 18px">
          <div style="font-size:.7rem;font-weight:700;color:{color};text-transform:uppercase;
               letter-spacing:.08em;margin-bottom:7px">{boro}</div>
          <div style="font-size:1.65rem;font-weight:900;color:#FAFAFA;line-height:1">{n_trips:,}</div>
          <div style="font-size:.65rem;color:#6B7280;margin-bottom:11px">{pct:.1f}% {t("bc_of_trips")}</div>
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px 10px">
            <div>
              <div style="font-size:.84rem;font-weight:700;color:#10B981">${avg_fare:.2f}</div>
              <div style="font-size:.58rem;color:#6B7280;text-transform:uppercase;letter-spacing:.04em">{t("bc_avg_fare")}</div>
            </div>
            <div>
              <div style="font-size:.84rem;font-weight:700;color:{color}">${drv_rev:.2f}</div>
              <div style="font-size:.58rem;color:#6B7280;text-transform:uppercase;letter-spacing:.04em">{t("bc_drv_rev")}</div>
            </div>
            <div>
              <div style="font-size:.84rem;font-weight:700;color:#F7C948">{avg_dist:.1f} mi</div>
              <div style="font-size:.58rem;color:#6B7280;text-transform:uppercase;letter-spacing:.04em">{t("bc_avg_dist")}</div>
            </div>
            <div>
              <div style="font-size:.84rem;font-weight:700;color:#9CA3AF">{peak_h:02d}:00</div>
              <div style="font-size:.58rem;color:#6B7280;text-transform:uppercase;letter-spacing:.04em">{t("bc_peak_hr")}</div>
            </div>
          </div>
        </div>"""
    bc_html += '</div>'
    st.markdown(bc_html, unsafe_allow_html=True)

    # ── Charts: 2×2 grid ─────────────────────────────────────────────────────
    bc1, bc2 = st.columns(2)
    with bc1:
        _section(t("bc_hourly"))
        _pchart(charts.borough_hourly(fdf), h=300)
    with bc2:
        _section(t("bc_revenue"))
        _pchart(charts.borough_revenue(fdf), h=300)

    bc3, bc4 = st.columns(2)
    with bc3:
        _section(t("bc_dow"))
        _pchart(charts.borough_dow(fdf), h=300)
    with bc4:
        _section(t("bc_radar"))
        _pchart(charts.borough_radar(fdf), h=300)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Model  (technical AI insights: performance · clustering · regression)
# ═════════════════════════════════════════════════════════════════════════════
def page_model():
    import plotly.graph_objects as go

    st.markdown(f'<div class="page-title">{t("model_title")}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="page-sub">{t("model_sub")}</div>',
                unsafe_allow_html=True)

    tab_perf, tab_clust, tab_regr = st.tabs(
        [t("tab_perf"), t("tab_clust"), t("tab_regr")]
    )

    # ── Model Performance ─────────────────────────────────────────────────────
    with tab_perf:
        with st.spinner(t("model_spinner_reg")):
            payload = load_regression_model()
        with st.spinner(t("model_spinner_xgb")):
            _, xgb_met, xgb_fi, xgb_yte, xgb_ypred = load_xgb_model()

        mn = payload["model_name"]
        m  = payload["metrics"]
        am = payload.get("all_metrics", {})

        _section(t("model_best_m"))
        _kpi_row([
            ("🤖", mn,                  t("model_active"),  t("kpi_best_r2")),
            ("📉", f"{m['mae']:.2f}",   "MAE",              t("kpi_avg_err")),
            ("📊", f"{m['rmse']:.2f}",  "RMSE",             t("kpi_rmse_desc")),
            ("📈", f"{m['r2']:.3f}",    "R² Score",         t("kpi_var_exp")),
            ("🏋️",f"{payload['n_train']:,}", t("regr_train"), t("kpi_all_years")),
        ], top_idx=3)

        r2v = m["r2"]
        if   r2v > 0.85: badge = t("model_r2_exc")
        elif r2v > 0.70: badge = t("model_r2_good")
        else:            badge = t("model_r2_fair")
        st.info(badge)

        if am:
            _section(t("model_all_comp"))
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

        _section(t("model_avp"))
        dc1, dc2 = st.columns(2)
        with dc1:
            _pchart(reg.chart_actual_vs_pred(payload["y_test"], payload["y_pred"], mn), h=320)
        with dc2:
            if payload.get("feature_importance") is not None:
                _pchart(reg.chart_feature_importance(payload["feature_importance"], mn), h=320)
            else:
                st.info(t("model_no_fi"))

    # ── Clustering ────────────────────────────────────────────────────────────
    with tab_clust:
        feat_map = clust.available_features(demand)
        if not feat_map:
            st.error(t("clust_no_feats"))
        else:
            cc1, cc2, cc3 = st.columns([2.5, 1, 1])
            with cc1:
                sel_f = st.multiselect(t("clust_feats"), list(feat_map.keys()),
                                       default=["trip_count","avg_fare","avg_distance"],
                                       format_func=lambda x: feat_map[x], key="cl_feats")
            with cc2:
                k = st.slider(t("clust_k"), 2, 8, 3, key="cl_k")
            with cc3:
                normalize = st.checkbox(t("clust_norm"), True, key="cl_norm")

            if len(sel_f) < 2:
                st.warning(t("clust_warn"))
            else:
                with st.spinner(t("clust_spinner")):
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
                    _section(t("clust_elbow"))
                    with st.spinner(t("clust_elbow_spin")):
                        ks_n, in_n = clust.compute_elbow(demand, sel_f, normalize=True)
                        ks_r, in_r = clust.compute_elbow(demand, sel_f, normalize=False)
                    _pchart(clust.chart_elbow(ks_n, in_n, in_r), h=280)

                st.markdown(
                    f'<div class="banner">K={k} · Inertia: {inertia:,.0f} · '
                    f'{"Normalized ✅" if normalize else "Raw ⚠️"}</div>',
                    unsafe_allow_html=True)

                _section(t("clust_stats"))
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
        _section(t("regr_feats"))
        rc1, rc2 = st.columns([3, 1])
        with rc1:
            sel_reg = st.multiselect(
                t("regr_feats"), list(reg.REGRESSION_FEATURES.keys()),
                default=list(reg.REGRESSION_FEATURES.keys()),
                format_func=lambda x: reg.REGRESSION_FEATURES[x], key="rg_feats")
        with rc2:
            st.markdown(f'<div style="margin-top:2rem;color:#9CA3AF;font-size:.8rem">'
                        f'{t("regr_target")}</div>', unsafe_allow_html=True)

        if not sel_reg:
            st.warning(t("regr_warn"))
        else:
            with st.spinner(t("regr_spinner")):
                out     = reg.get_regression_results(tuple(sorted(sel_reg)))
            results = out["results"]
            y_te    = out["y_te"]
            best_n  = max(results, key=lambda x: results[x]["r2"])
            best_r  = results[best_n]

            _kpi_row([
                ("🏆", best_n,                    t("regr_best"),     t("kpi_best_r2")),
                ("📉", f"{best_r['mae']:.2f}",    t("regr_best_mae"), ""),
                ("📈", f"{best_r['r2']:.3f}",     t("regr_best_r2"), ""),
                ("🏋️",f"{out['n_train']:,}",      t("regr_train"),    ""),
            ], top_idx=2)

            rc3, rc4 = st.columns(2)
            with rc3: _pchart(reg.chart_metrics_bar(results), h=300)
            with rc4: _pchart(reg.chart_r2_bar(results),      h=300)

            _section(t("model_avp"))
            tabs_m = st.tabs([f"📊 {mn}" for mn in results])
            for tab, (mname, res) in zip(tabs_m, results.items()):
                with tab:
                    _pchart(reg.chart_actual_vs_pred(y_te, res["y_pred"], mname), h=320)

            fi_list = [(mn, r) for mn, r in results.items() if "feature_importance" in r]
            if fi_list:
                _section(t("regr_fi"))
                fi_cols = st.columns(len(fi_list))
                for col, (mname, res) in zip(fi_cols, fi_list):
                    with col:
                        _pchart(reg.chart_feature_importance(res["feature_importance"], mname), h=300)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 5 — Driver Intelligence Center
# ═════════════════════════════════════════════════════════════════════════════
def page_intelligence():
    import plotly.graph_objects as go

    st.markdown(f'<div class="page-title">{t("intel_title")}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="page-sub">{t("intel_sub")}</div>', unsafe_allow_html=True)

    # ── Inputs ────────────────────────────────────────────────────────────────
    ic1, ic2 = st.columns([2, 1])
    with ic1:
        intel_date    = st.date_input(t("intel_date"), value=_today.date(),
                                      format="DD/MM/YYYY", key="it_date")
        intel_dow     = intel_date.weekday()
        intel_mon     = intel_date.month
        intel_dow_lbl = tl("days")[intel_dow]
        intel_mon_lbl = tl("months")[intel_mon - 1]
    with ic2:
        intel_hour = st.slider(t("intel_hour"), 0, 23, _now_hour, key="it_hour")

    # ── Core predictions (all three calls hit cache after first run) ──────────
    with st.spinner(t("intel_spinner")):
        zp   = _zone_preds(intel_hour, intel_dow, intel_mon)
        hcur = _hour_curve(intel_dow, intel_mon)
        dcur = _dow_curve(intel_hour, intel_mon)

    if zp.empty or hcur.empty:
        st.error(t("live_error"))
        return

    top3  = zp.nlargest(3, "Opportunity Score").reset_index(drop=True)
    best  = top3.iloc[0]
    score = int(best.get("Opportunity Score", 0))

    # ── Confidence calculation ────────────────────────────────────────────────
    mask      = ((demand["hour"]  == intel_hour) &
                 (demand["dow"]   == intel_dow)  &
                 (demand["month"] == intel_mon))
    n_pts     = int(mask.sum())
    max_zones = int(demand["PULocationID"].nunique())
    data_conf = min(100, int((n_pts / max(max_zones * 0.4, 1)) * 100))
    _, xgb_met, *_ = load_xgb_model()
    model_r2  = float(xgb_met["r2"]) if xgb_met else 0.75
    conf      = max(20, min(99, int(data_conf * 0.55 + model_r2 * 100 * 0.45)))
    if   conf >= 78: conf_lbl = t("intel_conf_high"); conf_clr = "#10B981"
    elif conf >= 52: conf_lbl = t("intel_conf_med");  conf_clr = "#F7C948"
    else:            conf_lbl = t("intel_conf_low");  conf_clr = "#EF4444"

    # ── Score ring colour ─────────────────────────────────────────────────────
    if   score >= 80: sc_clr = "#10B981"; sc_lbl = t("intel_score_exc")
    elif score >= 60: sc_clr = "#F7C948"; sc_lbl = t("intel_score_high")
    elif score >= 40: sc_clr = "#F97316"; sc_lbl = t("intel_score_med")
    else:             sc_clr = "#6B7280"; sc_lbl = t("intel_score_low")
    best_zone_name = str(best.get("Zone", ""))

    # ══════════════════════════════════════════════════════════════════════════
    # Row 1 — Opportunity Score  +  Confidence Meter
    # ══════════════════════════════════════════════════════════════════════════
    s1, s2 = st.columns(2)

    with s1:
        _section(t("intel_score_lbl"))
        st.markdown(f"""
        <div style="background:#1A1D27;border:1px solid {sc_clr}30;border-radius:18px;
                    padding:28px 20px;text-align:center">
          <div style="width:164px;height:164px;border-radius:50%;margin:0 auto 18px;
               background:conic-gradient({sc_clr} 0% {score}%, rgba(255,255,255,.07) {score}% 100%);
               display:flex;align-items:center;justify-content:center;
               box-shadow:0 0 44px {sc_clr}28">
            <div style="width:124px;height:124px;border-radius:50%;background:#111318;
                 display:flex;flex-direction:column;align-items:center;justify-content:center">
              <div style="font-size:2.7rem;font-weight:900;color:{sc_clr};line-height:1">{score}</div>
              <div style="font-size:.72rem;color:#6B7280;margin-top:2px">/ 100</div>
            </div>
          </div>
          <div style="font-size:1rem;font-weight:700;color:#FAFAFA;margin-bottom:5px">{sc_lbl}</div>
          <div style="font-size:.74rem;color:#6B7280">{t("intel_score_zone")}</div>
          <div style="font-size:.84rem;font-weight:600;color:{sc_clr};margin-top:4px">{best_zone_name}</div>
        </div>
        """, unsafe_allow_html=True)

    with s2:
        _section(t("intel_conf_title"))
        c_start = "#EF4444" if conf < 52 else "#F7C948"
        st.markdown(f"""
        <div style="background:#1A1D27;border:1px solid {conf_clr}30;border-radius:18px;
                    padding:28px 24px">
          <div style="font-size:3rem;font-weight:900;color:{conf_clr};line-height:1;margin-bottom:6px">
            {conf}%
          </div>
          <div style="font-size:.9rem;font-weight:700;color:{conf_clr};margin-bottom:18px">
            {conf_lbl}
          </div>
          <div style="background:#2D3044;border-radius:8px;height:12px;overflow:hidden;margin-bottom:14px">
            <div style="background:linear-gradient(90deg,{c_start},{conf_clr});
                 width:{conf}%;height:100%;border-radius:8px"></div>
          </div>
          <div style="font-size:.74rem;color:#6B7280;line-height:1.85">
            📊 {t("intel_conf_pts", n=n_pts)}<br>
            🤖 {t("intel_conf_model", r2=model_r2)}
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div style="margin-top:1rem"></div>', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # Row 2 — Top 3 Recommended Zones
    # ══════════════════════════════════════════════════════════════════════════
    _section(t("intel_top3_title"))
    st.markdown(
        f'<div style="color:#6B7280;font-size:.76rem;margin-bottom:12px">'
        f'{t("intel_top3_sub", dow=intel_dow_lbl, month=intel_mon_lbl, hour=intel_hour)}'
        f'</div>', unsafe_allow_html=True)

    rank_emoji  = ["🥇", "🥈", "🥉"]
    rank_colors = ["#F7C948", "#9CA3AF", "#CD7C2F"]
    zcols = st.columns(3)

    for col, idx in zip(zcols, range(min(3, len(top3)))):
        r    = top3.iloc[idx]
        rn   = str(r.get("Zone", ""))
        rb   = str(r.get("Borough", ""))
        dem  = float(r["predicted_demand"])
        fare = float(r.get("avg_fare", 15.0))
        rev8 = dem * fare * 0.7 * 8
        opp  = int(r.get("Opportunity Score", 0))
        re_  = rank_emoji[idx]
        rc_  = rank_colors[idx]

        with col:
            st.markdown(f"""
            <div style="background:#1A1D27;border:1px solid {rc_}35;border-radius:16px;
                        padding:18px 16px;height:100%">
              <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:10px">
                <span style="font-size:1.6rem">{re_}</span>
                <span style="background:{rc_}18;border:1px solid {rc_}50;border-radius:20px;
                  font-size:.8rem;font-weight:800;color:{rc_};padding:4px 12px">⭐ {opp}/100</span>
              </div>
              <div style="font-size:.92rem;font-weight:700;color:#FAFAFA;margin-bottom:3px;
                          line-height:1.25">{rn}</div>
              <div style="font-size:.72rem;color:#6B7280;margin-bottom:12px">{rb}</div>
              <div style="font-size:.78rem;color:#9CA3AF;margin-bottom:3px">
                🔮 <b style="color:#F7C948">{dem:.0f}</b> {t("intel_trips_hr")}
              </div>
              <div style="font-size:.78rem;color:#9CA3AF;margin-bottom:10px">
                💰 <b>${fare:.2f}</b> {t("intel_avg_fare")}
              </div>
              <div style="font-size:.86rem;font-weight:700;color:#10B981">
                {t("intel_top3_shift", rev=rev8)}
              </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div style="margin-top:1rem"></div>', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # Row 3 — Best Time To Drive  (24-hour demand curve)
    # ══════════════════════════════════════════════════════════════════════════
    _section(t("intel_time_title"))
    st.markdown(
        f'<div style="color:#6B7280;font-size:.76rem;margin-bottom:10px">'
        f'{t("intel_time_sub", dow=intel_dow_lbl, month=intel_mon_lbl)}'
        f'</div>', unsafe_allow_html=True)

    sorted_h = hcur.sort_values("max_demand", ascending=False).reset_index(drop=True)
    peak_h   = int(sorted_h.iloc[0]["hour"])
    peak_dem = float(sorted_h.iloc[0]["max_demand"])
    second_h = int(sorted_h.iloc[1]["hour"]) if len(sorted_h) > 1 else peak_h
    quiet_h  = int(sorted_h.iloc[-1]["hour"])

    _kpi_row([
        ("⭐", f"{peak_h:02d}:00",   t("intel_peak_hour"), f"{peak_dem:.0f} {t('intel_trips_hr')}"),
        ("📈", f"{second_h:02d}:00", t("intel_2nd_peak"),  ""),
        ("🌙", f"{quiet_h:02d}:00",  t("intel_quiet"),     ""),
    ], top_idx=0)

    p25h = float(hcur["max_demand"].quantile(0.25))
    p75h = float(hcur["max_demand"].quantile(0.75))
    p90h = float(hcur["max_demand"].quantile(0.90))
    h_clrs = [
        "#EF4444" if v >= p90h else
        "#F97316" if v >= p75h else
        "#FACC15" if v >= p25h else "#3B82F6"
        for v in hcur["max_demand"]
    ]
    fig_h = go.Figure(go.Bar(
        x=[f"{h:02d}:00" for h in hcur["hour"]],
        y=hcur["max_demand"].round(1),
        marker=dict(color=h_clrs, line_width=0),
        hovertemplate="%{x}  %{y:.0f} trips/hr<extra></extra>",
    ))
    for hl, hc, hn in [
        (f"{intel_hour:02d}:00", "rgba(255,255,255,.22)", "Now"),
        (f"{peak_h:02d}:00",     "rgba(247,201,72,.40)",  "⭐ Peak"),
    ]:
        fig_h.add_vline(x=hl, line_color=hc, line_width=1.5, line_dash="dot",
                        annotation_text=hn, annotation_position="top",
                        annotation_font=dict(color=hc, size=10))
    fig_h.update_layout(**_DRK, height=270,
                        xaxis=dict(tickangle=-45, tickfont=dict(size=8)),
                        yaxis_title="trips/hr", bargap=0.14,
                        margin=dict(t=36, b=10))
    _pchart(fig_h)

    # ══════════════════════════════════════════════════════════════════════════
    # Row 4 — Best Day of the Week
    # ══════════════════════════════════════════════════════════════════════════
    _section(t("intel_day_title"))
    st.markdown(
        f'<div style="color:#6B7280;font-size:.76rem;margin-bottom:10px">'
        f'{t("intel_day_sub", hour=intel_hour, month=intel_mon_lbl)}'
        f'</div>', unsafe_allow_html=True)

    best_d_row  = dcur.nlargest(1, "max_demand").iloc[0]
    best_d      = int(best_d_row["dow"])
    best_d_lbl  = tl("days")[best_d]
    today_rows  = dcur[dcur["dow"] == intel_dow]
    today_dem   = float(today_rows["max_demand"].iloc[0]) if not today_rows.empty else 0.0

    _kpi_row([
        ("🏆", best_d_lbl[:3],          t("intel_best_day"),  f"{float(best_d_row['max_demand']):.0f} {t('intel_trips_hr')}"),
        ("📅", tl("days")[intel_dow][:3], "Today",             f"{today_dem:.0f} {t('intel_trips_hr')}"),
    ], top_idx=0)

    p25d = float(dcur["max_demand"].quantile(0.25))
    p75d = float(dcur["max_demand"].quantile(0.75))
    p90d = float(dcur["max_demand"].quantile(0.90))
    d_clrs = [
        "#EF4444" if v >= p90d else
        "#F97316" if v >= p75d else
        "#FACC15" if v >= p25d else "#3B82F6"
        for v in dcur["max_demand"]
    ]
    d_labels = tl("days_short")
    fig_d = go.Figure(go.Bar(
        x=d_labels,
        y=dcur["max_demand"].round(1),
        marker=dict(color=d_clrs, line_width=0),
        hovertemplate="%{x}  %{y:.0f} trips/hr<extra></extra>",
    ))
    fig_d.add_vline(
        x=d_labels[intel_dow],
        line_color="rgba(255,255,255,.22)", line_width=1.5, line_dash="dot",
        annotation_text="Today", annotation_position="top",
        annotation_font=dict(color="rgba(255,255,255,.45)", size=10),
    )
    fig_d.update_layout(**_DRK, height=220,
                        xaxis=dict(tickfont=dict(size=10)),
                        yaxis_title="trips/hr", bargap=0.22,
                        margin=dict(t=36, b=10))
    _pchart(fig_d)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 6 — Future Demand Explorer
# ═════════════════════════════════════════════════════════════════════════════
def page_future():
    st.markdown(f'<div class="page-title">{t("future_title")}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="page-sub">{t("future_sub")}</div>', unsafe_allow_html=True)

    # ── Controls ─────────────────────────────────────────────────────────────
    fc1, fc2, fc3, fc4 = st.columns([1.6, 1.2, 1, 1.2])
    with fc1:
        fut_year = st.slider(t("future_year"), 2025, 2035,
                             min(max(_now_year + 1, 2025), 2035), key="fut_yr")
    with fc2:
        _mon_opts = tl("months")
        fut_mon_lbl = st.selectbox(t("future_month_lbl"), _mon_opts,
                                   index=_now_mon - 1, key="fut_mon")
        fut_mon = _mon_opts.index(fut_mon_lbl) + 1
    with fc3:
        fut_hour = st.slider(t("future_hour"), 0, 23, _now_hour, key="fut_hr")
    with fc4:
        _dow_opts = tl("days")
        fut_dow_lbl = st.selectbox(t("future_dow_lbl"), _dow_opts,
                                   index=_now_dow, key="fut_dow")
        fut_dow = _dow_opts.index(fut_dow_lbl)

    BASELINE = 2024

    with st.spinner(t("future_spinner")):
        master  = _future_full_forecast(fut_mon, fut_hour, fut_dow)
        monthly = _future_monthly_profile(fut_year, fut_hour, fut_dow)

    if master.empty:
        st.error("Regression model unavailable.")
        return

    # ── Derived aggregates ────────────────────────────────────────────────────
    annual = (master.groupby("year")["pred"]
              .sum().reset_index().rename(columns={"pred": "total"}))

    base_total = float(annual.loc[annual["year"] == BASELINE, "total"].values[0]) \
                 if BASELINE in annual["year"].values else 1.0
    sel_total  = float(annual.loc[annual["year"] == fut_year, "total"].values[0]) \
                 if fut_year in annual["year"].values else 0.0
    demand_chg = (sel_total - base_total) / max(base_total, 0.001) * 100

    # Linear growth rate (slope / mean, expressed as %/yr)
    _yrs  = annual["year"].values.astype(float)
    _tots = annual["total"].values.astype(float)
    _slope = float(np.polyfit(_yrs, _tots, 1)[0])
    growth_rate = _slope / max(float(_tots.mean()), 0.001) * 100

    best_boro = (master[master["year"] == fut_year]
                 .groupby("Borough")["pred"].sum()
                 .idxmax() if not master.empty else "N/A")

    if   demand_chg > 15: trend, t_clr = t("future_surging"),  "#10B981"
    elif demand_chg >  5: trend, t_clr = t("future_growing"),  "#F7C948"
    elif demand_chg > -5: trend, t_clr = t("future_stable"),   "#9CA3AF"
    else:                 trend, t_clr = t("future_declining"), "#EF4444"

    d_sign  = "+" if demand_chg  >= 0 else ""
    d_clr   = "#10B981" if demand_chg  >= 0 else "#EF4444"
    gr_sign = "+" if growth_rate >= 0 else ""
    gr_clr  = "#10B981" if growth_rate >= 0 else "#EF4444"

    # ── KPI strip ─────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin:1rem 0 1.4rem">
      <div class="wc-kpi">
        <div class="wc-kpi-val" style="color:{d_clr}">{d_sign}{demand_chg:.1f}%</div>
        <div class="wc-kpi-lbl">{t("future_demand_chg")} · {fut_year} {t("future_vs")} {BASELINE}</div>
      </div>
      <div class="wc-kpi">
        <div class="wc-kpi-val" style="color:#F7C948;font-size:1.1rem">{sel_total:,.0f}</div>
        <div class="wc-kpi-lbl">{t("future_total_vol")} · {fut_year}</div>
      </div>
      <div class="wc-kpi">
        <div class="wc-kpi-val" style="color:{gr_clr}">{gr_sign}{growth_rate:.1f}%/yr</div>
        <div class="wc-kpi-lbl">{t("future_growth_rt")}</div>
      </div>
      <div class="wc-kpi">
        <div class="wc-kpi-val" style="color:#F7C948;font-size:1rem">{best_boro}</div>
        <div class="wc-kpi-lbl">{t("future_best_boro")} · {fut_year}</div>
      </div>
      <div class="wc-kpi">
        <div class="wc-kpi-val" style="color:{t_clr};font-size:.95rem">{trend}</div>
        <div class="wc-kpi-lbl">{t("future_trend")}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Growth trajectory (full width) ───────────────────────────────────────
    _section(t("future_trajectory"))
    _pchart(charts.future_trajectory(annual, fut_year, _now_year), h=330)

    # ── Monthly forecast | Zone growth ranking ────────────────────────────────
    col_m, col_z = st.columns(2)
    with col_m:
        _section(t("future_monthly", year=fut_year))
        _pchart(charts.future_monthly(monthly, fut_year), h=340)
    with col_z:
        _section(t("future_zones_hdr", year=fut_year, base=BASELINE))
        _base_z   = master[master["year"] == BASELINE ][["PULocationID","Zone","Borough","pred"]].copy()
        _target_z = master[master["year"] == fut_year ][["PULocationID","Zone","Borough","pred"]].copy()
        if not _base_z.empty and not _target_z.empty:
            _pchart(charts.future_zone_growth(_base_z, _target_z), h=340)

    # ── Borough growth trajectories (full width) ──────────────────────────────
    _section(t("future_boro_hdr"))
    _pchart(charts.future_borough_trend(master, _now_year), h=360)


# ─────────────────────────────────────────────────────────────────────────────
# Router
# ─────────────────────────────────────────────────────────────────────────────
_ROUTES = {
    "live":      page_live,
    "shift":     page_shift,
    "analytics": page_analytics,
    "model":     page_model,
    "intel":     page_intelligence,
    "future":    page_future,
}
_ROUTES[page_key]()
