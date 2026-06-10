"""
TaxiWise — Plotly Charts (dark-mode)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Colour palette ────────────────────────────────────────────────────────────
C = dict(
    gold   = "#F7C948",
    blue   = "#3B82F6",
    green  = "#10B981",
    red    = "#EF4444",
    purple = "#8B5CF6",
    orange = "#F97316",
    bg     = "#0E1117",
    card   = "#1A1D27",
    text   = "#FAFAFA",
    muted  = "#9CA3AF",
    grid   = "rgba(255,255,255,0.05)",
    zero   = "rgba(255,255,255,0.08)",
)

_LAYOUT = dict(
    paper_bgcolor = C["bg"],
    plot_bgcolor  = C["bg"],
    font          = dict(color=C["text"], family="Inter, sans-serif", size=12),
    margin        = dict(l=50, r=24, t=48, b=40),
    legend        = dict(
        bgcolor     = "rgba(26,29,39,0.9)",
        bordercolor = "rgba(255,255,255,0.06)",
        borderwidth = 1,
        font        = dict(size=11),
    ),
)

DAY_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _dark(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(height=height, **_LAYOUT)
    fig.update_xaxes(
        gridcolor=C["grid"], zerolinecolor=C["zero"],
        linecolor=C["grid"], tickfont=dict(color=C["muted"]),
    )
    fig.update_yaxes(
        gridcolor=C["grid"], zerolinecolor=C["zero"],
        linecolor=C["grid"], tickfont=dict(color=C["muted"]),
    )
    return fig


# ── Overview charts ───────────────────────────────────────────────────────────

def trips_by_hour(df: pd.DataFrame) -> go.Figure:
    hourly   = df.groupby("hour").size().reindex(range(24), fill_value=0)
    peak_h   = int(hourly.idxmax())
    colors   = [C["gold"] if h == peak_h else C["blue"] for h in range(24)]

    fig = go.Figure(go.Bar(
        x=list(range(24)), y=hourly.values,
        marker_color=colors, marker_line_width=0,
        hovertemplate="<b>%{x}:00</b><br>Trips: %{y:,}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="Trips by Hour of Day", font=dict(size=15)),
        xaxis=dict(title="Hour", tickmode="linear", dtick=2),
        yaxis=dict(title="Trips"),
    )
    return _dark(fig)


def trips_by_dow(df: pd.DataFrame) -> go.Figure:
    counts = df.groupby("dow").size().reindex(range(7), fill_value=0)
    norm   = counts / counts.max()

    fig = go.Figure(go.Bar(
        x=DAY_LABELS, y=counts.values,
        marker=dict(
            color=norm.values,
            colorscale=[[0, C["blue"]], [1, C["gold"]]],
            showscale=False,
        ),
        hovertemplate="<b>%{x}</b><br>Trips: %{y:,}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="Trips by Day of Week", font=dict(size=15)),
        xaxis=dict(title=""),
        yaxis=dict(title="Trips"),
    )
    return _dark(fig)


def monthly_trend(df: pd.DataFrame) -> go.Figure:
    mon_names = ["Jan","Feb","Mar","Apr","May","Jun",
                 "Jul","Aug","Sep","Oct","Nov","Dec"]
    monthly   = df.groupby("month").size().reindex(range(1, 13), fill_value=0)

    fig = go.Figure(go.Scatter(
        x=[mon_names[m - 1] for m in monthly.index],
        y=monthly.values,
        mode="lines+markers",
        line  = dict(color=C["purple"], width=2.5),
        marker= dict(size=7, color=C["purple"]),
        fill  = "tozeroy",
        fillcolor = "rgba(139,92,246,0.1)",
        hovertemplate="<b>%{x}</b><br>Trips: %{y:,}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="Monthly Trip Volume", font=dict(size=15)),
        xaxis=dict(title="Month"),
        yaxis=dict(title="Trips"),
    )
    return _dark(fig)


def top_zones(df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    col    = "pickup_zone" if "pickup_zone" in df.columns else "PULocationID"
    counts = df[col].value_counts().head(top_n).sort_values()
    norm   = counts / counts.max()

    fig = go.Figure(go.Bar(
        x=counts.values,
        y=counts.index.astype(str),
        orientation="h",
        marker=dict(
            color=norm.values,
            colorscale=[[0, C["blue"]], [1, C["gold"]]],
            showscale=False,
        ),
        hovertemplate="<b>%{y}</b><br>Trips: %{x:,}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=f"Top {top_n} Pickup Zones", font=dict(size=15)),
        xaxis=dict(title="Trips"),
        yaxis=dict(title=""),
    )
    return _dark(fig, height=420)


# ── Analytics charts ──────────────────────────────────────────────────────────

def demand_heatmap(df: pd.DataFrame) -> go.Figure:
    pivot = (
        df.groupby(["hour", "dow"]).size()
        .unstack(fill_value=0)
        .reindex(index=range(24), columns=range(7), fill_value=0)
    )
    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=DAY_LABELS,
        y=[f"{h:02d}:00" for h in range(24)],
        colorscale="YlOrRd",
        hovertemplate="<b>%{x}, %{y}</b><br>Trips: %{z:,}<extra></extra>",
        colorbar=dict(title=dict(text="Trips", side="right"), thickness=14),
    ))
    fig.update_layout(
        title=dict(text="Demand Heatmap: Hour × Day of Week", font=dict(size=15)),
    )
    return _dark(fig, height=520)


def borough_flow(df: pd.DataFrame) -> go.Figure:
    if "pickup_borough" not in df.columns:
        return go.Figure()
    flow = (
        df.groupby(["pickup_borough", "dropoff_borough"]).size()
        .unstack(fill_value=0)
    )
    b    = sorted(set(list(flow.index) + list(flow.columns)))
    flow = flow.reindex(index=b, columns=b, fill_value=0)

    fig = go.Figure(go.Heatmap(
        z=flow.values, x=list(flow.columns), y=list(flow.index),
        colorscale="Blues",
        text=flow.values, texttemplate="%{text:,}",
        hovertemplate="<b>%{y} → %{x}</b><br>Trips: %{z:,}<extra></extra>",
        colorbar=dict(title=dict(text="Trips", side="right"), thickness=14),
    ))
    fig.update_layout(
        title=dict(text="Trip Flow Between Boroughs", font=dict(size=15)),
    )
    return _dark(fig, height=420)


def fare_dist(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Base Fare ($)", "Total Amount ($)"],
        horizontal_spacing=0.08,
    )
    for col_i, (col, color) in enumerate(
        [("fare_amount", C["blue"]), ("total_amount", C["gold"])], 1
    ):
        s   = df[(df[col] > 0) & (df[col] < df[col].quantile(0.99))][col]
        med = s.median()
        fig.add_histogram(
            x=s, nbinsx=60,
            marker_color=color, opacity=0.75,
            row=1, col=col_i,
            hovertemplate="$%{x:.1f}<br>Count: %{y:,}<extra></extra>",
        )
        fig.add_vline(
            x=med, line_dash="dash", line_color="white",
            annotation_text=f"Median: ${med:.1f}",
            annotation_font_color=C["text"],
            row=1, col=col_i,
        )
    fig.update_layout(
        title=dict(text="Fare Distribution", font=dict(size=15)),
        showlegend=False,
    )
    return _dark(fig)


def scatter_dist_fare(df: pd.DataFrame) -> go.Figure:
    q_d  = df["trip_distance"].quantile(0.99)
    q_f  = df["fare_amount"].quantile(0.99)
    samp = df[
        (df["trip_distance"] > 0) & (df["trip_distance"] < q_d) &
        (df["fare_amount"]   > 0) & (df["fare_amount"]   < q_f)
    ].sample(min(3000, len(df)), random_state=42)

    color_col = "pickup_borough" if "pickup_borough" in samp.columns else "hour"
    fig = px.scatter(
        samp, x="trip_distance", y="fare_amount", color=color_col,
        opacity=0.45,
        labels={"trip_distance": "Distance (miles)", "fare_amount": "Fare ($)"},
        color_discrete_sequence=px.colors.qualitative.Bold,
    )
    fig.update_traces(marker=dict(size=4))
    fig.update_layout(
        title=dict(text="Distance vs Fare by Borough", font=dict(size=15)),
    )
    return _dark(fig)


def speed_by_hour(df: pd.DataFrame) -> go.Figure:
    s = df.copy()
    s["speed"] = s["trip_distance"] / (s["trip_duration_min"] / 60)
    s = s[(s["speed"] > 1) & (s["speed"] < 80)]
    hourly = s.groupby("hour")["speed"].mean().reindex(range(24)).interpolate()

    fig = go.Figure(go.Scatter(
        x=list(range(24)), y=hourly.round(1),
        mode="lines+markers",
        line  = dict(color=C["green"], width=2.5),
        marker= dict(size=5, color=C["green"]),
        fill  = "tozeroy",
        fillcolor = "rgba(16,185,129,0.08)",
        hovertemplate="<b>%{x}:00</b><br>Avg Speed: %{y:.1f} mph<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="Average Travel Speed by Hour", font=dict(size=15)),
        xaxis=dict(title="Hour", tickmode="linear", dtick=2),
        yaxis=dict(title="Speed (mph)"),
    )
    return _dark(fig)


def tip_by_hour(df: pd.DataFrame) -> go.Figure:
    cc = df[(df["payment_type"] == 1) & (df["fare_amount"] > 0)].copy()
    cc["tip_pct"] = cc["tip_amount"] / cc["fare_amount"] * 100
    tip_hr = cc.groupby("hour")["tip_pct"].mean().reindex(range(24)).interpolate()

    fig = go.Figure(go.Scatter(
        x=list(range(24)), y=tip_hr.round(1),
        mode="lines+markers",
        line  = dict(color=C["gold"], width=2.5),
        marker= dict(size=5, color=C["gold"]),
        fill  = "tozeroy",
        fillcolor = "rgba(247,201,72,0.08)",
        hovertemplate="<b>%{x}:00</b><br>Avg Tip: %{y:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="Avg Tip % by Hour (Credit Card Only)", font=dict(size=15)),
        xaxis=dict(title="Hour", tickmode="linear", dtick=2),
        yaxis=dict(title="Tip (%)"),
    )
    return _dark(fig)


# ── ML / Model charts ─────────────────────────────────────────────────────────

def feature_importance(fi: pd.DataFrame) -> go.Figure:
    fi   = fi.sort_values("importance")
    norm = fi["importance"] / fi["importance"].max()

    fig = go.Figure(go.Bar(
        x=fi["importance"],
        y=fi["label"],
        orientation="h",
        marker=dict(
            color=norm.values,
            colorscale=[[0, C["blue"]], [0.5, C["purple"]], [1, C["gold"]]],
            showscale=False,
        ),
        hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="Feature Importance (XGBoost)", font=dict(size=15)),
        xaxis=dict(title="Importance Score"),
        yaxis=dict(title=""),
    )
    return _dark(fig)


def actual_vs_predicted(y_true: np.ndarray, y_pred: np.ndarray) -> go.Figure:
    lim = float(max(y_true.max(), y_pred.max())) * 1.05

    fig = go.Figure()
    fig.add_scatter(
        x=y_true, y=y_pred,
        mode="markers", name="Predictions",
        marker=dict(color=C["blue"], opacity=0.35, size=4),
        hovertemplate="Actual: %{x:.0f}<br>Predicted: %{y:.0f}<extra></extra>",
    )
    fig.add_scatter(
        x=[0, lim], y=[0, lim],
        mode="lines", name="Perfect Fit",
        line=dict(color=C["gold"], dash="dash", width=2),
    )
    fig.update_layout(
        title=dict(text="Actual vs Predicted Demand", font=dict(size=15)),
        xaxis=dict(title="Actual Trips",    range=[0, lim]),
        yaxis=dict(title="Predicted Trips", range=[0, lim]),
    )
    return _dark(fig)


# ── Year-comparison charts ────────────────────────────────────────────────────

YEAR_COLORS = {2023: C["blue"], 2024: C["gold"], 2025: C["green"]}
YEAR_FILL   = {
    2023: "rgba(59,130,246,0.09)",
    2024: "rgba(247,201,72,0.09)",
    2025: "rgba(16,185,129,0.09)",
}


def yearly_trip_comparison(df: pd.DataFrame) -> go.Figure:
    yearly = df.groupby("year").size().reset_index(name="trips")
    fig = go.Figure(go.Bar(
        x=yearly["year"].astype(str),
        y=yearly["trips"],
        marker_color=[YEAR_COLORS.get(y, C["blue"]) for y in yearly["year"]],
        text=yearly["trips"].apply(lambda x: f"{x:,}"),
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Trips: %{y:,}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="Total Trips by Year", font=dict(size=15)),
        xaxis=dict(title="Year"),
        yaxis=dict(title="Trips"),
    )
    return _dark(fig)


def yearly_monthly_trend(df: pd.DataFrame) -> go.Figure:
    mon_names = ["Jan","Feb","Mar","Apr","May","Jun",
                 "Jul","Aug","Sep","Oct","Nov","Dec"]
    fig = go.Figure()
    for year in sorted(df["year"].unique()):
        ydf = df[df["year"] == year]
        monthly = ydf.groupby("month").size().reindex(range(1, 13), fill_value=0)
        color = YEAR_COLORS.get(year, C["blue"])
        fig.add_scatter(
            x=[mon_names[m - 1] for m in monthly.index],
            y=monthly.values,
            name=str(year),
            mode="lines+markers",
            line=dict(color=color, width=2.5),
            marker=dict(size=6, color=color),
            hovertemplate=f"<b>{year}</b> %{{x}}<br>Trips: %{{y:,}}<extra></extra>",
        )
    fig.update_layout(
        title=dict(text="Monthly Demand Trends by Year", font=dict(size=15)),
        xaxis=dict(title="Month"),
        yaxis=dict(title="Trips"),
    )
    return _dark(fig)


def yearly_peak_hours(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for year in sorted(df["year"].unique()):
        ydf = df[df["year"] == year]
        hourly = ydf.groupby("hour").size().reindex(range(24), fill_value=0)
        color = YEAR_COLORS.get(year, C["blue"])
        fill  = YEAR_FILL.get(year, "rgba(59,130,246,0.09)")
        fig.add_scatter(
            x=list(range(24)),
            y=hourly.values,
            name=str(year),
            mode="lines+markers",
            line=dict(color=color, width=2.5),
            marker=dict(size=5, color=color),
            fill="tozeroy",
            fillcolor=fill,
            hovertemplate=f"<b>{year}</b> %{{x}}:00<br>Trips: %{{y:,}}<extra></extra>",
        )
    fig.update_layout(
        title=dict(text="Peak Hour Comparison by Year", font=dict(size=15)),
        xaxis=dict(title="Hour", tickmode="linear", dtick=2),
        yaxis=dict(title="Trips"),
    )
    return _dark(fig)


def yearly_top_zones(df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    col = "pickup_zone" if "pickup_zone" in df.columns else "PULocationID"
    top_zones = df[col].value_counts().head(top_n).index.tolist()
    fig = go.Figure()
    for year in sorted(df["year"].unique()):
        ydf = df[df["year"] == year]
        counts = ydf[col].value_counts()
        fig.add_bar(
            x=[str(z)[:22] for z in top_zones],
            y=[counts.get(z, 0) for z in top_zones],
            name=str(year),
            marker_color=YEAR_COLORS.get(year, C["blue"]),
            hovertemplate=f"<b>{year}</b><br>%{{x}}<br>Trips: %{{y:,}}<extra></extra>",
        )
    fig.update_layout(
        title=dict(text=f"Top {top_n} Zones — Year Comparison", font=dict(size=15)),
        xaxis=dict(title="", tickangle=-30),
        yaxis=dict(title="Trips"),
        barmode="group",
    )
    return _dark(fig, height=430)


def yearly_fare_trend(df: pd.DataFrame) -> go.Figure:
    mon_names = ["Jan","Feb","Mar","Apr","May","Jun",
                 "Jul","Aug","Sep","Oct","Nov","Dec"]
    fig = go.Figure()
    for year in sorted(df["year"].unique()):
        ydf = df[df["year"] == year]
        fare_m = (
            ydf.groupby("month")["fare_amount"]
            .mean()
            .reindex(range(1, 13))
        )
        color = YEAR_COLORS.get(year, C["blue"])
        fig.add_scatter(
            x=[mon_names[m - 1] for m in fare_m.index],
            y=fare_m.round(2),
            name=str(year),
            mode="lines+markers",
            line=dict(color=color, width=2.5),
            marker=dict(size=6, color=color),
            hovertemplate=f"<b>{year}</b> %{{x}}<br>Avg Fare: $%{{y:.2f}}<extra></extra>",
        )
    fig.update_layout(
        title=dict(text="Average Fare Trend by Year", font=dict(size=15)),
        xaxis=dict(title="Month"),
        yaxis=dict(title="Avg Fare ($)"),
    )
    return _dark(fig)


# ── Hot zones ─────────────────────────────────────────────────────────────────

def hot_zones_chart(hot_df: pd.DataFrame) -> go.Figure:
    if hot_df.empty:
        return go.Figure()

    hot = hot_df.copy()
    hot["label"] = hot.apply(
        lambda r: f"{r.get('Zone', 'Zone ' + str(r['PULocationID']))} "
                  f"({r.get('Borough', '')})",
        axis=1,
    )

    fig = go.Figure(go.Bar(
        x=hot["predicted_demand"].round(1),
        y=hot["label"],
        orientation="h",
        marker=dict(
            color=hot["predicted_demand"],
            colorscale="YlOrRd",
            showscale=True,
            colorbar=dict(title=dict(text="Demand", side="right"), thickness=14),
        ),
        hovertemplate=(
            "<b>%{y}</b><br>Predicted demand: %{x:.1f} trips/hr<extra></extra>"
        ),
    ))
    fig.update_layout(
        title=dict(text="Predicted Hot Zones", font=dict(size=15)),
        xaxis=dict(title="Predicted Trips / Hour"),
        yaxis=dict(autorange="reversed"),
    )
    return _dark(fig, height=460)


# ── Borough comparison ────────────────────────────────────────────────────────

BORO_COLORS = {
    "Manhattan":     C["gold"],
    "Brooklyn":      C["blue"],
    "Queens":        C["green"],
    "Bronx":         C["purple"],
    "Staten Island": C["orange"],
    "EWR":           C["muted"],
}
BORO_FILL = {
    "Manhattan":     "rgba(247,201,72,0.13)",
    "Brooklyn":      "rgba(59,130,246,0.13)",
    "Queens":        "rgba(16,185,129,0.13)",
    "Bronx":         "rgba(139,92,246,0.13)",
    "Staten Island": "rgba(249,115,22,0.13)",
}
_MAIN_BOROS = ["Manhattan", "Brooklyn", "Queens", "Bronx"]


def borough_hourly(df: pd.DataFrame) -> go.Figure:
    """Multi-line hourly demand curve, one line per main borough."""
    if "pickup_borough" not in df.columns:
        return go.Figure()
    fig = go.Figure()
    for boro in _MAIN_BOROS:
        sub = df[df["pickup_borough"] == boro]
        if sub.empty:
            continue
        hourly = sub.groupby("hour").size().reindex(range(24), fill_value=0)
        color  = BORO_COLORS[boro]
        fig.add_scatter(
            x=list(range(24)), y=hourly.values, name=boro,
            mode="lines+markers",
            line=dict(color=color, width=2.5),
            marker=dict(size=5, color=color),
            hovertemplate=f"<b>{boro}</b> %{{x}}:00<br>Trips: %{{y:,}}<extra></extra>",
        )
    fig.update_layout(
        title=dict(text="Hourly Demand by Borough", font=dict(size=15)),
        xaxis=dict(title="Hour", tickmode="linear", dtick=2),
        yaxis=dict(title="Trips"),
    )
    return _dark(fig)


def borough_dow(df: pd.DataFrame) -> go.Figure:
    """Multi-line day-of-week demand, one line per main borough."""
    if "pickup_borough" not in df.columns:
        return go.Figure()
    fig = go.Figure()
    for boro in _MAIN_BOROS:
        sub = df[df["pickup_borough"] == boro]
        if sub.empty:
            continue
        counts = sub.groupby("dow").size().reindex(range(7), fill_value=0)
        color  = BORO_COLORS[boro]
        fig.add_scatter(
            x=DAY_LABELS, y=counts.values, name=boro,
            mode="lines+markers",
            line=dict(color=color, width=2.5),
            marker=dict(size=6, color=color),
            hovertemplate=f"<b>{boro}</b> %{{x}}<br>Trips: %{{y:,}}<extra></extra>",
        )
    fig.update_layout(
        title=dict(text="Day-of-Week Demand by Borough", font=dict(size=15)),
        xaxis=dict(title=""),
        yaxis=dict(title="Trips"),
    )
    return _dark(fig)


def borough_revenue(df: pd.DataFrame) -> go.Figure:
    """Stacked bar: avg base fare + avg tip per borough (revenue per trip)."""
    if "pickup_borough" not in df.columns:
        return go.Figure()
    boros, fares, tips = [], [], []
    for boro in _MAIN_BOROS:
        sub = df[df["pickup_borough"] == boro]
        if sub.empty:
            continue
        boros.append(boro)
        fares.append(float(sub["fare_amount"].mean()))
        tips.append(float(sub["tip_amount"].mean()) if "tip_amount" in sub.columns else 0.0)
    if not boros:
        return go.Figure()
    fig = go.Figure()
    fig.add_bar(
        x=boros, y=fares, name="Avg Base Fare",
        marker_color=[BORO_COLORS[b] for b in boros],
        hovertemplate="<b>%{x}</b><br>Base Fare: $%{y:.2f}<extra></extra>",
    )
    fig.add_bar(
        x=boros, y=tips, name="Avg Tip",
        marker=dict(color=C["gold"], opacity=0.60),
        hovertemplate="<b>%{x}</b><br>Tip: $%{y:.2f}<extra></extra>",
    )
    fig.update_layout(
        title=dict(text="Revenue per Trip by Borough", font=dict(size=15)),
        barmode="stack",
        xaxis=dict(title=""),
        yaxis=dict(title="Avg Amount ($)"),
    )
    return _dark(fig)


def borough_radar(df: pd.DataFrame) -> go.Figure:
    """Radar chart — 4 boroughs × 5 normalized metrics."""
    if "pickup_borough" not in df.columns:
        return go.Figure()
    cats = ["Volume", "Avg Fare", "Avg Distance", "Avg Duration", "Avg Tip"]
    raw: dict[str, np.ndarray] = {}
    for boro in _MAIN_BOROS:
        sub = df[df["pickup_borough"] == boro]
        if sub.empty:
            continue
        raw[boro] = np.array([
            float(len(sub)),
            float(sub["fare_amount"].mean()),
            float(sub["trip_distance"].mean()),
            float(sub["trip_duration_min"].mean()),
            float(sub["tip_amount"].mean()) if "tip_amount" in sub.columns else 0.0,
        ])
    if len(raw) < 2:
        return go.Figure()
    arr   = np.stack(list(raw.values()))
    denom = np.where(arr.max(axis=0) - arr.min(axis=0) > 0,
                     arr.max(axis=0) - arr.min(axis=0), 1.0)
    norm  = (arr - arr.min(axis=0)) / denom
    fig = go.Figure()
    for (boro, _), nvals in zip(raw.items(), norm):
        closed = list(nvals) + [nvals[0]]
        color  = BORO_COLORS[boro]
        fill   = BORO_FILL.get(boro, "rgba(255,255,255,0.08)")
        fig.add_scatterpolar(
            r=closed, theta=cats + [cats[0]],
            fill="toself", name=boro,
            line=dict(color=color, width=2.2),
            fillcolor=fill,
            hovertemplate=f"<b>{boro}</b> — %{{theta}}: %{{r:.2f}}<extra></extra>",
        )
    fig = _dark(fig)
    fig.update_layout(
        title=dict(text="Multi-Metric Borough Comparison (Normalized)", font=dict(size=15)),
        polar=dict(
            bgcolor=C["bg"],
            radialaxis=dict(
                visible=True, range=[0, 1], showticklabels=False,
                gridcolor=C["grid"], linecolor=C["grid"],
            ),
            angularaxis=dict(
                tickfont=dict(color=C["text"], size=11),
                gridcolor=C["grid"], linecolor=C["grid"],
            ),
        ),
    )
    return fig
