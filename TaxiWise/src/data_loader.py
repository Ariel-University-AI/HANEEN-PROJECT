"""
TaxiWise — Data Loading & Feature Engineering
Priority per year:
  1. PARQUET files in data/raw/ matching *{year}*.parquet
  2. yellow_taxi_{year}_small_merged.csv  (per-year enriched CSV)
  3. yellow_taxi_{year}.csv               (plain per-year CSV)
  4. Minimal synthetic fallback (20k rows) — run prepare_data.py to avoid this
"""

from pathlib import Path

import pandas as pd
import streamlit as st

ROOT     = Path(__file__).parent.parent
ZONE_CSV = ROOT / "data" / "taxi_zone_lookup.csv"
RAW_DIR  = ROOT / "data" / "raw"
YEARS    = [2023, 2024, 2025, 2026]

PAYMENT_LABELS = {
    1: "Credit Card", 2: "Cash", 3: "No Charge",
    4: "Dispute",     5: "Unknown", 6: "Voided",
}
DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

# Columns we never use — skip when reading CSV to cut parse time and memory
_CSV_DROP = frozenset({
    "data_month", "extra", "mta_tax", "tolls_amount",
    "improvement_surcharge", "congestion_surcharge",
    "Airport_fee", "cbd_congestion_fee",
    "pickup_service_zone", "dropoff_service_zone",
    "store_and_fwd_flag",
})

_zone_cache: pd.DataFrame | None = None


def _zone_lookup() -> pd.DataFrame | None:
    global _zone_cache
    if _zone_cache is None and ZONE_CSV.exists():
        _zone_cache = pd.read_csv(ZONE_CSV)
    return _zone_cache


def _optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Convert float64 → float32 to halve memory for large numeric columns."""
    for col in df.select_dtypes("float64").columns:
        df[col] = df[col].astype("float32")
    return df


@st.cache_data(show_spinner=False)
def load_trips() -> pd.DataFrame:
    """Load taxi trips for all years. Returns empty DataFrame on total failure."""
    frames = []

    for year in YEARS:
        df = _load_year(year)
        if df is not None and not df.empty:
            frames.append(df)

    if not frames:
        st.error(
            "⚠️ לא ניתן לטעון נתונים. "
            "הרץ `python prepare_data.py` כדי לייצר את קבצי הנתונים."
        )
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _load_year(year: int) -> "pd.DataFrame | None":
    parquet_files  = sorted(RAW_DIR.glob(f"*{year}*.parquet"))
    merged_csv     = ROOT / f"yellow_taxi_{year}_small_merged.csv"
    root_csv       = ROOT / f"yellow_taxi_{year}.csv"

    try:
        if parquet_files:
            df = pd.concat(
                [pd.read_parquet(f) for f in parquet_files],
                ignore_index=True,
            )
            df = _ensure_duration(df)

        elif merged_csv.exists():
            df = pd.read_csv(
                merged_csv,
                usecols=lambda c: c not in _CSV_DROP,
                parse_dates=["tpep_pickup_datetime", "tpep_dropoff_datetime"],
            )
            df = _ensure_duration(df)

        elif root_csv.exists():
            df = pd.read_csv(
                root_csv,
                usecols=lambda c: c not in _CSV_DROP,
                parse_dates=["tpep_pickup_datetime", "tpep_dropoff_datetime"],
            )
            df = _ensure_duration(df)

        else:
            # Last-resort fallback — run prepare_data.py to eliminate this path
            from src.utils import generate_synthetic_data
            df = generate_synthetic_data(n_rows=20_000, seed=year, years=[year])
            df = _ensure_duration(df)

        df = _optimize_dtypes(df)
        df["year"] = year
        return _enrich(df)

    except Exception as exc:
        st.warning(f"⚠️ שגיאה בטעינת נתוני {year}: {exc}")
        return None


def _ensure_duration(df: pd.DataFrame) -> pd.DataFrame:
    if "trip_duration_min" not in df.columns:
        for col in ["tpep_pickup_datetime", "tpep_dropoff_datetime"]:
            df[col] = pd.to_datetime(df[col], errors="coerce")
        df["trip_duration_min"] = (
            (df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"])
            .dt.total_seconds() / 60
        )
    return df


def _enrich(df: pd.DataFrame) -> pd.DataFrame:
    df["tpep_pickup_datetime"] = pd.to_datetime(
        df["tpep_pickup_datetime"], errors="coerce"
    )
    dt = df["tpep_pickup_datetime"]
    df["hour"]          = dt.dt.hour
    df["dow"]           = dt.dt.dayofweek
    df["month"]         = dt.dt.month
    df["date"]          = dt.dt.date
    df["day_name"]      = df["dow"].map(dict(enumerate(DAY_NAMES)))
    df["payment_label"] = df["payment_type"].map(PAYMENT_LABELS).fillna("Unknown")

    z = _zone_lookup()
    if z is not None:
        if "pickup_borough" not in df.columns:
            pu = z.rename(columns={
                "LocationID": "PULocationID",
                "Borough":    "pickup_borough",
                "Zone":       "pickup_zone",
            })[["PULocationID", "pickup_borough", "pickup_zone"]]
            df = df.merge(pu, on="PULocationID", how="left")
        if "dropoff_borough" not in df.columns:
            do = z.rename(columns={
                "LocationID": "DOLocationID",
                "Borough":    "dropoff_borough",
                "Zone":       "dropoff_zone",
            })[["DOLocationID", "dropoff_borough", "dropoff_zone"]]
            df = df.merge(do, on="DOLocationID", how="left")

    df = df[
        (df["fare_amount"]       > 0)   & (df["fare_amount"]       < 500) &
        (df["trip_distance"]     >= 0)  & (df["trip_distance"]     < 200) &
        (df["trip_duration_min"] > 0)   & (df["trip_duration_min"] < 600)
    ].copy()
    return df


@st.cache_data(show_spinner=False)
def load_zones() -> pd.DataFrame:
    return pd.read_csv(ZONE_CSV)


@st.cache_data(show_spinner=False)
def compute_demand() -> pd.DataFrame:
    """Aggregate trip stats per (zone, hour, dow, month) — used by ML model."""
    df = load_trips()
    if df.empty:
        return pd.DataFrame()
    agg = (
        df.groupby(["PULocationID", "hour", "dow", "month"])
        .agg(
            trip_count   =("fare_amount",       "count"),
            avg_fare     =("fare_amount",       "mean"),
            avg_distance =("trip_distance",     "mean"),
            avg_duration =("trip_duration_min", "mean"),
            avg_tip      =("tip_amount",        "mean"),
        )
        .reset_index()
    )
    zone_totals = df.groupby("PULocationID").size().rename("zone_total_trips")
    return agg.merge(zone_totals, on="PULocationID", how="left")


def compute_kpis(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "total_trips": 0, "avg_fare": 0.0, "avg_distance": 0.0,
            "avg_duration": 0.0, "top_zone": "N/A", "peak_hour": 0,
            "active_zones": 0, "total_revenue": 0.0, "credit_pct": 0.0,
        }
    clean = df[(df["total_amount"] > 0) & (df["total_amount"] < 500)]
    top_zone = (
        df["pickup_zone"].value_counts().idxmax()
        if "pickup_zone" in df.columns else "N/A"
    )
    return {
        "total_trips":   len(df),
        "avg_fare":      float(clean["total_amount"].mean()) if len(clean) > 0 else 0.0,
        "avg_distance":  float(df["trip_distance"].mean()),
        "avg_duration":  float(df["trip_duration_min"].mean()),
        "top_zone":      top_zone,
        "peak_hour":     int(df["hour"].value_counts().idxmax()),
        "active_zones":  int(df["PULocationID"].nunique()),
        "total_revenue": float(clean["total_amount"].sum()) if len(clean) > 0 else 0.0,
        "credit_pct":    float((df["payment_type"] == 1).mean() * 100),
    }


@st.cache_data(show_spinner=False)
def get_kpis() -> dict:
    return compute_kpis(load_trips())
