#!/usr/bin/env python3
"""
eda.py

Exploratory data analysis for RMS crime incidents mapped to CVI areas.

Inputs
------
- rms_data_cleaned_mapped.csv (from cvi_mapping.py), which must contain:
    - incident_occurred_at (timestamp-like string)
    - longitude, latitude
    - cvi_area (CVI area name or "Non-CVI")
    - buffer (boolean)

Parameters
----------
--input         path to RMS CSV (default: rms_data_cleaned_mapped.csv)
--out-dir       directory for all outputs (default: eda_outputs)
--program-start program start date (default: 2023-08-01)
--cvi-file      CVI polygons (GeoJSON, same as cvi_mapping.py)
--name-col      column name in CVI file with area names (default: CVI_AREA_NAME)

Outputs (in out-dir)
--------------------
- monthly_city.html
- monthly_cvi_vs_non.html
- adf_results.csv
- seasonal_city.png, seasonal_<area>.png ...
- acf_city.png, pacf_city.png, acf_<area>.png, pacf_<area>.png ...
- heatmap_time.html
"""

import argparse
from pathlib import Path
import sys
from typing import Tuple

import numpy as np
import pandas as pd
import geopandas as gpd

import plotly.express as px
import plotly.graph_objects as go

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

import matplotlib.pyplot as plt

import folium
from folium.plugins import HeatMapWithTime

WGS84 = "EPSG:4326"

def load_rms(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "incident_occurred_at" not in df.columns:
        raise KeyError("Input CSV must contain 'incident_occurred_at' column.")
    df["incident_occurred_at"] = pd.to_datetime(df["incident_occurred_at"], errors="coerce")
    df = df.dropna(subset=["incident_occurred_at"]).copy()

    # Basic temporal fields
    df["month"] = df["incident_occurred_at"].dt.to_period("M").dt.to_timestamp()
    df["hour"] = df["incident_occurred_at"].dt.hour
    df["day_of_week"] = df["incident_occurred_at"].dt.day_name()

    # Geography column (prefer cvi_zone if present)
    geo_col = "cvi_zone" if "cvi_zone" in df.columns else "cvi_area"
    if geo_col not in df.columns:
        raise KeyError("Input CSV must contain 'cvi_zone' or 'cvi_area' from cvi_mapping.")

    df["geo_area"] = df[geo_col].fillna("Non-CVI")

    # Zone type: CVI vs Non-CVI
    df["zone_type"] = np.where(df["geo_area"] == "Non-CVI", "Non-CVI", "CVI")

    return df


def ensure_out_dir(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# 1. Monthly counts plots
# ---------------------------------------------------------------------

def plot_monthly_counts_city(df: pd.DataFrame, out_html: Path, program_start: pd.Timestamp) -> None:
    # Convert 'incident_occurred_at' column to datetime
    df['incident_occurred_at'] = pd.to_datetime(df['incident_occurred_at'], errors='coerce')

    # Create a new column for month-year
    df['incident_month'] = df['incident_occurred_at'].dt.to_period('M').dt.to_timestamp()

    # Group the dataset by incident_month and count incidents per month
    monthly_counts = df.groupby('incident_month').size().reset_index(name='incident_count')

    # Sort by date to ensure proper chronological order
    monthly_counts = monthly_counts.sort_values('incident_month')

    # Create an interactive plotly line chart 
    fig = px.line(
        monthly_counts, 
        x='incident_month', 
        y='incident_count',
        markers=True,
        title='Monthly Incident Count Over Time',
        labels={'incident_month': 'Month', 'incident_count': 'Number of Incidents'},
        template='plotly_white'
    )

    # Customize hover information
    fig.update_traces(
        hovertemplate='<b>Date</b>: %{x|%B %Y}<br><b>Incidents</b>: %{y}<extra></extra>'
    )

    # Add a red vertical line at August 2023
    aug2023 = pd.to_datetime('2023-08-01')
    fig.add_vline(x=aug2023, line_width=2, line_dash='dash', line_color='red')

    # Add annotation above the red vertical line
    fig.add_annotation(x=aug2023, y=1.01, xref="x", yref="paper", text='ShotStopper', showarrow=False, font=dict(color='red', size=12), yanchor='bottom')

    # Improve layout
    fig.update_layout(
        xaxis_title='Month',
        yaxis_title='Incident Count',
        hovermode='closest'
    )

    fig.write_html(out_html.as_posix())
    print(f"Saved city-wide monthly plot to {out_html}")

def plot_monthly_counts_cvi_vs_non(
    df: pd.DataFrame,
    out_html_non: Path,
    out_html_cvi: Path,
    program_start: pd.Timestamp,
) -> None:
    # Ensure datetime + month
    df["incident_occurred_at"] = pd.to_datetime(df["incident_occurred_at"], errors="coerce")
    df["incident_month"] = df["incident_occurred_at"].dt.to_period("M").dt.to_timestamp()

    # ---------- (A) Non-CVI only ----------
    non = df[df["geo_area"] == "Non-CVI"].copy()
    non_monthly = (
        non.groupby("incident_month")
           .size()
           .reset_index(name="incident_count")
           .sort_values("incident_month")
    )

    fig_non = px.line(
        non_monthly,
        x="incident_month",
        y="incident_count",
        markers=True,
        title="Monthly Incident Count Over Time (Non-CVI Only)",
        labels={"incident_month": "Month", "incident_count": "Incident Count"},
        template="plotly_white",
    )
    fig_non.update_traces(
        hovertemplate='<b>Date</b>: %{x|%B %Y}<br><b>Incidents</b>: %{y}<extra></extra>'
    )
    fig_non.add_vline(x=program_start, line_width=2, line_dash="dash", line_color="red")
    fig_non.add_annotation(
        x=program_start, y=1.01, xref="x", yref="paper",
        text="ShotStopper", showarrow=False,
        font=dict(color="red", size=12), yanchor="bottom"
    )
    fig_non.update_layout(hovermode="closest")
    fig_non.write_html(out_html_non.as_posix())
    print(f"Saved Non-CVI monthly plot to {out_html_non}")

    # ---------- (B) CVI geographies only (FACETS) ----------
    cvi = df[df["geo_area"] != "Non-CVI"].copy()
    cvi_monthly = (
        cvi.groupby(["incident_month", "geo_area"])
           .size()
           .reset_index(name="incident_count")
           .sort_values(["geo_area", "incident_month"])
    )

    # Facet settings
    n_cols = 3  # adjust to taste
    fig_cvi = px.line(
        cvi_monthly,
        x="incident_month",
        y="incident_count",
        facet_col="geo_area",
        facet_col_wrap=n_cols,
        markers=True,
        title="Monthly Incident Count Over Time (CVI Areas Only)",
        labels={"incident_month": "Month", "incident_count": "Incident Count"},
        template="plotly_white",
    )

    # Hover template (works per-trace)
    fig_cvi.update_traces(
        hovertemplate='<b>Date</b>: %{x|%B %Y}<br><b>Incidents</b>: %{y}<extra></extra>'
    )

    # Add ShotStopper line + label to every facet
    # We add a vertical line once; Plotly applies it across facets
    fig_cvi.add_vline(x=program_start, line_width=2, line_dash="dash", line_color="red")

    # Annotation: place once at top; line is the key visual cue anyway
    fig_cvi.add_annotation(
        x=program_start, y=1.03, xref="x", yref="paper",
        text="ShotStopper", showarrow=False,
        font=dict(color="red", size=12), yanchor="bottom"
    )

    # Make facets readable
    fig_cvi.update_xaxes(matches=None)  # allows independent x-axis labels per facet
    fig_cvi.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))  # cleaner facet titles

    # More breathing room
    # (height scales with number of facets)
    n_facets = cvi_monthly["geo_area"].nunique()
    n_rows = (n_facets + n_cols - 1) // n_cols
    fig_cvi.update_layout(height=320 * n_rows, hovermode="closest")

    fig_cvi.write_html(out_html_cvi.as_posix())
    print(f"Saved CVI-only faceted monthly plot to {out_html_cvi}")

# ---------------------------------------------------------------------
# 2. ADF tests
# ---------------------------------------------------------------------

def run_adf(series: pd.Series) -> Tuple[float, float, dict, bool]:
    series = series.dropna()
    if len(series) < 12:  # need enough points
        return np.nan, np.nan, {}, False
    result = adfuller(series)
    stat = result[0]
    pval = result[1]
    crit_vals = result[4]
    stationary = bool(pval < 0.05)
    return stat, pval, crit_vals, stationary


def run_adf_all(df: pd.DataFrame, out_csv: Path) -> None:
    records = []

    # city-wide
    city_monthly = df.groupby("month").size()
    stat, pval, crit, stationary = run_adf(city_monthly)
    records.append({
        "series": "City-wide",
        "level": "city",
        "adf_stat": stat,
        "p_value": pval,
        "crit_1pct": crit.get("1%", np.nan),
        "crit_5pct": crit.get("5%", np.nan),
        "crit_10pct": crit.get("10%", np.nan),
        "stationary": stationary,
    })

    # per CVI area (including Non-CVI)
    for area, sub in df.groupby("geo_area"):
        monthly = sub.groupby("month").size()
        stat, pval, crit, stationary = run_adf(monthly)
        records.append({
            "series": area,
            "level": "cvi_area",
            "adf_stat": stat,
            "p_value": pval,
            "crit_1pct": crit.get("1%", np.nan),
            "crit_5pct": crit.get("5%", np.nan),
            "crit_10pct": crit.get("10%", np.nan),
            "stationary": stationary,
        })

    res = pd.DataFrame.from_records(records)
    res.to_csv(out_csv, index=False)
    print("\nAugmented Dickey-Fuller Test for Stationarity:")
    print(res)
    print(f"\nSaved ADF results to {out_csv}")

# ---------------------------------------------------------------------
# 3. Seasonal decomposition
# ---------------------------------------------------------------------

def seasonal_decomp_plots(df: pd.DataFrame, out_dir: Path) -> None:
    """Create seasonal decomposition PNGs for city-wide and each CVI area."""
    def decomp_and_plot(ts: pd.Series, title: str, fname: Path):
        ts = ts.dropna()
        if len(ts) < 24:  # need enough data for decomposition
            print(f"Skipping decomposition for {title}: not enough data points.")
            return
        result = seasonal_decompose(ts, model="additive", period=12)
        fig = result.plot()
        fig.suptitle(title)
        fig.set_size_inches(10, 8)
        plt.tight_layout()
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        print(f"Saved seasonal decomposition to {fname}")

    # City-wide
    city_ts = df.groupby("month").size()
    decomp_and_plot(city_ts, "Seasonal Decomposition - City-wide", out_dir / "seasonal_city.png")

    # Each CVI area (including Non-CVI)
    for area, sub in df.groupby("geo_area"):
        ts = sub.groupby("month").size()
        safe_name = str(area).replace(" ", "_").replace("/", "_")
        fname = out_dir / f"seasonal_{safe_name}.png"
        decomp_and_plot(ts, f"Seasonal Decomposition - {area}", fname)    

# ---------------------------------------------------------------------
# 4. ACF & PACF plots
# ---------------------------------------------------------------------

def acf_pacf_plots(df: pd.DataFrame, out_dir: Path) -> None:
    def plot_pair(ts: pd.Series, title_prefix: str, fname_acf: Path, fname_pacf: Path):
        ts = ts.dropna()
        if len(ts) < 24:
            print(f"Skipping ACF/PACF for {title_prefix}: not enough data points.")
            return
        fig_acf = plot_acf(ts, lags=24)
        fig_acf.suptitle(f"{title_prefix} - ACF")
        fig_acf.set_size_inches(8, 4)
        plt.tight_layout()
        fig_acf.savefig(fname_acf, dpi=150)
        plt.close(fig_acf)

        fig_pacf = plot_pacf(ts, lags=24, method="ywm")
        fig_pacf.suptitle(f"{title_prefix} - PACF")
        fig_pacf.set_size_inches(8, 4)
        plt.tight_layout()
        fig_pacf.savefig(fname_pacf, dpi=150)
        plt.close(fig_pacf)

        print(f"Saved ACF to {fname_acf} and PACF to {fname_pacf}")

    city_ts = df.groupby("month").size()
    plot_pair(city_ts, "City-wide", out_dir / "acf_city.png", out_dir / "pacf_city.png")

    for area, sub in df.groupby("geo_area"):
        ts = sub.groupby("month").size()
        safe_name = str(area).replace(" ", "_").replace("/", "_")
        plot_pair(
            ts,
            f"{area}",
            out_dir / f"acf_{safe_name}.png",
            out_dir / f"pacf_{safe_name}.png",
        )

# ---------------------------------------------------------------------
# 5. Animated spatial heatmap (monthly)
# ---------------------------------------------------------------------

def animated_heatmap(df: pd.DataFrame, out_html: Path) -> None:
    if "latitude" not in df.columns or "longitude" not in df.columns:
        print("Skipping heatmap: latitude/longitude not found.")
        return

    # Group incidents by month
    df = df.dropna(subset=["latitude", "longitude"]).copy()
    df["month_str"] = df["month"].dt.strftime("%Y-%m")

    months = sorted(df["month_str"].unique())
    data = []
    for m in months:
        sub = df[df["month_str"] == m]
        data.append(sub[["latitude", "longitude"]].values.tolist())

    # Center of Detroit-ish
    center_lat = df["latitude"].mean()
    center_lon = df["longitude"].mean()

    m = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles="cartodbpositron")
    HeatMapWithTime(
        data,
        index=months,
        auto_play=False,
        max_opacity=0.8,
        radius=12,
        use_local_extrema=False,
    ).add_to(m)

    m.save(out_html.as_posix())
    print(f"Saved animated heatmap to {out_html}")

def main():
    parser = argparse.ArgumentParser(description="Exploratory Data Analysis for RMS CVI-mapped incidents.")
    parser.add_argument("--input", default="rms_data_cleaned_mapped.csv", help="RMS mapped CSV input file")
    parser.add_argument("--out-dir", default="eda_outputs", help="Output directory for EDA artifacts")
    parser.add_argument("--program-start", default="2023-08-01", help="ShotStopper start date (YYYY-MM-DD)")
    parser.add_argument("--cvi-file", default="Community_Violence_Intervention_Areas.geojson",
                        help="CVI polygons file (GeoJSON)")
    parser.add_argument("--name-col", default="CVI_AREA_NAME", help="Area name column in CVI polygons")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    ensure_out_dir(out_dir)

    program_start = pd.to_datetime(args.program_start)

    # Load RMS data
    df = load_rms(args.input)

    # 1. Monthly plots
    plot_monthly_counts_city(df, out_dir / "monthly_city.html", program_start)
    plot_monthly_counts_cvi_vs_non(
        df,
        out_dir / "monthly_non_cvi.html",
        out_dir / "monthly_cvi_only.html",
        program_start,
    )

    # 2. ADF tests
    run_adf_all(df, out_dir / "adf_results.csv")

    # 3. Seasonal decomposition
    seasonal_decomp_plots(df, out_dir)

    # 4. ACF & PACF
    acf_pacf_plots(df, out_dir)

    # 5. Animated heatmap
    animated_heatmap(df, out_dir / "heatmap_time.html")

if __name__ == "__main__":
    main()
