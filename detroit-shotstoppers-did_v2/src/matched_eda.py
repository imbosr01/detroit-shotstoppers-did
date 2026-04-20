#!/usr/bin/env python3
"""
matched_eda.py

Light exploratory analysis for matched CVI vs matched control zones.

Includes:
    - Monthly pre-treatment RMS trend plots for each matched pair (HTML)
    - Seasonal decomposition plots for each matched pair (PNG)

Inputs:
    - RMS mapped CSV
    - CVI polygons
    - matched candidate controls GeoJSON
    - cvi_candidate_matches.csv

Outputs:
    - one HTML trend plot per matched pair
    - one seasonal decomposition PNG per CVI series
    - one seasonal decomposition PNG per matched control series
    - monthly_pairs_panel.csv

Notes:
    - This script focuses on matched-pair validation, not citywide EDA.
    - Seasonal decomposition uses monthly counts and period=12.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd
import plotly.graph_objects as go

from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt


# -----------------------------
# Configuration
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
AREA_CRS = "EPSG:26917"

DEFAULT_RMS_FILE = BASE_DIR / "src" / "rms_data_cleaned_mapped.csv"
DEFAULT_CVI_FILE = BASE_DIR / "assets" / "Community_Violence_Intervention_Areas.geojson"
DEFAULT_MATCHED_CONTROLS_FILE = BASE_DIR / "src" / "matched_controls" / "matched_candidate_controls.geojson"
DEFAULT_MATCHES_FILE = BASE_DIR / "src" / "matched_controls" / "cvi_candidate_matches.csv"

DEFAULT_LAT_COL = "latitude"
DEFAULT_LON_COL = "longitude"
DEFAULT_DATE_COL = "incident_occurred_at"
DEFAULT_CVI_NAME_COL = "CVI_AREA_NAME"

DEFAULT_PRE_START = "2021-01-01"
DEFAULT_PRE_END = "2023-07-31"
DEFAULT_TREATMENT_START = "2023-08-01"


# -----------------------------
# Argument parsing
# -----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Matched-pair EDA for CVIs and matched control zones."
    )
    parser.add_argument(
        "--rms-file",
        default=DEFAULT_RMS_FILE,
        help=f"Path to RMS mapped CSV (default: {DEFAULT_RMS_FILE})",
    )
    parser.add_argument(
        "--cvi-file",
        default=DEFAULT_CVI_FILE,
        help=f"Path to CVI GeoJSON (default: {DEFAULT_CVI_FILE})",
    )
    parser.add_argument(
        "--matched-controls-file",
        default=DEFAULT_MATCHED_CONTROLS_FILE,
        help=f"Path to matched controls GeoJSON (default: {DEFAULT_MATCHED_CONTROLS_FILE})",
    )
    parser.add_argument(
        "--matches-file",
        default=DEFAULT_MATCHES_FILE,
        help=f"Path to cvi_candidate_matches.csv (default: {DEFAULT_MATCHES_FILE})",
    )
    parser.add_argument(
        "--lat-col",
        default=DEFAULT_LAT_COL,
        help=f"Latitude column in RMS CSV (default: {DEFAULT_LAT_COL})",
    )
    parser.add_argument(
        "--lon-col",
        default=DEFAULT_LON_COL,
        help=f"Longitude column in RMS CSV (default: {DEFAULT_LON_COL})",
    )
    parser.add_argument(
        "--date-col",
        default=DEFAULT_DATE_COL,
        help=f"Date column in RMS CSV (default: {DEFAULT_DATE_COL})",
    )
    parser.add_argument(
        "--cvi-name-col",
        default=DEFAULT_CVI_NAME_COL,
        help=f"CVI name column in CVI GeoJSON (default: {DEFAULT_CVI_NAME_COL})",
    )
    parser.add_argument(
        "--pre-start",
        default=DEFAULT_PRE_START,
        help=f"Pre-treatment window start (default: {DEFAULT_PRE_START})",
    )
    parser.add_argument(
        "--pre-end",
        default=DEFAULT_PRE_END,
        help=f"Pre-treatment window end (default: {DEFAULT_PRE_END})",
    )
    parser.add_argument(
        "--treatment-start",
        default=DEFAULT_TREATMENT_START,
        help=f"Treatment start date (default: {DEFAULT_TREATMENT_START})",
    )
    parser.add_argument(
        "--out-dir",
        default="matched_controls",
        help="Output directory (default: matched_controls)",
    )
    return parser.parse_args()


# -----------------------------
# Helpers
# -----------------------------

def ensure_out_dir(path: str | Path) -> Path:
    out_dir = Path(path)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def safe_name(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in str(s))


def load_rms_points(
    rms_csv_path,
    lon_col="longitude",
    lat_col="latitude",
    date_col="incident_occurred_at",
):
    df = pd.read_csv(rms_csv_path)
    df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")
    df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    df = df.dropna(subset=[lon_col, lat_col, date_col]).copy()

    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
        crs="EPSG:4326",
    ).to_crs(AREA_CRS)

    return gdf


def compute_monthly_counts_for_pairs(
    rms_points,
    cvi_gdf,
    matched_controls_gdf,
    matches_df,
    date_col="incident_occurred_at",
    pre_start="2021-01-01",
    pre_end="2023-07-31",
):
    pre_start = pd.Timestamp(pre_start)
    pre_end = pd.Timestamp(pre_end)

    rms_points = rms_points.loc[
        (rms_points[date_col] >= pre_start) & (rms_points[date_col] <= pre_end)
    ].copy()

    cvi_gdf = cvi_gdf[["cvi_name", cvi_gdf.geometry.name]].copy()
    matched_controls_gdf = matched_controls_gdf[
        ["candidate_id", "matched_cvi_name", matched_controls_gdf.geometry.name]
    ].copy()

    # Join RMS points to CVIs
    rms_cvi = gpd.sjoin(
        rms_points,
        cvi_gdf,
        how="inner",
        predicate="within",
    ).copy()
    rms_cvi["month"] = rms_cvi[date_col].dt.to_period("M").dt.to_timestamp()

    cvi_monthly = (
        rms_cvi.groupby(["cvi_name", "month"])
        .size()
        .rename("count")
        .reset_index()
    )
    cvi_monthly["group"] = "CVI"

    # Join RMS points to matched controls
    rms_ctrl = gpd.sjoin(
        rms_points,
        matched_controls_gdf,
        how="inner",
        predicate="within",
    ).copy()
    rms_ctrl["month"] = rms_ctrl[date_col].dt.to_period("M").dt.to_timestamp()

    ctrl_monthly = (
        rms_ctrl.groupby(["matched_cvi_name", "candidate_id", "month"])
        .size()
        .rename("count")
        .reset_index()
        .rename(columns={"matched_cvi_name": "cvi_name"})
    )
    ctrl_monthly["group"] = "Matched Control"

    # Build complete monthly panel for every matched pair
    months = pd.period_range(pre_start, pre_end, freq="M").to_timestamp()
    rows = []

    for _, row in matches_df.iterrows():
        cvi_name = row["cvi_name"]
        candidate_id = row["candidate_id"]

        cvi_sub = cvi_monthly.loc[cvi_monthly["cvi_name"] == cvi_name, ["month", "count"]]
        ctrl_sub = ctrl_monthly.loc[
            (ctrl_monthly["cvi_name"] == cvi_name)
            & (ctrl_monthly["candidate_id"] == candidate_id),
            ["month", "count"]
        ]

        cvi_lookup = dict(zip(cvi_sub["month"], cvi_sub["count"]))
        ctrl_lookup = dict(zip(ctrl_sub["month"], ctrl_sub["count"]))

        for m in months:
            rows.append({
                "cvi_name": cvi_name,
                "candidate_id": candidate_id,
                "month": m,
                "group": "CVI",
                "count": cvi_lookup.get(m, 0),
            })
            rows.append({
                "cvi_name": cvi_name,
                "candidate_id": candidate_id,
                "month": m,
                "group": "Matched Control",
                "count": ctrl_lookup.get(m, 0),
            })

    return pd.DataFrame(rows)


# -----------------------------
# Trend plots
# -----------------------------

def plot_matched_pair_trend(
    monthly_df,
    cvi_name,
    treatment_start="2023-08-01",
    out_dir=None,
    show_plot=False,
):
    sub = monthly_df.loc[monthly_df["cvi_name"] == cvi_name].copy()

    if sub.empty:
        raise ValueError(f"No monthly data found for CVI '{cvi_name}'")

    candidate_id = sub["candidate_id"].iloc[0]

    fig = go.Figure()

    for group_name in ["CVI", "Matched Control"]:
        g = sub.loc[sub["group"] == group_name].sort_values("month")
        fig.add_trace(
            go.Scatter(
                x=g["month"],
                y=g["count"],
                mode="lines+markers",
                name=group_name,
            )
        )

    treatment_start = pd.Timestamp(treatment_start)

    fig.add_shape(
        type="line",
        x0=treatment_start,
        x1=treatment_start,
        y0=0,
        y1=1,
        xref="x",
        yref="paper",
        line=dict(dash="dash", width=1),
    )

    fig.add_annotation(
        x=treatment_start,
        y=1,
        xref="x",
        yref="paper",
        text="ShotStoppers",
        showarrow=False,
        xanchor="left",
        yanchor="bottom",
    )

    fig.update_layout(
        title=f"{cvi_name} vs {candidate_id} (Pre-treatment Monthly RMS Incidents)",
        xaxis_title="Month",
        yaxis_title="Monthly RMS Incidents",
        hovermode="x unified",
    )

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        out_path = out_dir / f"{safe_name(cvi_name)}__vs__{safe_name(candidate_id)}.html"
        fig.write_html(out_path)
        print(f"Wrote {out_path}")

    if show_plot:
        fig.show()

    return fig


def plot_all_matched_pair_trends(
    monthly_df,
    treatment_start="2023-08-01",
    out_dir="matched_controls",
    show_plot=False,
):
    for cvi_name in monthly_df["cvi_name"].drop_duplicates():
        plot_matched_pair_trend(
            monthly_df=monthly_df,
            cvi_name=cvi_name,
            treatment_start=treatment_start,
            out_dir=out_dir,
            show_plot=show_plot,
        )


# -----------------------------
# Seasonal decomposition
# -----------------------------

def seasonal_decomp_plot(ts: pd.Series, title: str, out_path: Path, period: int = 12) -> None:
    ts = ts.sort_index().dropna()

    if len(ts) < 24:
        print(f"Skipping seasonal decomposition for {title}: not enough data points.")
        return

    result = seasonal_decompose(ts, model="additive", period=period)

    fig = result.plot()
    fig.suptitle(title)
    fig.set_size_inches(10, 8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def make_seasonal_plots(monthly_df: pd.DataFrame, out_dir: Path) -> None:
    seasonal_dir = out_dir / "seasonal"
    seasonal_dir.mkdir(parents=True, exist_ok=True)

    for cvi_name in monthly_df["cvi_name"].drop_duplicates():
        sub = monthly_df.loc[monthly_df["cvi_name"] == cvi_name].copy()
        candidate_id = sub["candidate_id"].iloc[0]

        # CVI series
        cvi_ts = (
            sub.loc[sub["group"] == "CVI", ["month", "count"]]
            .sort_values("month")
            .set_index("month")["count"]
        )

        # Control series
        ctrl_ts = (
            sub.loc[sub["group"] == "Matched Control", ["month", "count"]]
            .sort_values("month")
            .set_index("month")["count"]
        )

        cvi_out = seasonal_dir / f"seasonal__{safe_name(cvi_name)}__CVI.png"
        ctrl_out = seasonal_dir / f"seasonal__{safe_name(cvi_name)}__{safe_name(candidate_id)}.png"

        seasonal_decomp_plot(
            cvi_ts,
            title=f"Seasonal Decomposition - {cvi_name} (CVI)",
            out_path=cvi_out,
        )
        seasonal_decomp_plot(
            ctrl_ts,
            title=f"Seasonal Decomposition - {cvi_name} vs {candidate_id} (Matched Control)",
            out_path=ctrl_out,
        )


# -----------------------------
# Main
# -----------------------------

def main():
    args = parse_args()
    out_dir = ensure_out_dir(args.out_dir)

    # Load inputs
    rms_points = load_rms_points(
        args.rms_file,
        lon_col=args.lon_col,
        lat_col=args.lat_col,
        date_col=args.date_col,
    )

    cvis = gpd.read_file(args.cvi_file).to_crs(AREA_CRS)
    cvis["cvi_name"] = cvis[args.cvi_name_col].astype(str).str.strip()

    matched_controls = gpd.read_file(args.matched_controls_file).to_crs(AREA_CRS)
    matches = pd.read_csv(args.matches_file)

    # Build monthly panel
    monthly_pairs = compute_monthly_counts_for_pairs(
        rms_points=rms_points,
        cvi_gdf=cvis,
        matched_controls_gdf=matched_controls,
        matches_df=matches,
        date_col=args.date_col,
        pre_start=args.pre_start,
        pre_end=args.pre_end,
    )

    panel_path = out_dir / "monthly_pairs_panel.csv"
    monthly_pairs.to_csv(panel_path, index=False)
    print(f"Wrote {panel_path}")

    # Trend HTMLs
    plot_all_matched_pair_trends(
        monthly_pairs,
        treatment_start=args.treatment_start,
        out_dir=out_dir,
        show_plot=False,
    )

    # Seasonal decomposition PNGs
    make_seasonal_plots(monthly_pairs, out_dir)


if __name__ == "__main__":
    main()