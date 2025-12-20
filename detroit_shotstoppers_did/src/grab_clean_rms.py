#!/usr/bin/env python3
"""
Grab + clean Detroit RMS Crime Incidents for modeling.

Server-side filters:
  incident_year >= 2021 AND (
      offense_description = 'NON-FATAL SHOOTING' OR
      charge_description   = 'NON-FATAL SHOOTING' OR
      arrest_charge        = '13004'               OR
      offense_category     = 'HOMICIDE'
  )

Data cleaning:
  - Convert incident_occurred_at (epoch ms) -> America/Detroit local datetime (naive wall clock)
  - Strip *leading* apostrophes from certain text-like fields seen in the CSV
  - Coerce longitude/latitude to numeric; drop rows with NaN lon/lat and print the count
  - Border check (30 m buffer) vs the City of Detroit boundary; write outsides to rms_outside_Detroit.csv
  - Keep API field names in outputs
  - Write cleaned file: rms_data_cleaned.csv

Usage:
  python grab_clean_rms.py \
    --out-clean rms_data_cleaned.csv \
    --out-outside rms_outside_Detroit.csv \
    --buffer-m 30 \
    --boundary "https://opendata.arcgis.com/api/v3/datasets/86b221bb68ca4364afe81d156e54f95c_0/downloads/data?format=geojson&spatialRefId=4326" \
    --page-size 2000
"""

import argparse
import sys
import time
from typing import Dict, List, Tuple

import pandas as pd
import geopandas as gpd
import requests

# ArcGIS layer (RMS incidents)
LAYER_QUERY_URL = (
    "https://services2.arcgis.com/qvkbeam7Wirps6zC/ArcGIS/rest/services/"
    "RMS_Crime_Incidents/FeatureServer/0/query"
)

# API field names to keep
API_FIELDS = [
    "incident_entry_id",
    "nearest_intersection",      # if absent in layer, we drop it silently
    "offense_category",
    "offense_description",
    "state_offense_code",
    "arrest_charge",
    "charge_description",
    "incident_occurred_at",
    "scout_car_area",
    "police_precinct",
    "zip_code",
    "longitude",
    "latitude",
    "ESRI_OID",  # for stable paging; removed before final write
]

# CRS for buffer-in-meters math (Detroit ≈ UTM Zone 17N)
WGS84 = "EPSG:4326"
METRIC_CRS = "EPSG:32617"

# Default Detroit boundary (live GeoJSON from the Data Portal)
DEFAULT_BOUNDARY_URL = (
    "https://opendata.arcgis.com/api/v3/datasets/86b221bb68ca4364afe81d156e54f95c_0/"
    "downloads/data?format=geojson&spatialRefId=4326"
)


# Query building & fetching

def build_where_clause() -> str:
    # Same logic you used previously in grab_rms.py
    return (
        "(incident_year >= 2021) AND ("
        "offense_description = 'NON-FATAL SHOOTING' OR "
        "charge_description = 'NON-FATAL SHOOTING' OR "
        "arrest_charge = '13004' OR "
        "offense_category = 'HOMICIDE'"
        ")"
    )


def fetch_page(offset: int, page_size: int, timeout: int = 60) -> Dict:
    params = {
        "where": build_where_clause(),
        "outFields": ",".join(API_FIELDS),
        "returnGeometry": "false",
        "f": "json",
        "orderByFields": "ESRI_OID ASC",
        "resultOffset": offset,
        "resultRecordCount": page_size,
        "sqlFormat": "standard",
    }
    r = requests.get(LAYER_QUERY_URL, params=params, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    if "error" in data:
        msg = data["error"].get("message", "ArcGIS error")
        details = " | ".join(data["error"].get("details", []))
        raise RuntimeError(f"ArcGIS query error: {msg}. {details}")
    return data


def collect_records(page_size: int, sleep_s: float, max_pages: int | None) -> pd.DataFrame:
    rows: List[Dict] = []
    offset = 0
    page = 0
    while True:
        page += 1
        if max_pages is not None and page > max_pages:
            break
        data = fetch_page(offset=offset, page_size=page_size)
        feats = data.get("features", [])
        if not feats:
            break
        for f in feats:
            attrs = f.get("attributes", {}) or {}
            rows.append({k: attrs.get(k) for k in API_FIELDS})
        if len(feats) < page_size:
            break
        offset += page_size
        time.sleep(sleep_s)
    df = pd.DataFrame.from_records(rows)
    return df


# City boundary

def load_boundary(boundary_path_or_url: str) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(boundary_path_or_url)
    if gdf.crs is None:
        gdf = gdf.set_crs(WGS84, allow_override=True)
    # dissolve to a single polygon for fast within()
    if len(gdf) > 1:
        gdf = gdf.dissolve(by=None, as_index=False)
    return gdf


def buffer_boundary(boundary_gdf: gpd.GeoDataFrame, buffer_m: float) -> gpd.GeoSeries:
    b_m = boundary_gdf.to_crs(METRIC_CRS).buffer(buffer_m).unary_union
    return gpd.GeoSeries([b_m], crs=METRIC_CRS).to_crs(WGS84)


# Data cleaning

def strip_leading_apostrophes(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Remove a single leading apostrophe that sometimes appears in CSV exports."""
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype("string").str.replace(r"^'+", "", regex=True)
    return df


def coerce_lon_lat(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Make lon/lat numeric; count and drop rows with NaN afterwards."""
    for c in ("longitude", "latitude"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    before = len(df)
    df = df.dropna(subset=["longitude", "latitude"]).copy()
    removed = before - len(df)
    return df, removed


def convert_incident_time(df: pd.DataFrame, tz: str = "America/Detroit") -> pd.DataFrame:
    """
    incident_occurred_at is epoch ms in ArcGIS. Convert to local Detroit naive datetime.
    """
    if "incident_occurred_at" in df.columns:
        dt_utc = pd.to_datetime(df["incident_occurred_at"], unit="ms", utc=True)
        df["incident_occurred_at"] = dt_utc.dt.tz_convert(tz).dt.tz_localize(None)
    return df


def boundary_check_and_split(
    df: pd.DataFrame, boundary_path_or_url: str, buffer_m: float
) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
    """Return (inside_df, outside_df, num_outside)."""
    gdf_pts = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs=WGS84,
    )
    boundary = load_boundary(boundary_path_or_url)
    boundary_wgs = buffer_boundary(boundary, buffer_m)
    inside_mask = gdf_pts.geometry.within(boundary_wgs.iloc[0])
    outside_df = gdf_pts.loc[~inside_mask].drop(columns=["geometry"]).copy()
    inside_df = gdf_pts.loc[inside_mask].drop(columns=["geometry"]).copy()
    return inside_df, outside_df, int(len(outside_df))


def clean_rms(
    df_raw: pd.DataFrame,
    boundary_path_or_url: str,
    buffer_m: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Full cleaning pipeline. Returns (clean_df, outside_df, stats).
    """

    # 1) Keep only fields that actually arrived (some layers might lack nearest_intersection)
    keep = [c for c in API_FIELDS if c in df_raw.columns]
    df = df_raw[keep].copy()

    # 2) Strip leading apostrophes from fields where you observed them
    df = strip_leading_apostrophes(
        df,
        cols=[
            "state_offense_code",
            "arrest_charge",
            "scout_car_area",
            "police_precinct",
            "latitude",  # seen in your CSV
        ],
    )

    # 3) Convert time field
    df = convert_incident_time(df, tz="America/Detroit")

    # 4) Coerce lon/lat and drop missing
    df, n_missing_ll = coerce_lon_lat(df)

    # 5) Boundary split
    inside_df, outside_df, n_outside = boundary_check_and_split(
        df, boundary_path_or_url, buffer_m
    )

    # 6) Remove ESRI_OID before writing (internal paging field)
    if "ESRI_OID" in inside_df.columns:
        inside_df = inside_df.drop(columns=["ESRI_OID"])
    if "ESRI_OID" in outside_df.columns:
        outside_df = outside_df.drop(columns=["ESRI_OID"])

    stats = {
        "rows_input": int(len(df_raw)),
        "rows_after_lonlat_drop": int(len(df)),
        "rows_missing_lonlat_removed": int(n_missing_ll),
        "rows_outside_detroit": int(n_outside),
        "buffer_meters": float(buffer_m),
    }
    return inside_df, outside_df, stats


# Pipeline

def main():
    p = argparse.ArgumentParser(description="Grab + clean Detroit RMS incidents")
    p.add_argument("--out-clean", default="rms_data_cleaned.csv", help="Output CSV for cleaned RMS")
    p.add_argument("--out-outside", default="rms_outside_Detroit.csv",
                   help="CSV for RMS points outside Detroit (written only if any)")
    p.add_argument("--boundary", default=DEFAULT_BOUNDARY_URL,
                   help="Detroit boundary GeoJSON/URL")
    p.add_argument("--buffer-m", type=float, default=30.0, help="Boundary buffer in meters (default: 30)")
    p.add_argument("--page-size", type=int, default=2000, help="Records per request (<=2000)")
    p.add_argument("--sleep", type=float, default=0.1, help="Seconds to sleep between pages")
    p.add_argument("--max-pages", type=int, default=None, help="Optional cap on number of pages")
    args = p.parse_args()

    if args.page_size <= 0 or args.page_size > 2000:
        print("page-size must be in 1..2000 (service limit).", file=sys.stderr)
        sys.exit(2)

    # Fetch
    df_raw = collect_records(page_size=args.page_size, sleep_s=args.sleep, max_pages=args.max_pages)
    if df_raw.empty:
        print("No records returned from service with the specified filters.", file=sys.stderr)
        sys.exit(0)

    # Clean
    clean_df, outside_df, stats = clean_rms(df_raw, args.boundary, args.buffer_m)

    # Outside reporting
    if stats["rows_outside_detroit"] > 0:
        outside_df.to_csv(args.out_outside, index=False)
        print(f"{stats['rows_outside_detroit']} RMS points outside of Detroit.")
    else:
        print("No RMS points outside of Detroit.")

    # Final write (API field names preserved)
    clean_df.to_csv(args.out_clean, index=False)
    print(f"{stats['rows_missing_lonlat_removed']} rows with missing lat/long coordinates removed.")
    print(f"Output written to {args.out_clean}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[grab_clean_rms.py] ERROR: {e}", file=sys.stderr)
        sys.exit(1)
