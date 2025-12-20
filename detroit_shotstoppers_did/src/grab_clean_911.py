#!/usr/bin/env python3
"""
Grab + clean Detroit Police-Serviced 911 Calls focused on ShotSpotter-like events.

Server-side filters:
  - called_at >= 2021-01-01
  - (category == 'SHOTSPT' OR code_description == 'SHOTS FIRED IP')

Data cleaning:
  - Parse "Call Time" to datetime
  - Keep only rows with Total Response Time <= 180
  - Drop rows with missing Latitude/Longitude
  - Flag calls outside Detroit boundary (buffered), export to 911_outside_Detroit.csv if any
  - Output cleaned CSV as 911_data_cleaned.csv

Usage:
  python grab_clean_911.py \
    --out-clean 911_data_cleaned.csv \
    --boundary City_of_Detroit_Boundary.geojson \
    --buffer-m 30 \
    --page-size 2000
"""

import argparse
import csv
import sys
import time
from typing import Dict, List

import pandas as pd
import geopandas as gpd
import requests

# --- ArcGIS FeatureServer endpoint (authoritative backend for the Portal dataset) ---
LAYER_QUERY_URL = (
    "https://services2.arcgis.com/qvkbeam7Wirps6zC/ArcGIS/rest/services/"
    "Police_Serviced_911_Calls/FeatureServer/0/query"
)

# --- API field names we will request (no spaces) ---
API_FIELDS = [
    "incident_entry_id",
    "incident_location",
    "call_description",
    "category",
    "called_at",
    "total_response_time",
    "zip_code",
    "precinct",
    "scout_car_area",
    "longitude",
    "latitude",
    "ESRI_OID",  # useful for paging
]

# --- Keep output column names the same as the Detroit Data Portal (with spaces/casing) ---
RENAME_FOR_PORTAL = {
    "incident_entry_id": "incident_entry_id",
    "incident_location": "Nearest Intersection",
    "call_description": "Code Description",
    "category": "Category",
    "called_at": "Call Time",
    "total_response_time": "Total Response Time",
    "zip_code": "Zip Code",
    "precinct": "Precinct",
    "scout_car_area": "Scout Car Area",
    "longitude": "Longitude",
    "latitude": "Latitude",
    # ESRI_OID intentionally not kept in final outputs
}

# For buffering in meters: Detroit ~ UTM Zone 17N
WGS84 = "EPSG:4326"
METRIC_CRS = "EPSG:32617"

def build_where_clause() -> str:
    """
    Build the ArcGIS 'where' SQL clause to fetch ShotSpotter and gunfire-related 911 calls
    from 2021 onward in the new Police Serviced 911 Calls dataset.
    """
    return (
        "called_at >= timestamp '2021-01-01 00:00:00' AND "
        "(category = 'SHOTSPT' OR category = 'SHOTS IP' OR category = 'SHOTS JH')"
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
            # Keep only requested API fields to stabilize memory footprint
            rows.append({k: attrs.get(k) for k in API_FIELDS})
        if len(feats) < page_size:
            break
        offset += page_size
        time.sleep(sleep_s)

    df = pd.DataFrame.from_records(rows)
    return df


def load_boundary(boundary_path_or_url: str) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(boundary_path_or_url)
    if gdf.crs is None:
        # GeoJSON often omits CRS; default to WGS84 lon/lat
        gdf = gdf.set_crs(WGS84, allow_override=True)
    # Dissolve to one geometry if multiple parts
    if len(gdf) > 1:
        gdf = gdf.dissolve(by=None, as_index=False)
    return gdf


def buffer_boundary(boundary_gdf: gpd.GeoDataFrame, buffer_m: float) -> gpd.GeoSeries:
    b_metric = boundary_gdf.to_crs(METRIC_CRS)
    b_buf = b_metric.buffer(buffer_m)
    unioned = b_buf.unary_union
    return gpd.GeoSeries([unioned], crs=METRIC_CRS).to_crs(WGS84)


def clean_911(
    df_raw: pd.DataFrame,
    boundary_path_or_url: str,
    buffer_m: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Apply cleaning steps and boundary check.
    Returns: (clean_df, outside_df, stats_dict)
    """
    # --- Start by assigning df_raw to df ---
    df = df_raw.copy()

    # --- Convert epoch milliseconds to datetime before filtering ---
    if "called_at" in df.columns:
        dt_utc = pd.to_datetime(df["called_at"], unit="ms", utc=True)
        dt_detroit = dt_utc.dt.tz_convert("America/Detroit")
        df["called_at"] = dt_detroit.dt.tz_localize(None)  # Detroit local time (naive)

    # --- Filter Total Response Time <= 180 ---
    over_180 = df["total_response_time"] > 180
    num_over_180 = int(over_180.sum(skipna=True))
    df = df.loc[~over_180].copy()

    # --- Drop rows missing lat/long ---
    lat_missing = df["latitude"].isna()
    lon_missing = df["longitude"].isna()
    missing_ll = (lat_missing | lon_missing)
    num_missing_ll = int(missing_ll.sum())
    df = df.loc[~missing_ll].copy()

    # --- Boundary check against Detroit boundary (+buffer) ---
    gdf_pts = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs=WGS84,
    )
    boundary_gdf = load_boundary(boundary_path_or_url)
    boundary_wgs = buffer_boundary(boundary_gdf, buffer_m)

    inside = gdf_pts.geometry.within(boundary_wgs.iloc[0])
    outside_df = gdf_pts.loc[~inside].drop(columns=["geometry"]).copy()
    num_outside = int(len(outside_df))
    clean_df = gdf_pts.loc[inside].drop(columns=["geometry"]).copy()

    # --- Drop internal field before final write ---
    if "ESRI_OID" in clean_df.columns:
        clean_df = clean_df.drop(columns=["ESRI_OID"])
    if "ESRI_OID" in outside_df.columns:
        outside_df = outside_df.drop(columns=["ESRI_OID"])

    stats = {
        "rows_dropped_over_180": num_over_180,
        "rows_dropped_missing_latlon": num_missing_ll,
        "rows_outside_detroit": num_outside,
        "buffer_meters": float(buffer_m),
    }
    return clean_df, outside_df, stats


def main():
    parser = argparse.ArgumentParser(description="Grab + clean Detroit ShotSpotter-related 911 calls")
    parser.add_argument("--out-clean", default="911_data_cleaned.csv", help="Output CSV for cleaned 911 data")
    parser.add_argument("--out-outside", default="911_outside_Detroit.csv",
                        help="Output CSV for calls outside Detroit (only written if any)")
    #parser.add_argument("--boundary", default="City_of_Detroit_Boundary.geojson",
    #                    help="Detroit boundary GeoJSON/URL")
    parser.add_argument("--boundary",
        default="https://opendata.arcgis.com/api/v3/datasets/86b221bb68ca4364afe81d156e54f95c_0/downloads/data?format=geojson&spatialRefId=4326",
        help="URL or path to Detroit boundary GeoJSON file (defaults to live Detroit Data Portal endpoint)")
    parser.add_argument("--buffer-m", type=float, default=30.0, help="Boundary buffer in meters (default: 30)")
    parser.add_argument("--page-size", type=int, default=2000, help="Records per page (<=2000 service limit)")
    parser.add_argument("--sleep", type=float, default=0.1, help="Seconds to sleep between pages")
    parser.add_argument("--max-pages", type=int, default=None, help="Optional cap on number of pages")
    args = parser.parse_args()

    if args.page_size <= 0 or args.page_size > 2000:
        print("page-size must be in 1..2000 (service limit).", file=sys.stderr)
        sys.exit(2)

    # 1) Fetch
    df_raw = collect_records(page_size=args.page_size, sleep_s=args.sleep, max_pages=args.max_pages)

    if df_raw.empty:
        print("No records returned from service with the specified filters.", file=sys.stderr)
        sys.exit(0)

    # 2) Clean
    clean_df, outside_df, stats = clean_911(df_raw, args.boundary, args.buffer_m)

    # 3) Write outside-only CSV if needed
    if stats["rows_outside_detroit"] > 0:
        outside_df.to_csv(args.out_outside, index=False)
        print(f"{stats['rows_outside_detroit']} 911 calls outside of Detroit.")
    else:
        print("No 911 calls outside of Detroit.")

    # 4) Write cleaned CSV
    clean_df.to_csv(args.out_clean, index=False)

    # 5) Print cleaning stats
    print(f"{stats['rows_dropped_over_180']} rows with Total Response Time > 180 removed.")
    print(f"{stats['rows_dropped_missing_latlon']} rows with missing lat/long coordinates removed.")
    print(f"Output written to {args.out_clean}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[grab_clean_911.py] ERROR: {e}", file=sys.stderr)
        sys.exit(1)