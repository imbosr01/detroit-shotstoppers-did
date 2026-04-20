#!/usr/bin/env python3
"""
cvi_mapping.py

Map RMS & 911 incidents to CVI areas with a small buffer tolerance.
- Inputs: rms_data_cleaned.csv, 911_data_cleaned.csv
- Params:
    --buffer-m  (default 7.0 meters)
    --cvi-file  (default Community_Violence_Intervention_Areas.geojson)
    --name-col  (default CVI_AREA_NAME)
- Outputs:
    911_data_cleaned_mapped.csv
    rms_data_cleaned_mapped.csv
    rms_map.html
    911_map.html

Assignment rule (in metric CRS):
  - inside polygon -> cvi_zone = polygon name, buffer = False (dist_m == 0)
  - within buffer (0 < dist_m ≤ buffer_m) -> cvi_zone = nearest polygon name, buffer = True
  - else -> cvi_zone = "Non-CVI", buffer = False
"""

import argparse
from pathlib import Path
import sys

import pandas as pd
import geopandas as gpd
import folium

WGS84 = "EPSG:4326"
METRIC = "EPSG:32617"  # Detroit ≈ UTM Zone 17N


# ---------------- helpers ----------------

def load_cvi(cvi_file: str, name_col: str) -> gpd.GeoDataFrame:
    cvi = gpd.read_file(cvi_file)
    if cvi.crs is None:
        cvi = cvi.set_crs(WGS84, allow_override=True)
    if name_col not in cvi.columns:
        raise KeyError(f"Column '{name_col}' not found in CVI file. Available: {list(cvi.columns)}")
    return cvi[[name_col, "geometry"]].to_crs(WGS84)


def build_cvi_buffer(cvi_wgs: gpd.GeoDataFrame, buffer_m: float) -> gpd.GeoDataFrame:
    cvi_m = cvi_wgs.to_crs(METRIC)
    buf_m = cvi_m.copy()
    buf_m["geometry"] = cvi_m.geometry.buffer(buffer_m)
    return buf_m.to_crs(WGS84)


def load_points(csv_path: str) -> gpd.GeoDataFrame:
    df = pd.read_csv(csv_path)
    if "longitude" not in df.columns or "latitude" not in df.columns:
        raise KeyError("Input CSV must contain 'longitude' and 'latitude' columns.")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df = df.dropna(subset=["longitude", "latitude"]).copy()
    return gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs=WGS84
    )


def assign_cvi(
    pts_wgs: gpd.GeoDataFrame,
    cvi_wgs: gpd.GeoDataFrame,
    buffer_m: float,
    name_col: str,
) -> gpd.GeoDataFrame:
    """
    Nearest-within-buffer assignment. Returns a GeoDataFrame including:
      - cvi_zone (str)         # stable geography label (polygon-based)
      - buffer (bool)
      - dist_m (float, NaN if Non-CVI)
    """
    pts_m = pts_wgs.to_crs(METRIC)
    cvi_m = cvi_wgs.to_crs(METRIC)

    assigned_m = gpd.sjoin_nearest(
        pts_m,
        cvi_m[[name_col, "geometry"]],
        how="left",
        max_distance=buffer_m,
        distance_col="dist_m",
    )
    assigned = assigned_m.to_crs(WGS84)

    # Raw polygon name (or Non-CVI)
    assigned["cvi_zone_raw"] = assigned[name_col].fillna("Non-CVI")

    # Stable geography label: make it explicit the Team Pursuit polygon = Wayne Metro/Team Pursuit geography
    assigned["cvi_zone"] = assigned["cvi_zone_raw"].where(
        assigned["cvi_zone_raw"] != "Team Pursuit",
        "Wayne Metro/Team Pursuit",
    )

    assigned["buffer"] = assigned["dist_m"].apply(lambda d: bool(pd.notna(d) and d > 0))

    # Backwards-compatible alias for older scripts (EDA etc.)
    assigned["cvi_area"] = assigned["cvi_zone"]

    return assigned


def print_summary(tag: str, gdf: gpd.GeoDataFrame):
    total = len(gdf)
    print(f"\n=== {tag} SUMMARY ===")
    print(f"Total points: {total}")

    col = "cvi_zone" if "cvi_zone" in gdf.columns else "cvi_area"
    print(f"\nValue counts for {col}:")
    print(gdf[col].value_counts(dropna=False))

    if "buffer" in gdf.columns:
        n_buf = int(gdf["buffer"].sum())
        pct = (n_buf / total * 100.0) if total else 0.0
        print(f"\nBuffer-flagged: {n_buf} / {total} ({pct:.2f}%)")


def make_map(
    tag: str,
    cvi_wgs: gpd.GeoDataFrame,
    cvi_buf_wgs: gpd.GeoDataFrame,
    pts_wgs: gpd.GeoDataFrame,
    out_html: str,
):
    m = cvi_wgs.explore(
        name="CVI Areas",
        style_kwds=dict(color="#2D7FB8", weight=2, fill=True, fillOpacity=0.2),
    )
    cvi_buf_wgs.explore(
        m=m,
        name="CVI Buffers",
        style_kwds=dict(color="#FF7F0E", weight=1, fill=True, fillOpacity=0.15),
    )

    pts_buf = pts_wgs.loc[pts_wgs["buffer"] == True]
    pts_core = pts_wgs.loc[pts_wgs["buffer"] == False]

    base_tooltip = [c for c in ["cvi_zone", "buffer"] if c in pts_wgs.columns]

    pts_core.explore(
        m=m,
        name=f"{tag} (inside/non-CVI)",
        color="#d62728",
        marker_kwds=dict(radius=3, fill=True, fill_opacity=0.7),
        tooltip=base_tooltip + [
            c for c in [
                "incident_entry_id", "offense_description", "offense_category",
                "call_description"
            ] if c in pts_core.columns
        ],
    )
    pts_buf.explore(
        m=m,
        name=f"{tag} (buffer points)",
        color="#6a3d9a",
        marker_kwds=dict(radius=4, fill=True, fill_opacity=0.9),
        tooltip=base_tooltip
               + [c for c in ["dist_m"] if c in pts_buf.columns]
               + [c for c in [
                    "incident_entry_id", "offense_description", "offense_category",
                    "call_description"
                 ] if c in pts_buf.columns],
    )

    m.add_child(folium.LayerControl(collapsed=False))
    m.save(out_html)
    print(f"Map saved: {out_html}")


# ---------------- cli ----------------

def main():
    ap = argparse.ArgumentParser(description="Assign RMS & 911 incidents to CVI areas with buffer tolerance.")
    ap.add_argument("--rms-in", default="rms_data_cleaned.csv", help="Input RMS cleaned CSV")
    ap.add_argument("--calls-in", default="911_data_cleaned.csv", help="Input 911 cleaned CSV")
    ap.add_argument("--buffer-m", type=float, default=7.0, help="Buffer tolerance in meters (default: 7.0)")
    ap.add_argument("--cvi-file", default="Community_Violence_Intervention_Areas.geojson",
                    help="CVI polygons (GeoJSON/FeatureServer URL or local path)")
    ap.add_argument("--name-col", default="CVI_AREA_NAME", help="CVI area name column in the CVI file")
    ap.add_argument("--rms-out", default="rms_data_cleaned_mapped.csv", help="Output RMS mapped CSV")
    ap.add_argument("--calls-out", default="911_data_cleaned_mapped.csv", help="Output 911 mapped CSV")
    ap.add_argument("--rms-map", default="rms_map.html", help="Output RMS map HTML")
    ap.add_argument("--calls-map", default="911_map.html", help="Output 911 map HTML")
    args = ap.parse_args()

    # Load CVI polygons + buffer
    cvi_wgs = load_cvi(args.cvi_file, args.name_col)
    cvi_buf_wgs = build_cvi_buffer(cvi_wgs, args.buffer_m)

    # RMS
    rms_pts = load_points(args.rms_in)
    rms_assigned = assign_cvi(rms_pts, cvi_wgs, args.buffer_m, args.name_col)

    rms_assigned.drop(
        columns=["geometry", "index_right", args.name_col, "dist_m", "cvi_zone_raw"],
        errors="ignore"
    ).to_csv(args.rms_out, index=False)
    print_summary("RMS", rms_assigned)
    make_map("RMS", cvi_wgs, cvi_buf_wgs, rms_assigned, args.rms_map)

    # 911
    calls_pts = load_points(args.calls_in)
    calls_assigned = assign_cvi(calls_pts, cvi_wgs, args.buffer_m, args.name_col)

    calls_assigned.drop(
        columns=["geometry", "index_right", args.name_col, "dist_m", "cvi_zone_raw"],
        errors="ignore"
    ).to_csv(args.calls_out, index=False)
    print_summary("911", calls_assigned)
    make_map("911", cvi_wgs, cvi_buf_wgs, calls_assigned, args.calls_map)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[cvi_mapping.py] ERROR: {e}", file=sys.stderr)
        sys.exit(1)
