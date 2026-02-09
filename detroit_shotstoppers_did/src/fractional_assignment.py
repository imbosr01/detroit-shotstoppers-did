#!/usr/bin/env python3
"""
fractional_assignment.py

Purpose:
  Build a single long-format dataset combining RMS + 911 mapped CSVs and apply
  fractional (split) assignment for buffer points.

Key guarantees for downstream R script:
  - event_time (datetime)
  - source in {"RMS","911"}
  - cvi_area (stable geography) ALWAYS present
  - cvi_zone present, ALWAYS identical to cvi_area (alias)
  - buffer is a proper boolean
  - weight (float) sums to 1 per original incident row
  - neighbors_count indicates how many zones each buffer point was split across
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
import geopandas as gpd

WGS84 = "EPSG:4326"
METRIC = "EPSG:32617"  # Detroit ≈ UTM 17N


# ------------------------- helpers -------------------------

def normalize_zone_name(name) -> str:
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return "Non-CVI"
    s = str(name).strip()
    if s == "":
        return "Non-CVI"
    # keep the stable geography label consistent with your mapping script
    if s == "Team Pursuit":
        return "Wayne Metro/Team Pursuit"
    return s


def detect_time_column(df: pd.DataFrame) -> str:
    # RMS
    if "incident_occurred_at" in df.columns:
        return "incident_occurred_at"
    # 911
    if "called_at" in df.columns:
        return "called_at"
    raise KeyError("No time column found. Expected 'incident_occurred_at' (RMS) or 'called_at' (911).")


def load_points(csv_path: str, source: str) -> gpd.GeoDataFrame:
    df = pd.read_csv(csv_path, low_memory=False)

    # Drop join debris if present
    for col in ["index_right", "dist_m", "CVI_AREA_NAME"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # coords
    for col in ["longitude", "latitude"]:
        if col not in df.columns:
            raise KeyError(f"{source}: missing required column '{col}'.")

    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df = df.dropna(subset=["longitude", "latitude"]).copy()

    # buffer
    if "buffer" not in df.columns:
        df["buffer"] = False
    df["buffer"] = df["buffer"].fillna(False).astype(bool)

    # geography (must exist post cvi_mapping.py, but we handle gracefully)
    if "cvi_zone" in df.columns:
        base = df["cvi_zone"]
    elif "cvi_area" in df.columns:
        base = df["cvi_area"]
    else:
        base = "Non-CVI"

    df["cvi_area"] = pd.Series(base).apply(normalize_zone_name)
    df["cvi_zone"] = df["cvi_area"]  # ALWAYS mirror

    # time
    time_col = detect_time_column(df)
    df["event_time"] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=["event_time"]).copy()

    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs=WGS84,
    )
    gdf["source"] = source
    return gdf


def load_cvi_buffer(cvi_file: str, name_col: str, buffer_m: float) -> gpd.GeoDataFrame:
    """
    Build buffered polygons in METRIC CRS for neighbor membership tests.
    """
    cvi = gpd.read_file(cvi_file)
    if cvi.crs is None:
        cvi = cvi.set_crs(WGS84, allow_override=True)

    if name_col not in cvi.columns:
        raise KeyError(f"Column '{name_col}' not found in CVI file. Available: {list(cvi.columns)}")

    cvi = cvi[[name_col, "geometry"]].copy().to_crs(METRIC)
    cvi["cvi_area"] = cvi[name_col].apply(normalize_zone_name)
    cvi["cvi_zone"] = cvi["cvi_area"]  # mirror

    cvi_buf = cvi[["cvi_area", "cvi_zone", "geometry"]].copy()
    cvi_buf["geometry"] = cvi_buf.geometry.buffer(buffer_m)

    # IMPORTANT: keep name columns and geometry; in METRIC for sjoin predicate
    return cvi_buf


def explode_fractional(pts: gpd.GeoDataFrame, cvi_buf_m: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Non-buffer points: 1 row, weight=1, neighbors_count=1.
    Buffer points: split evenly across {Non-CVI} U {all CVI buffers containing point} U {base assignment if CVI}.
                 (Non-CVI always included.)
    """
    core = pts.loc[~pts["buffer"]].copy()
    buf = pts.loc[pts["buffer"]].copy()

    out_rows: list[dict] = []

    # ---- core points ----
    if len(core) > 0:
        core_df = core.drop(columns=["geometry"], errors="ignore").copy()
        core_df["neighbors_count"] = 1
        core_df["weight"] = 1.0
        # enforce mirror
        core_df["cvi_area"] = core_df["cvi_area"].apply(normalize_zone_name)
        core_df["cvi_zone"] = core_df["cvi_area"]
        out_rows.extend(core_df.to_dict(orient="records"))

    if len(buf) == 0:
        return pd.DataFrame(out_rows)

    # ---- buffer points ----
    # Rename point-side labels to avoid geopandas suffix collisions
    buf_m = buf.to_crs(METRIC).copy()
    if "cvi_area" in buf_m.columns:
        buf_m = buf_m.rename(columns={"cvi_area": "cvi_area_assigned"})
    if "cvi_zone" in buf_m.columns:
        buf_m = buf_m.rename(columns={"cvi_zone": "cvi_zone_assigned"})

    if "index_right" in buf_m.columns:
        buf_m = buf_m.drop(columns=["index_right"])
    if "index_right" in cvi_buf_m.columns:
        cvi_buf_m = cvi_buf_m.drop(columns=["index_right"])

    joined = gpd.sjoin(
        buf_m,
        cvi_buf_m[["cvi_area", "geometry"]],
        how="left",
        predicate="within",
    )

    # point-index -> neighbor zones
    neighbor_map: dict[int, set[str]] = {}
    for idx, sub in joined.groupby(joined.index):
        zones = set(sub["cvi_area"].dropna().apply(normalize_zone_name).tolist())
        neighbor_map[int(idx)] = zones

    buf_plain = buf.drop(columns=["geometry"], errors="ignore").copy()

    for idx, row in buf_plain.iterrows():
        base_area = normalize_zone_name(row.get("cvi_area", row.get("cvi_zone", "Non-CVI")))
        zones = neighbor_map.get(int(idx), set()).copy()

        # Always split among all neighbors including Non-CVI, regardless of original assignment
        zones.add("Non-CVI")
        if base_area != "Non-CVI":
            zones.add(base_area)

        zones_list = sorted(zones)
        k = len(zones_list)
        w = 1.0 / k if k else 1.0

        for z in zones_list:
            rec = row.to_dict()
            rec["cvi_area"] = z
            rec["cvi_zone"] = z  # mirror
            rec["neighbors_count"] = k
            rec["weight"] = float(w)
            out_rows.append(rec)

    return pd.DataFrame(out_rows)


# ------------------------- main -------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Fractional assignment for CVI buffer points (RMS + 911).")
    ap.add_argument("--rms", default="rms_data_cleaned_mapped.csv", help="Mapped RMS CSV input")
    ap.add_argument("--calls", default="911_data_cleaned_mapped.csv", help="Mapped 911 CSV input")
    ap.add_argument("--cvi-file", default="Community_Violence_Intervention_Areas.geojson", help="CVI polygons file")
    ap.add_argument("--name-col", default="CVI_AREA_NAME", help="CVI name column in polygons file")
    ap.add_argument("--buffer-m", type=float, default=7.0, help="Buffer size in meters (must match cvi_mapping.py)")
    ap.add_argument("--out", default="incidents_long_fractional.csv", help="Output long-format CSV")
    args = ap.parse_args()

    print("[fractional_assignment.py] Loading mapped inputs…")
    rms = load_points(args.rms, source="RMS")
    calls = load_points(args.calls, source="911")

    pts = pd.concat([rms, calls], axis=0)
    pts = gpd.GeoDataFrame(pts, geometry="geometry", crs=WGS84)

    print(f"[fractional_assignment.py] Total points loaded: {len(pts)}")
    print(f"[fractional_assignment.py] Buffer-flagged rows (input): {int(pts['buffer'].sum())}")

    print("[fractional_assignment.py] Loading CVI buffers…")
    cvi_buf_m = load_cvi_buffer(args.cvi_file, args.name_col, args.buffer_m)

    print("[fractional_assignment.py] Applying fractional assignment…")
    out_df = explode_fractional(pts, cvi_buf_m)

    # Drop geometry + join debris
    out_df = out_df.drop(columns=["geometry", "index_right"], errors="ignore")

    # Enforce canonical columns for downstream R script
    out_df["cvi_area"] = out_df.get("cvi_area", "Non-CVI").apply(normalize_zone_name)
    out_df["cvi_zone"] = out_df["cvi_area"]  # mirror
    out_df["buffer"] = out_df.get("buffer", False).fillna(False).astype(bool)
    out_df["weight"] = pd.to_numeric(out_df.get("weight", 1.0), errors="coerce").fillna(1.0)

    # Key column ordering (R expects event_time, source, cvi_area, weight at minimum)
    key = ["source", "event_time", "cvi_area", "cvi_zone", "buffer", "neighbors_count", "weight"]
    for col in key:
        if col not in out_df.columns:
            out_df[col] = pd.NA

    rest = [c for c in out_df.columns if c not in key]
    out_df = out_df[key + rest]

    out_path = Path(args.out)
    out_df.to_csv(out_path, index=False)

    print(f"[fractional_assignment.py] Wrote: {out_path} ({len(out_df)} rows)")

    # Quick sanity
    buf_rows = out_df[out_df["buffer"] == True]
    print(f"[fractional_assignment.py] Buffer rows in output: {len(buf_rows)}")
    if len(buf_rows) > 0:
        still_one = buf_rows[buf_rows["weight"] == 1.0]
        print(f"[fractional_assignment.py] Buffer rows with weight==1.0: {len(still_one)}")
        print(f"[fractional_assignment.py] Mean buffer weight: {buf_rows['weight'].mean():.6f}")

    print("\n[fractional_assignment.py] cvi_area value counts:")
    print(out_df["cvi_area"].value_counts())


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[fractional_assignment.py] ERROR: {e}", file=sys.stderr)
        sys.exit(1)

