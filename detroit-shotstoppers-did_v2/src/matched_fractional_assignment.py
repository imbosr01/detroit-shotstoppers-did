#!/usr/bin/env python3
"""
matched_fractional_assignment.py

Purpose
-------
Build a long-format dataset for matched DID analysis by combining:
    - RMS mapped CSV
    - 911 mapped CSV

and applying fractional assignment for buffer points across the matched
analysis zones:

    - 7 treated CVI zones
    - 7 matched control zones

Key output guarantees
---------------------
- event_time (datetime)
- source in {"RMS","911"}
- zone_name (stable analysis zone label)
- zone_type in {"CVI","Control"}
- pair_id identifying matched pair
- pair_cvi_name identifying the treated geography for the pair
- treated in {0,1}
- buffer is a proper boolean
- weight sums to 1 per kept original incident
- neighbors_count indicates how many matched zones each buffer point was split across

Notes
-----
- Non-buffer points are assigned only if they fall within one matched zone polygon.
- Buffer points are split evenly across all matched-zone buffers containing the point.
- Points outside the matched analysis sample are dropped.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
import geopandas as gpd

WGS84 = "EPSG:4326"
METRIC = "EPSG:26917"  # consistent with matched scripts


# -------------------------
# Defaults
# -------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DEFAULT_RMS = BASE_DIR / "src" / "rms_data_cleaned_mapped.csv"
DEFAULT_CALLS = BASE_DIR / "src" / "911_data_cleaned_mapped.csv"
DEFAULT_CVI_FILE = BASE_DIR / "assets" / "Community_Violence_Intervention_Areas.geojson"
DEFAULT_MATCHED_CONTROLS = BASE_DIR / "src" / "matched_controls" / "matched_candidate_controls.geojson"
DEFAULT_MATCHES = BASE_DIR / "src" / "matched_controls" / "cvi_candidate_matches.csv"
DEFAULT_OUT = BASE_DIR / "src" / "matched_controls" / "incidents_long_matched_fractional.csv"

DEFAULT_CVI_NAME_COL = "CVI_AREA_NAME"
DEFAULT_BUFFER_M = 7.0


# -------------------------
# Helpers
# -------------------------

def normalize_zone_name(name) -> str | None:
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return None
    s = str(name).strip()
    return s if s else None


def detect_time_column(df: pd.DataFrame) -> str:
    if "incident_occurred_at" in df.columns:
        return "incident_occurred_at"
    if "called_at" in df.columns:
        return "called_at"
    raise KeyError(
        "No time column found. Expected 'incident_occurred_at' (RMS) or 'called_at' (911)."
    )


def detect_lon_lat_columns(df: pd.DataFrame) -> tuple[str, str]:
    lon_candidates = ["longitude", "Longitude"]
    lat_candidates = ["latitude", "Latitude"]

    lon_col = next((c for c in lon_candidates if c in df.columns), None)
    lat_col = next((c for c in lat_candidates if c in df.columns), None)

    if lon_col is None or lat_col is None:
        raise KeyError(
            f"Could not find longitude/latitude columns. Available columns: {list(df.columns)}"
        )

    return lon_col, lat_col


def load_points(csv_path: str | Path, source: str) -> gpd.GeoDataFrame:
    df = pd.read_csv(csv_path, low_memory=False)

    lon_col, lat_col = detect_lon_lat_columns(df)
    time_col = detect_time_column(df)

    df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")
    df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")

    df = df.dropna(subset=[lon_col, lat_col, time_col]).copy()

    if "buffer" not in df.columns:
        df["buffer"] = False
    df["buffer"] = df["buffer"].fillna(False).astype(bool)

    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
        crs=WGS84,
    )

    gdf["source"] = source
    gdf["event_time"] = gdf[time_col]
    gdf["orig_event_id"] = [f"{source}_{i}" for i in range(len(gdf))]

    return gdf


def build_analysis_zones(
    cvi_file: str | Path,
    matched_controls_file: str | Path,
    matches_file: str | Path,
    cvi_name_col: str,
) -> gpd.GeoDataFrame:
    """
    Build one GeoDataFrame containing the 14 matched analysis zones:
        - treated CVI polygons
        - matched control polygons
    """
    matches = pd.read_csv(matches_file).copy()
    matches["cvi_name"] = matches["cvi_name"].map(normalize_zone_name)
    matches["candidate_id"] = matches["candidate_id"].map(normalize_zone_name)

    # Create pair ids in match-file order
    matches = matches.reset_index(drop=True)
    matches["pair_id"] = [f"PAIR_{i+1:02d}" for i in range(len(matches))]

    # Treated CVIs
    cvis = gpd.read_file(cvi_file)
    if cvis.crs is None:
        cvis = cvis.set_crs(WGS84, allow_override=True)

    cvis = cvis.to_crs(METRIC)

    if cvi_name_col not in cvis.columns:
        raise KeyError(
            f"Column '{cvi_name_col}' not found in CVI file. Available: {list(cvis.columns)}"
        )

    cvis = cvis[[cvi_name_col, "geometry"]].copy()
    cvis["cvi_name"] = cvis[cvi_name_col].map(normalize_zone_name)

    cvi_zones = cvis.merge(
        matches[["cvi_name", "pair_id"]],
        on="cvi_name",
        how="inner",
        validate="1:1",
    ).copy()

    cvi_zones["zone_name"] = cvi_zones["cvi_name"]
    cvi_zones["pair_cvi_name"] = cvi_zones["cvi_name"]
    cvi_zones["zone_type"] = "CVI"
    cvi_zones["treated"] = 1

    cvi_zones = cvi_zones[
        ["zone_name", "zone_type", "treated", "pair_id", "pair_cvi_name", "geometry"]
    ].copy()

    # Matched controls
    ctrls = gpd.read_file(matched_controls_file)
    if ctrls.crs is None:
        ctrls = ctrls.set_crs(WGS84, allow_override=True)

    ctrls = ctrls.to_crs(METRIC)

    if "candidate_id" not in ctrls.columns:
        raise KeyError(
            f"'candidate_id' not found in matched controls file. Available: {list(ctrls.columns)}"
        )

    ctrls["candidate_id"] = ctrls["candidate_id"].map(normalize_zone_name)

    ctrl_zones = ctrls.merge(
        matches[["candidate_id", "cvi_name", "pair_id"]],
        on="candidate_id",
        how="inner",
        validate="1:1",
    ).copy()

    ctrl_zones["zone_name"] = ctrl_zones["candidate_id"]
    ctrl_zones["pair_cvi_name"] = ctrl_zones["cvi_name"]
    ctrl_zones["zone_type"] = "Control"
    ctrl_zones["treated"] = 0

    ctrl_zones = ctrl_zones[
        ["zone_name", "zone_type", "treated", "pair_id", "pair_cvi_name", "geometry"]
    ].copy()

    zones = pd.concat([cvi_zones, ctrl_zones], axis=0, ignore_index=True)
    zones = gpd.GeoDataFrame(zones, geometry="geometry", crs=METRIC)

    print(zones.crs)
    
    return zones


def build_buffered_zones(zones: gpd.GeoDataFrame, buffer_m: float) -> gpd.GeoDataFrame:
    z = zones.copy()
    z["geometry"] = z.geometry.buffer(buffer_m)
    return z


def assign_core_points(
    core_pts: gpd.GeoDataFrame,
    zones_m: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """
    Assign non-buffer points to matched zones using actual polygons.
    Keeps only points within exactly one matched zone.
    """
    if len(core_pts) == 0:
        return pd.DataFrame()

    pts_m = core_pts.to_crs(METRIC).copy()

    joined = gpd.sjoin(
        pts_m,
        zones_m[["zone_name", "zone_type", "treated", "pair_id", "pair_cvi_name", "geometry"]],
        how="inner",
        predicate="within",
    ).copy()

    if joined.empty:
        return pd.DataFrame()

    # If a point somehow joins multiple matched zones, drop it from core assignment.
    counts = joined.groupby("orig_event_id").size().rename("n_matches")
    joined = joined.merge(counts, on="orig_event_id", how="left")
    joined = joined.loc[joined["n_matches"] == 1].copy()

    if joined.empty:
        return pd.DataFrame()

    out = joined.drop(columns=["geometry", "index_right", "n_matches"], errors="ignore").copy()
    out["neighbors_count"] = 1
    out["weight"] = 1.0

    return out


def assign_buffer_points_fractionally(
    buffer_pts: gpd.GeoDataFrame,
    buffered_zones_m: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """
    Split buffer points evenly across all matched-zone buffers containing the point.
    Keeps only points that fall inside at least one matched-zone buffer.
    """
    if len(buffer_pts) == 0:
        return pd.DataFrame()

    pts_m = buffer_pts.to_crs(METRIC).copy()

    joined = gpd.sjoin(
        pts_m,
        buffered_zones_m[["zone_name", "zone_type", "treated", "pair_id", "pair_cvi_name", "geometry"]],
        how="left",
        predicate="within",
    ).copy()

    # Build orig_event_id -> list of matched zones
    out_rows: list[dict] = []

    for orig_event_id, sub in joined.groupby("orig_event_id"):
        sub = sub.dropna(subset=["zone_name"]).copy()
        if sub.empty:
            continue

        sub = sub.drop_duplicates(
            subset=["zone_name", "zone_type", "treated", "pair_id", "pair_cvi_name"]
        ).copy()

        k = len(sub)
        w = 1.0 / k

        base_row = sub.iloc[0].drop(
            labels=["geometry", "index_right"],
            errors="ignore",
        ).to_dict()

        # copy full original point record from each zone row, overwrite zone-specific fields
        for _, zone_row in sub.iterrows():
            rec = zone_row.drop(labels=["geometry", "index_right"], errors="ignore").to_dict()
            rec["neighbors_count"] = k
            rec["weight"] = float(w)
            out_rows.append(rec)

    return pd.DataFrame(out_rows)


# -------------------------
# Main
# -------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Fractional assignment for matched CVI/control zones (RMS + 911)."
    )
    ap.add_argument("--rms", default=DEFAULT_RMS, help="Mapped RMS CSV input")
    ap.add_argument("--calls", default=DEFAULT_CALLS, help="Mapped 911 CSV input")
    ap.add_argument("--cvi-file", default=DEFAULT_CVI_FILE, help="CVI polygons file")
    ap.add_argument("--matched-controls-file", default=DEFAULT_MATCHED_CONTROLS, help="Matched controls GeoJSON")
    ap.add_argument("--matches-file", default=DEFAULT_MATCHES, help="Matched pairs CSV")
    ap.add_argument("--cvi-name-col", default=DEFAULT_CVI_NAME_COL, help="CVI name column in polygons file")
    ap.add_argument("--buffer-m", type=float, default=DEFAULT_BUFFER_M, help="Buffer size in meters")
    ap.add_argument("--out", default=DEFAULT_OUT, help="Output long-format CSV")
    args = ap.parse_args()

    print("[matched_fractional_assignment.py] Loading mapped inputs...")
    rms = load_points(args.rms, source="RMS")
    calls = load_points(args.calls, source="911")

    pts = pd.concat([rms, calls], axis=0, ignore_index=True)
    pts = gpd.GeoDataFrame(pts, geometry="geometry", crs=WGS84)

    print(f"[matched_fractional_assignment.py] Total points loaded: {len(pts)}")
    print(f"[matched_fractional_assignment.py] Buffer-flagged rows (input): {int(pts['buffer'].sum())}")

    print("[matched_fractional_assignment.py] Building matched analysis zones...")
    zones_m = build_analysis_zones(
        cvi_file=args.cvi_file,
        matched_controls_file=args.matched_controls_file,
        matches_file=args.matches_file,
        cvi_name_col=args.cvi_name_col,
    )
    buffered_zones_m = build_buffered_zones(zones_m, args.buffer_m)

    print(f"[matched_fractional_assignment.py] Analysis zones loaded: {len(zones_m)}")

    core_pts = pts.loc[~pts["buffer"]].copy()
    buffer_pts = pts.loc[pts["buffer"]].copy()

    print("[matched_fractional_assignment.py] Assigning non-buffer points...")
    core_out = assign_core_points(core_pts, zones_m)

    print("[matched_fractional_assignment.py] Applying fractional assignment to buffer points...")
    buffer_out = assign_buffer_points_fractionally(buffer_pts, buffered_zones_m)

    out_df = pd.concat([core_out, buffer_out], axis=0, ignore_index=True)

    if out_df.empty:
        raise ValueError("No incidents were assigned to the matched analysis sample.")

    # Clean columns
    out_df = out_df.drop(columns=["geometry", "index_right"], errors="ignore")

    # Canonical columns
    out_df["zone_name"] = out_df["zone_name"].map(normalize_zone_name)
    out_df["zone_type"] = out_df["zone_type"].astype(str)
    out_df["pair_id"] = out_df["pair_id"].astype(str)
    out_df["pair_cvi_name"] = out_df["pair_cvi_name"].map(normalize_zone_name)
    out_df["treated"] = pd.to_numeric(out_df["treated"], errors="coerce").fillna(0).astype(int)
    out_df["buffer"] = out_df.get("buffer", False).fillna(False).astype(bool)
    out_df["weight"] = pd.to_numeric(out_df.get("weight", 1.0), errors="coerce").fillna(1.0)
    out_df["neighbors_count"] = pd.to_numeric(out_df.get("neighbors_count", 1), errors="coerce").fillna(1).astype(int)
    out_df["event_time"] = pd.to_datetime(out_df["event_time"], errors="coerce")
    out_df["month"] = out_df["event_time"].dt.to_period("M").dt.to_timestamp()

    # Order columns
    key_cols = [
        "orig_event_id",
        "source",
        "event_time",
        "month",
        "zone_name",
        "zone_type",
        "pair_id",
        "pair_cvi_name",
        "treated",
        "buffer",
        "neighbors_count",
        "weight",
    ]
    for col in key_cols:
        if col not in out_df.columns:
            out_df[col] = pd.NA

    rest = [c for c in out_df.columns if c not in key_cols]
    out_df = out_df[key_cols + rest]

    # Sanity: weights should sum to 1 per original kept incident
    weight_check = out_df.groupby("orig_event_id", as_index=False)["weight"].sum()
    bad = weight_check.loc[(weight_check["weight"] - 1.0).abs() > 1e-9]
    if not bad.empty:
        raise ValueError(
            f"Weight sums are not 1.0 for {len(bad)} incidents. "
            "Check fractional assignment logic."
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    print(f"[matched_fractional_assignment.py] Wrote: {out_path} ({len(out_df)} rows)")

    print("\n[matched_fractional_assignment.py] Zone counts:")
    print(out_df["zone_name"].value_counts())

    print("\n[matched_fractional_assignment.py] Source counts:")
    print(out_df["source"].value_counts())

    print("\n[matched_fractional_assignment.py] Buffer rows in output:")
    print(int(out_df["buffer"].sum()))

    if int(out_df["buffer"].sum()) > 0:
        print("\n[matched_fractional_assignment.py] Mean buffer weight:")
        print(out_df.loc[out_df["buffer"], "weight"].mean())

    print("\n[matched_fractional_assignment.py] Weight sanity passed.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[matched_fractional_assignment.py] ERROR: {e}", file=sys.stderr)
        sys.exit(1)