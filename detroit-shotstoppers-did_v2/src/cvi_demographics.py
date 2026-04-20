#!/usr/bin/env python3
"""
cvi_demographics.py

Estimate CVI-level demographics by intersecting Community Violence Intervention
(CVI) polygons with Detroit census tracts and area-weighting tract-level ACS
count variables.

Outputs:
    - cvi_tract_overlap.csv
    - cvi_tract_overlap.geojson
    - cvi_demographics.csv
    - detroit_tracts_with_acs.csv

Expected workflow position:
    Run this after cvi_mapping.py, once CVI polygons are finalized.

Notes:
    - Request U.S. Census API Key from here: https://api.census.gov/data/key_signup.html
    - This script area-weights tract COUNT variables into CVIs.
    - Derived rates (poverty_rate, vacancy_rate, occupancy_rate) are computed
      after aggregation.
    - ACS data are pulled for all Wayne County tracts, then spatially filtered
      to Detroit using the supplied Detroit boundary file.

Usage:
 python cvi_demographics.py \
   --out-dir cvi_demographics \
   --api-key "YOUR API KEY"
   --verbose
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Optional

import geopandas as gpd
import pandas as pd
from census import Census

# -----------------------------
# Configuration
# -----------------------------

STATE_FIPS = "26"    # Michigan
COUNTY_FIPS = "163"  # Wayne County

ACS_FIELDS = {
    "total_population": "B01003_001E",
    "poverty_universe": "B17001_001E",
    "below_poverty": "B17001_002E",
    "housing_units": "B25001_001E",
    "occupied_units": "B25002_002E",
    "vacant_units": "B25002_003E",
}

DEFAULT_ACS_YEAR = 2023
AREA_CRS = "EPSG:26917"

BASE_DIR = Path(__file__).resolve().parent.parent

DEFAULT_TRACT_FILE = BASE_DIR / "assets/Detroit_Census_Tracts.geojson"
DEFAULT_CVI_FILE = BASE_DIR / "assets/Community_Violence_Intervention_Areas.geojson"
DEFAULT_BOUNDARY_FILE = BASE_DIR / "assets/City_of_Detroit_Boundary.geojson"

DEFAULT_CVI_NAME_COL = "CVI_AREA_NAME"
DEFAULT_TRACT_GEOID_COL = "GEOID"


# -----------------------------
# Logging
# -----------------------------

def setup_logging(verbose: bool) -> None:
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


# -----------------------------
# Argument parsing
# -----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate CVI demographics by intersecting CVIs with census tracts."
    )
    parser.add_argument(
        "--tract-file",
        default=DEFAULT_TRACT_FILE,
        help=f"Path to Detroit census tract GeoJSON (default: {DEFAULT_TRACT_FILE}).",
    )
    parser.add_argument(
        "--cvi-file",
        default=DEFAULT_CVI_FILE,
        help=f"Path to CVI GeoJSON (default: {DEFAULT_CVI_FILE}).",
    )
    parser.add_argument(
        "--boundary-file",
        default=DEFAULT_BOUNDARY_FILE,
        help="City of Detroit boundary GeoJSON"
    )

    parser.add_argument(
        "--out-dir",
        required=True,
        help="Directory to write outputs.",
    )
    parser.add_argument(
    "--api-key",
    required=True,
    help="Census API key",
    )
    parser.add_argument(
        "--acs-year",
        type=int,
        default=DEFAULT_ACS_YEAR,
        help=f"ACS 5-year year to query (default: {DEFAULT_ACS_YEAR}).",
    )
    parser.add_argument(
        "--cvi-name-col",
        default=DEFAULT_CVI_NAME_COL,
        help=f"CVI name/id column in the CVI file (default: {DEFAULT_CVI_NAME_COL}).",
    )
    parser.add_argument(
        "--tract-geoid-col",
        default=DEFAULT_TRACT_GEOID_COL,
        help=f"Tract GEOID column in the tract file (default: {DEFAULT_TRACT_GEOID_COL}).",
    )
    parser.add_argument(
        "--keep-all-tracts-touching-boundary",
        action="store_true",
        help=(
            "If set, keep tracts that intersect Detroit boundary. "
            "Default behavior uses centroid-within-boundary filtering."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )
    return parser.parse_args()


# -----------------------------
# Helpers
# -----------------------------

def ensure_output_dir(path: str | Path) -> Path:
    out_dir = Path(path)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def standardize_geoid(value: object) -> Optional[str]:
    if pd.isna(value):
        return None
    s = str(value).strip()
    if not s:
        return None
    s = s.replace(".0", "")
    digits = "".join(ch for ch in s if ch.isdigit())
    return digits if digits else None


def load_geodata(path: str | Path) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)
    if gdf.empty:
        raise ValueError(f"No features found in file: {path}")
    if gdf.crs is None:
        raise ValueError(f"File has no CRS defined: {path}")
    return gdf


def fetch_acs_tract_data(api_key: str, year: int) -> pd.DataFrame:
    """
    Pull ACS 5-year data for all Wayne County tracts, then create full GEOID.
    """
    logging.info(f"Requesting ACS 5-year tract data for Wayne County, year={year}")
    c = Census(api_key, year=year)

    raw = c.acs5.state_county_tract(
        fields=list(ACS_FIELDS.values()),
        state_fips=STATE_FIPS,
        county_fips=COUNTY_FIPS,
        tract="*",
    )
    acs = pd.DataFrame(raw)

    rename_map = {v: k for k, v in ACS_FIELDS.items()}
    acs = acs.rename(columns=rename_map)

    acs["GEOID"] = (
        acs["state"].astype(str).str.zfill(2)
        + acs["county"].astype(str).str.zfill(3)
        + acs["tract"].astype(str).str.zfill(6)
    )

    keep_cols = ["GEOID"] + list(ACS_FIELDS.keys())
    acs = acs[keep_cols].copy()

    for col in ACS_FIELDS.keys():
        acs[col] = pd.to_numeric(acs[col], errors="coerce")

    logging.info(f"Fetched ACS rows: {len(acs):,}")
    return acs


def spatial_filter_to_detroit(
    tracts: gpd.GeoDataFrame,
    boundary: gpd.GeoDataFrame,
    keep_intersections: bool = False,
) -> gpd.GeoDataFrame:
    """
    Filter tract features to Detroit using either:
      - centroid within boundary (default)
      - any intersection with boundary (optional)
    """
    boundary_union = (
        boundary.geometry.union_all()
        if hasattr(boundary.geometry, "union_all")
        else boundary.unary_union
    )

    if keep_intersections:
        mask = tracts.intersects(boundary_union)
        filtered = tracts.loc[mask].copy()
        logging.info(
            f"Filtered tracts by intersection with Detroit boundary: {len(filtered):,} retained"
        )
        return filtered

    centroids = tracts.geometry.centroid
    mask = centroids.within(boundary_union)
    filtered = tracts.loc[mask].copy()
    logging.info(
        f"Filtered tracts by centroid within Detroit boundary: {len(filtered):,} retained"
    )
    return filtered


def compute_area_sq_miles(gdf: gpd.GeoDataFrame, geom_col: str = "geometry") -> pd.Series:
    """
    Assumes projected CRS in meters. Converts square meters to square miles.
    """
    square_meters = gdf[geom_col].area
    return square_meters / 2_589_988.110336


def safe_rate(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denom = denominator.where(denominator > 0)
    return numerator / denom


# -----------------------------
# Main processing
# -----------------------------

def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    out_dir = ensure_output_dir(args.out_dir)

    api_key = args.api_key
    if not api_key:
        raise ValueError("Census API key is required. Pass it with --api-key.")

    logging.info("Loading geographic files...")
    tracts = load_geodata(args.tract_file)
    cvis = load_geodata(args.cvi_file)
    boundary = load_geodata(args.boundary_file)

    if args.cvi_name_col not in cvis.columns:
        raise ValueError(
            f"CVI name column '{args.cvi_name_col}' not found in CVI file. "
            f"Available columns: {list(cvis.columns)}"
        )

    if args.tract_geoid_col not in tracts.columns:
        raise ValueError(
            f"Tract GEOID column '{args.tract_geoid_col}' not found in tract file. "
            f"Available columns: {list(tracts.columns)}"
        )

    logging.info(f"Using CVI name column: {args.cvi_name_col}")
    logging.info(f"Using tract GEOID column: {args.tract_geoid_col}")

    # Keep only tract GEOID source column + geometry before standardizing/dissolving
    tracts = tracts[[args.tract_geoid_col, tracts.geometry.name]].copy()
    tracts["GEOID"] = tracts[args.tract_geoid_col].map(standardize_geoid)

    missing_geoids = tracts["GEOID"].isna().sum()
    if missing_geoids > 0:
        logging.warning(
            f"{missing_geoids:,} tract rows have missing GEOID after standardization."
        )
        tracts = tracts.loc[tracts["GEOID"].notna()].copy()

    # Check for duplicate GEOIDs and dissolve if needed
    dup_counts = tracts["GEOID"].value_counts()
    dups = dup_counts[dup_counts > 1]
    if not dups.empty:
        logging.warning(
            f"{len(dups):,} GEOIDs appear more than once in tract file. "
            "Dissolving tract geometries by GEOID before ACS merge."
        )
        tracts = tracts[["GEOID", tracts.geometry.name]].dissolve(by="GEOID", as_index=False)
    else:
        tracts = tracts[["GEOID", tracts.geometry.name]].copy()

    cvis = cvis.copy()
    cvis["cvi_name"] = cvis[args.cvi_name_col].astype(str).str.strip()

    logging.info(f"Projecting all layers to {AREA_CRS} for area calculations...")
    tracts = tracts.to_crs(AREA_CRS)
    cvis = cvis.to_crs(AREA_CRS)
    boundary = boundary.to_crs(AREA_CRS)

    tracts = spatial_filter_to_detroit(
        tracts,
        boundary,
        keep_intersections=args.keep_all_tracts_touching_boundary,
    )

    acs = fetch_acs_tract_data(api_key=api_key, year=args.acs_year)

    tracts = tracts.merge(acs, on="GEOID", how="left", validate="1:1")
    missing_acs = tracts["total_population"].isna().sum()
    if missing_acs > 0:
        logging.warning(
            f"{missing_acs:,} tracts are missing ACS data after merge. "
            "They will still be kept, but weighted values may be NaN."
        )

    detroit_tracts_csv = out_dir / "detroit_tracts_with_acs.csv"
    tracts.drop(columns=[tracts.geometry.name]).to_csv(detroit_tracts_csv, index=False)
    logging.info(f"Wrote: {detroit_tracts_csv}")

    tracts["tract_area_sq_miles"] = compute_area_sq_miles(tracts)
    cvis["cvi_area_sq_miles"] = compute_area_sq_miles(cvis)

    tract_keep = ["GEOID", "tract_area_sq_miles"] + list(ACS_FIELDS.keys()) + [tracts.geometry.name]
    cvi_keep = ["cvi_name", "cvi_area_sq_miles", cvis.geometry.name]

    tracts_for_overlay = tracts[tract_keep].copy()
    cvis_for_overlay = cvis[cvi_keep].copy()

    logging.info("Computing CVI x tract intersection...")
    overlap = gpd.overlay(cvis_for_overlay, tracts_for_overlay, how="intersection")

    if overlap.empty:
        raise ValueError(
            "No intersections found between CVIs and tracts. "
            "Check CRS, extent, and source files."
        )

    overlap["intersection_area_sq_miles"] = compute_area_sq_miles(overlap)
    overlap["overlap_share"] = (
        overlap["intersection_area_sq_miles"] / overlap["tract_area_sq_miles"]
    ).clip(lower=0, upper=1)

    for col in ACS_FIELDS.keys():
        overlap[f"{col}_weighted"] = overlap["overlap_share"] * overlap[col]

    overlap_csv = out_dir / "cvi_tract_overlap.csv"
    overlap_geojson = out_dir / "cvi_tract_overlap.geojson"

    overlap.drop(columns=[overlap.geometry.name]).to_csv(overlap_csv, index=False)
    overlap.to_file(overlap_geojson, driver="GeoJSON")

    logging.info(f"Wrote: {overlap_csv}")
    logging.info(f"Wrote: {overlap_geojson}")

    weighted_cols = [f"{col}_weighted" for col in ACS_FIELDS.keys()]

    cvi_demo = (
        overlap.groupby("cvi_name", as_index=False)
        .agg(
            cvi_area_sq_miles=("cvi_area_sq_miles", "first"),
            intersecting_tracts=("GEOID", "nunique"),
            intersection_area_sq_miles=("intersection_area_sq_miles", "sum"),
            **{col: (col, "sum") for col in weighted_cols},
        )
        .copy()
    )

    rename_back = {f"{col}_weighted": col for col in ACS_FIELDS.keys()}
    cvi_demo = cvi_demo.rename(columns=rename_back)

    cvi_demo["poverty_rate"] = safe_rate(
        cvi_demo["below_poverty"],
        cvi_demo["poverty_universe"],
    )
    cvi_demo["vacancy_rate"] = safe_rate(
        cvi_demo["vacant_units"],
        cvi_demo["housing_units"],
    )
    cvi_demo["occupancy_rate"] = safe_rate(
        cvi_demo["occupied_units"],
        cvi_demo["housing_units"],
    )
    cvi_demo["population_density_per_sq_mile"] = safe_rate(
        cvi_demo["total_population"],
        cvi_demo["cvi_area_sq_miles"],
    )

    round_cols = [
        "cvi_area_sq_miles",
        "intersection_area_sq_miles",
        "total_population",
        "poverty_universe",
        "below_poverty",
        "housing_units",
        "occupied_units",
        "vacant_units",
        "poverty_rate",
        "vacancy_rate",
        "occupancy_rate",
        "population_density_per_sq_mile",
    ]
    for col in round_cols:
        if col in cvi_demo.columns:
            cvi_demo[col] = cvi_demo[col].round(6 if "rate" in col or "sq_miles" in col else 2)

    cvi_demo_csv = out_dir / "cvi_demographics.csv"
    cvi_demo.to_csv(cvi_demo_csv, index=False)
    logging.info(f"Wrote: {cvi_demo_csv}")

    logging.info("Sanity check: CVI estimated areas")
    for _, row in cvi_demo[["cvi_name", "cvi_area_sq_miles", "intersection_area_sq_miles"]].iterrows():
        logging.info(
            f"{row['cvi_name']}: original_area={row['cvi_area_sq_miles']:.3f} sq mi | "
            f"intersected_area={row['intersection_area_sq_miles']:.3f} sq mi"
        )

    logging.info("Done.")

if __name__ == "__main__":
    main()
