#!/usr/bin/env python3
"""
match_cvis_to_controls.py

Match each CVI to one unique candidate non-CVI control zone using:
    - structural / demographic similarity
    - pre-treatment RMS violent crime similarity

Current assumptions:
    - RMS file already contains the homicide + non-fatal shooting incidents
      relevant to ShotStoppers.
    - Candidate zones were built from whole tracts (not partial tract pieces).
    - Candidate demographics are computed from tract-level ACS values.

Outputs:
    - cvi_candidate_matches.csv
    - cvi_candidate_match_features.csv
    - cvi_candidate_distance_matrix.csv
    - matched_candidate_controls.geojson
    - matched_candidate_controls_map.html
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Set

import folium
import geopandas as gpd
import numpy as np
import pandas as pd


# -----------------------------
# Configuration
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
AREA_CRS = "EPSG:26917"

DEFAULT_CVI_FILE = BASE_DIR / "assets" / "Community_Violence_Intervention_Areas.geojson"
DEFAULT_BOUNDARY_FILE = BASE_DIR / "assets" / "City_of_Detroit_Boundary.geojson"

DEFAULT_RMS_FILE = BASE_DIR / "src" / "rms_data_cleaned_mapped.csv"
DEFAULT_CVI_DEMOGRAPHICS_FILE = BASE_DIR / "src" / "cvi_demographics" / "cvi_demographics.csv"
DEFAULT_TRACTS_ACS_FILE = BASE_DIR / "src" / "cvi_demographics" / "detroit_tracts_with_acs.csv"
DEFAULT_CANDIDATE_GEOJSON = BASE_DIR / "src" / "candidate_controls" / "candidate_control_zones.geojson"
DEFAULT_CANDIDATE_MEMBERSHIP_FILE = BASE_DIR / "src" / "candidate_controls" / "candidate_control_zone_membership.csv"

DEFAULT_CVI_NAME_COL = "CVI_AREA_NAME"
DEFAULT_TRACT_GEOID_COL = "GEOID"

DEFAULT_LAT_COL = "latitude"
DEFAULT_LON_COL = "longitude"
DEFAULT_DATE_COL = "incident_occurred_at"

DEFAULT_PRE_START = "2021-01-01"
DEFAULT_PRE_END = "2023-07-31"

DEFAULT_VIOLENCE_WEIGHT = 0.60
DEFAULT_STRUCTURAL_WEIGHT = 0.40


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
        description="Match CVIs to candidate non-CVI control zones."
    )
    parser.add_argument(
        "--rms-file",
        default=DEFAULT_RMS_FILE,
        help=f"Path to RMS mapped CSV (default: {DEFAULT_RMS_FILE}).",
    )
    parser.add_argument(
        "--cvi-file",
        default=DEFAULT_CVI_FILE,
        help=f"Path to CVI GeoJSON (default: {DEFAULT_CVI_FILE}).",
    )
    parser.add_argument(
        "--boundary-file",
        default=DEFAULT_BOUNDARY_FILE,
        help=f"Path to Detroit boundary GeoJSON (default: {DEFAULT_BOUNDARY_FILE}).",
    )
    parser.add_argument(
        "--cvi-demographics-file",
        default=DEFAULT_CVI_DEMOGRAPHICS_FILE,
        help=f"Path to CVI demographics CSV (default: {DEFAULT_CVI_DEMOGRAPHICS_FILE}).",
    )
    parser.add_argument(
        "--tracts-acs-file",
        default=DEFAULT_TRACTS_ACS_FILE,
        help=f"Path to tract ACS CSV (default: {DEFAULT_TRACTS_ACS_FILE}).",
    )
    parser.add_argument(
        "--candidate-geojson",
        default=DEFAULT_CANDIDATE_GEOJSON,
        help=f"Path to candidate control zones GeoJSON (default: {DEFAULT_CANDIDATE_GEOJSON}).",
    )
    parser.add_argument(
        "--candidate-membership-file",
        default=DEFAULT_CANDIDATE_MEMBERSHIP_FILE,
        help=f"Path to candidate membership CSV (default: {DEFAULT_CANDIDATE_MEMBERSHIP_FILE}).",
    )
    parser.add_argument(
        "--cvi-name-col",
        default=DEFAULT_CVI_NAME_COL,
        help=f"CVI name column in CVI GeoJSON (default: {DEFAULT_CVI_NAME_COL}).",
    )
    parser.add_argument(
        "--tract-geoid-col",
        default=DEFAULT_TRACT_GEOID_COL,
        help=f"Tract GEOID column (default: {DEFAULT_TRACT_GEOID_COL}).",
    )
    parser.add_argument(
        "--lat-col",
        default=DEFAULT_LAT_COL,
        help=f"Latitude column in RMS CSV (default: {DEFAULT_LAT_COL}).",
    )
    parser.add_argument(
        "--lon-col",
        default=DEFAULT_LON_COL,
        help=f"Longitude column in RMS CSV (default: {DEFAULT_LON_COL}).",
    )
    parser.add_argument(
        "--date-col",
        default=DEFAULT_DATE_COL,
        help=f"Date column in RMS CSV (default: {DEFAULT_DATE_COL}).",
    )
    parser.add_argument(
        "--pre-start",
        default=DEFAULT_PRE_START,
        help=f"Pre-treatment window start date (default: {DEFAULT_PRE_START}).",
    )
    parser.add_argument(
        "--pre-end",
        default=DEFAULT_PRE_END,
        help=f"Pre-treatment window end date (default: {DEFAULT_PRE_END}).",
    )
    parser.add_argument(
        "--violence-weight",
        type=float,
        default=DEFAULT_VIOLENCE_WEIGHT,
        help=f"Weight on violence distance (default: {DEFAULT_VIOLENCE_WEIGHT}).",
    )
    parser.add_argument(
        "--structural-weight",
        type=float,
        default=DEFAULT_STRUCTURAL_WEIGHT,
        help=f"Weight on structural distance (default: {DEFAULT_STRUCTURAL_WEIGHT}).",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Directory to write outputs.",
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


def standardize_geoid(value: object) -> str | None:
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


def compute_area_sq_miles(gdf: gpd.GeoDataFrame, geom_col: str = "geometry") -> pd.Series:
    square_meters = gdf[geom_col].area
    return square_meters / 2_589_988.110336


def safe_rate(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denom = denominator.where(denominator > 0)
    return numerator / denom


def make_points_gdf(
    df: pd.DataFrame,
    lon_col: str,
    lat_col: str,
    date_col: str,
) -> gpd.GeoDataFrame:
    df = df.copy()

    if lon_col not in df.columns or lat_col not in df.columns:
        raise ValueError(
            f"Latitude/longitude columns not found. Available columns: {list(df.columns)}"
        )
    if date_col not in df.columns:
        raise ValueError(
            f"Date column '{date_col}' not found. Available columns: {list(df.columns)}"
        )

    df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")
    df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    df = df.dropna(subset=[lon_col, lat_col, date_col]).copy()

    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
        crs="EPSG:4326",
    )
    return gdf


def compute_monthly_features(
    joined_points: gpd.GeoDataFrame,
    unit_col: str,
    unit_df: pd.DataFrame,
    date_col: str,
    pre_start: str,
    pre_end: str,
) -> pd.DataFrame:
    """
    Compute pre-treatment monthly violent crime features for each unit.
    """
    pre_start_ts = pd.Timestamp(pre_start)
    pre_end_ts = pd.Timestamp(pre_end)

    months = pd.period_range(pre_start_ts, pre_end_ts, freq="M")
    month_idx = np.arange(len(months))

    base_units = unit_df[[unit_col]].drop_duplicates().copy()

    points = joined_points.copy()
    points = points.loc[
        (points[date_col] >= pre_start_ts) & (points[date_col] <= pre_end_ts)
    ].copy()

    points["month"] = points[date_col].dt.to_period("M")

    monthly = (
        points.groupby([unit_col, "month"])
        .size()
        .rename("events")
        .reset_index()
    )

    full_index = pd.MultiIndex.from_product(
        [base_units[unit_col].tolist(), months],
        names=[unit_col, "month"],
    )

    monthly = (
        monthly.set_index([unit_col, "month"])
        .reindex(full_index, fill_value=0)
        .reset_index()
    )

    def summarize(group: pd.DataFrame) -> pd.Series:
        y = group["events"].to_numpy(dtype=float)

        total_events = float(y.sum())
        avg_monthly_events = float(y.mean())
        monthly_sd = float(y.std(ddof=0))

        if len(y) >= 2:
            slope = float(np.polyfit(month_idx, y, 1)[0])
        else:
            slope = 0.0

        return pd.Series(
            {
                "pre_total_events": total_events,
                "pre_avg_monthly_events": avg_monthly_events,
                "pre_monthly_sd": monthly_sd,
                "pre_trend_slope": slope,
            }
        )

    features = (
        monthly.groupby(unit_col, as_index=False)
        .apply(summarize, include_groups=False)
        .reset_index()
    )

    if "level_0" in features.columns:
        features = features.drop(columns=["level_0"])

    return features


def compute_candidate_demographics(
    candidate_membership: pd.DataFrame,
    tracts_acs: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute candidate control demographics by aggregating whole-tract ACS values.
    """
    tracts_acs = tracts_acs.copy()
    tracts_acs["GEOID"] = tracts_acs["GEOID"].map(standardize_geoid)

    candidate_membership = candidate_membership.copy()
    candidate_membership["GEOID"] = candidate_membership["GEOID"].map(standardize_geoid)

    merged = candidate_membership.merge(
        tracts_acs,
        on="GEOID",
        how="left",
        validate="m:1",
    )

    area_col = "tract_area_sq_miles_x" if "tract_area_sq_miles_x" in merged.columns else "tract_area_sq_miles"

    candidate_demo = (
        merged.groupby("candidate_id", as_index=False)
        .agg(
            area_sq_miles=(area_col, "sum"),
            total_population=("total_population", "sum"),
            poverty_universe=("poverty_universe", "sum"),
            below_poverty=("below_poverty", "sum"),
            housing_units=("housing_units", "sum"),
            occupied_units=("occupied_units", "sum"),
            vacant_units=("vacant_units", "sum"),
            n_source_tracts=("GEOID", "nunique"),
        )
        .copy()
    )

    candidate_demo["poverty_rate"] = safe_rate(
        candidate_demo["below_poverty"],
        candidate_demo["poverty_universe"],
    )
    candidate_demo["vacancy_rate"] = safe_rate(
        candidate_demo["vacant_units"],
        candidate_demo["housing_units"],
    )
    candidate_demo["population_density_per_sq_mile"] = safe_rate(
        candidate_demo["total_population"],
        candidate_demo["area_sq_miles"],
    )

    return candidate_demo


def standardize_features(
    cvi_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    feature_cols: List[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Standardize features across the combined CVI + candidate pool.
    """
    combined = pd.concat(
        [
            cvi_df[feature_cols].assign(_source="cvi"),
            candidate_df[feature_cols].assign(_source="candidate"),
        ],
        axis=0,
        ignore_index=True,
    )

    means = combined[feature_cols].mean()
    stds = combined[feature_cols].std(ddof=0).replace(0, 1)

    cvi_scaled = (cvi_df[feature_cols] - means) / stds
    candidate_scaled = (candidate_df[feature_cols] - means) / stds

    return cvi_scaled, candidate_scaled


def pairwise_euclidean(
    left: pd.DataFrame,
    right: pd.DataFrame,
) -> np.ndarray:
    """
    Compute pairwise Euclidean distances between rows of left and right.
    """
    left_arr = left.to_numpy(dtype=float)
    right_arr = right.to_numpy(dtype=float)

    dists = np.sqrt(
        ((left_arr[:, None, :] - right_arr[None, :, :]) ** 2).sum(axis=2)
    )
    return dists


def make_json_safe_subset(
    gdf: gpd.GeoDataFrame,
    keep_cols: List[str],
) -> gpd.GeoDataFrame:
    cols = [c for c in keep_cols if c in gdf.columns] + [gdf.geometry.name]
    out = gdf[cols].copy()
    for col in out.columns:
        if col == out.geometry.name:
            continue
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = out[col].astype(str)
    return out


def make_color_lookup(ids: List[str]) -> Dict[str, str]:
    palette = [
        "#1f77b4",
        "#2ca02c",
        "#ff7f0e",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#17becf",
        "#bcbd22",
        "#7f7f7f",
        "#393b79",
    ]
    unique_ids = sorted(set(ids))
    return {cid: palette[i % len(palette)] for i, cid in enumerate(unique_ids)}


def make_html_map(
    boundary: gpd.GeoDataFrame,
    cvis: gpd.GeoDataFrame,
    matched_candidates: gpd.GeoDataFrame,
    out_html: Path,
) -> None:
    boundary_wgs = make_json_safe_subset(boundary.to_crs(epsg=4326), [])
    cvis_wgs = make_json_safe_subset(
        cvis.to_crs(epsg=4326),
        ["cvi_name", "matched_candidate_id"],
    )
    matched_wgs = make_json_safe_subset(
        matched_candidates.to_crs(epsg=4326),
        ["candidate_id", "matched_cvi_name", "total_distance"],
    )

    center_geom = (
        boundary_wgs.geometry.union_all()
        if hasattr(boundary_wgs.geometry, "union_all")
        else boundary_wgs.unary_union
    )
    center = center_geom.centroid

    m = folium.Map(location=[center.y, center.x], zoom_start=11, tiles="CartoDB positron")

    focus_css = """
    <style>
    .leaflet-interactive:focus,
    path.leaflet-interactive:focus,
    svg path:focus,
    .leaflet-container a:focus,
    .leaflet-container *:focus {
        outline: none !important;
        box-shadow: none !important;
    }
    </style>
    """
    m.get_root().html.add_child(folium.Element(focus_css))

    color_lookup = make_color_lookup(matched_wgs["candidate_id"].dropna().tolist())

    folium.GeoJson(
        data=json.loads(boundary_wgs.to_json()),
        name="Detroit Boundary",
        style_function=lambda _: {
            "color": "black",
            "weight": 2,
            "fillOpacity": 0,
        },
    ).add_to(m)

    folium.GeoJson(
        data=json.loads(cvis_wgs.to_json()),
        name="CVIs",
        style_function=lambda _: {
            "color": "#d62728",
            "weight": 2,
            "fillColor": "#d62728",
            "fillOpacity": 0.30,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=[c for c in ["cvi_name", "matched_candidate_id"] if c in cvis_wgs.columns],
            aliases=["CVI:", "Matched candidate:"],
            sticky=False,
        ),
    ).add_to(m)

    for candidate_id in sorted(matched_wgs["candidate_id"].dropna().unique()):
        this_gdf = matched_wgs.loc[matched_wgs["candidate_id"] == candidate_id].copy()
        color = color_lookup.get(candidate_id, "#1f77b4")
        matched_cvi_name = this_gdf["matched_cvi_name"].iloc[0] if "matched_cvi_name" in this_gdf.columns else ""

        feature_group = folium.FeatureGroup(
            name=f"{candidate_id} → {matched_cvi_name}",
            show=True,
        )

        folium.GeoJson(
            data=json.loads(this_gdf.to_json()),
            style_function=lambda _, color=color: {
                "color": color,
                "weight": 2,
                "fillColor": color,
                "fillOpacity": 0.28,
            },
            tooltip=folium.GeoJsonTooltip(
                fields=[c for c in ["candidate_id", "matched_cvi_name", "total_distance"] if c in this_gdf.columns],
                aliases=["Candidate:", "Matched to CVI:", "Distance:"],
                sticky=False,
            ),
        ).add_to(feature_group)

        feature_group.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    m.save(str(out_html))


# -----------------------------
# New overlap + constrained match helpers
# -----------------------------

def build_candidate_overlap_lookup(candidate_gdf: gpd.GeoDataFrame) -> Dict[str, Set[str]]:
    """
    Build a lookup of candidate zones that overlap each other with positive area.
    Touching at edges/points is allowed; only true area overlap is flagged.
    """
    candidate_gdf = candidate_gdf[["candidate_id", candidate_gdf.geometry.name]].copy()
    overlap_lookup: Dict[str, Set[str]] = {cid: set() for cid in candidate_gdf["candidate_id"]}

    for i in range(len(candidate_gdf)):
        cid_i = candidate_gdf.iloc[i]["candidate_id"]
        geom_i = candidate_gdf.iloc[i].geometry

        for j in range(i + 1, len(candidate_gdf)):
            cid_j = candidate_gdf.iloc[j]["candidate_id"]
            geom_j = candidate_gdf.iloc[j].geometry

            inter = geom_i.intersection(geom_j)
            if not inter.is_empty and inter.area > 0:
                overlap_lookup[cid_i].add(cid_j)
                overlap_lookup[cid_j].add(cid_i)

    return overlap_lookup


def greedy_nonoverlap_match(
    cvi_features: pd.DataFrame,
    candidate_features: pd.DataFrame,
    total_dist: np.ndarray,
    violence_dist: np.ndarray,
    structural_dist: np.ndarray,
    overlap_lookup: Dict[str, Set[str]],
) -> pd.DataFrame:
    """
    Greedy constrained matching:
      - one candidate per CVI
      - candidate used once
      - selected candidates may touch but may not overlap in area
      - chooses lowest-distance pairs first
    """
    pair_rows = []

    cvi_names = cvi_features["cvi_name"].tolist()
    candidate_ids = candidate_features["candidate_id"].tolist()

    for i, cvi_name in enumerate(cvi_names):
        for j, candidate_id in enumerate(candidate_ids):
            pair_rows.append(
                {
                    "cvi_name": cvi_name,
                    "candidate_id": candidate_id,
                    "total_distance": float(total_dist[i, j]),
                    "violence_distance": float(violence_dist[i, j]),
                    "structural_distance": float(structural_dist[i, j]),
                }
            )

    pair_df = pd.DataFrame(pair_rows).sort_values("total_distance").reset_index(drop=True)

    selected = []
    matched_cvis = set()
    used_candidates = set()

    for _, row in pair_df.iterrows():
        cvi_name = row["cvi_name"]
        candidate_id = row["candidate_id"]

        if cvi_name in matched_cvis:
            continue
        if candidate_id in used_candidates:
            continue

        overlaps_selected = any(
            candidate_id in overlap_lookup.get(chosen, set())
            for chosen in used_candidates
        )
        if overlaps_selected:
            continue

        selected.append(row.to_dict())
        matched_cvis.add(cvi_name)
        used_candidates.add(candidate_id)

        if len(matched_cvis) == len(cvi_names):
            break

    matches_df = pd.DataFrame(selected)

    if len(matches_df) < len(cvi_names):
        raise ValueError(
            "Could not find a full set of non-overlapping matched controls. "
            "Try relaxing candidate-generation constraints or increasing the candidate pool."
        )

    return matches_df.sort_values("cvi_name").reset_index(drop=True)


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    out_dir = ensure_output_dir(args.out_dir)

    weight_sum = args.violence_weight + args.structural_weight
    if not np.isclose(weight_sum, 1.0):
        logging.warning(
            f"Violence + structural weights sum to {weight_sum:.3f}, not 1.0. "
            "They will still be used as provided."
        )

    logging.info("Loading spatial and tabular inputs...")
    cvis = load_geodata(args.cvi_file)
    boundary = load_geodata(args.boundary_file)
    candidate_gdf = load_geodata(args.candidate_geojson)

    cvi_demo = pd.read_csv(args.cvi_demographics_file)
    tracts_acs = pd.read_csv(args.tracts_acs_file)
    candidate_membership = pd.read_csv(args.candidate_membership_file)
    rms = pd.read_csv(args.rms_file)

    if args.cvi_name_col not in cvis.columns:
        raise ValueError(
            f"CVI name column '{args.cvi_name_col}' not found in CVI file. "
            f"Available columns: {list(cvis.columns)}"
        )

    cvis = cvis[[args.cvi_name_col, cvis.geometry.name]].copy()
    cvis["cvi_name"] = cvis[args.cvi_name_col].astype(str).str.strip()
    cvis = cvis[["cvi_name", cvis.geometry.name]].copy()

    candidate_gdf = candidate_gdf.copy()
    if "candidate_id" not in candidate_gdf.columns:
        raise ValueError(
            f"'candidate_id' not found in candidate geojson. Available columns: {list(candidate_gdf.columns)}"
        )

    cvis = cvis.to_crs(AREA_CRS)
    boundary = boundary.to_crs(AREA_CRS)
    candidate_gdf = candidate_gdf.to_crs(AREA_CRS)

    logging.info("Computing candidate structural features...")
    candidate_demo = compute_candidate_demographics(candidate_membership, tracts_acs)

    cvi_struct = cvi_demo.copy().rename(
        columns={
            "cvi_area_sq_miles": "area_sq_miles",
        }
    )

    cvi_struct = cvi_struct[
        [
            "cvi_name",
            "area_sq_miles",
            "total_population",
            "poverty_rate",
            "vacancy_rate",
            "population_density_per_sq_mile",
        ]
    ].copy()

    candidate_struct = candidate_demo[
        [
            "candidate_id",
            "area_sq_miles",
            "total_population",
            "poverty_rate",
            "vacancy_rate",
            "population_density_per_sq_mile",
        ]
    ].copy()

    logging.info("Preparing RMS incident points...")
    rms_points = make_points_gdf(
        rms,
        lon_col=args.lon_col,
        lat_col=args.lat_col,
        date_col=args.date_col,
    ).to_crs(AREA_CRS)

    logging.info("Computing pre-treatment RMS features for CVIs...")
    rms_cvi = gpd.sjoin(
        rms_points,
        cvis[["cvi_name", cvis.geometry.name]],
        how="inner",
        predicate="within",
    )

    cvi_violence = compute_monthly_features(
        joined_points=rms_cvi,
        unit_col="cvi_name",
        unit_df=cvis[["cvi_name"]],
        date_col=args.date_col,
        pre_start=args.pre_start,
        pre_end=args.pre_end,
    )

    logging.info("Computing pre-treatment RMS features for candidate controls...")
    rms_candidate = gpd.sjoin(
        rms_points,
        candidate_gdf[["candidate_id", candidate_gdf.geometry.name]],
        how="inner",
        predicate="within",
    )

    candidate_violence = compute_monthly_features(
        joined_points=rms_candidate,
        unit_col="candidate_id",
        unit_df=candidate_gdf[["candidate_id"]],
        date_col=args.date_col,
        pre_start=args.pre_start,
        pre_end=args.pre_end,
    )

    cvi_features = cvi_struct.merge(cvi_violence, on="cvi_name", how="left", validate="1:1")
    candidate_features = candidate_struct.merge(candidate_violence, on="candidate_id", how="left", validate="1:1")

    violence_cols_base = [
        "pre_total_events",
        "pre_avg_monthly_events",
        "pre_monthly_sd",
        "pre_trend_slope",
    ]
    for col in violence_cols_base:
        cvi_features[col] = cvi_features[col].fillna(0.0)
        candidate_features[col] = candidate_features[col].fillna(0.0)

    cvi_features["pre_avg_monthly_rate_per_1000"] = safe_rate(
        cvi_features["pre_avg_monthly_events"] * 1000,
        cvi_features["total_population"],
    ).fillna(0.0)

    candidate_features["pre_avg_monthly_rate_per_1000"] = safe_rate(
        candidate_features["pre_avg_monthly_events"] * 1000,
        candidate_features["total_population"],
    ).fillna(0.0)

    structural_cols = [
        "area_sq_miles",
        "total_population",
        "poverty_rate",
        "vacancy_rate",
        "population_density_per_sq_mile",
    ]
    violence_cols = [
        "pre_avg_monthly_events",
        "pre_avg_monthly_rate_per_1000",
        "pre_monthly_sd",
        "pre_trend_slope",
    ]

    logging.info("Standardizing feature groups...")
    cvi_struct_scaled, candidate_struct_scaled = standardize_features(
        cvi_features, candidate_features, structural_cols
    )
    cvi_viol_scaled, candidate_viol_scaled = standardize_features(
        cvi_features, candidate_features, violence_cols
    )

    logging.info("Computing pairwise distances...")
    structural_dist = pairwise_euclidean(cvi_struct_scaled, candidate_struct_scaled)
    violence_dist = pairwise_euclidean(cvi_viol_scaled, candidate_viol_scaled)

    total_dist = (
        args.structural_weight * structural_dist
        + args.violence_weight * violence_dist
    )

    distance_matrix = pd.DataFrame(
        total_dist,
        index=cvi_features["cvi_name"].tolist(),
        columns=candidate_features["candidate_id"].tolist(),
    )

    distance_matrix_path = out_dir / "cvi_candidate_distance_matrix.csv"
    try:
        distance_matrix.to_csv(distance_matrix_path)
    except PermissionError:
        raise PermissionError(
            f"Could not write to {distance_matrix_path}. "
            "Close the file if it is open in Excel or another program, then rerun."
        )
    logging.info(f"Wrote: {distance_matrix_path}")

    logging.info("Building candidate overlap lookup...")
    overlap_lookup = build_candidate_overlap_lookup(candidate_gdf)

    logging.info("Selecting one-to-one non-overlapping matches...")
    matches_df = greedy_nonoverlap_match(
        cvi_features=cvi_features,
        candidate_features=candidate_features,
        total_dist=total_dist,
        violence_dist=violence_dist,
        structural_dist=structural_dist,
        overlap_lookup=overlap_lookup,
    )

    matches_path = out_dir / "cvi_candidate_matches.csv"
    matches_df.to_csv(matches_path, index=False)
    logging.info(f"Wrote: {matches_path}")

    comparison_rows = []
    for _, row in matches_df.iterrows():
        cvi_row = cvi_features.loc[cvi_features["cvi_name"] == row["cvi_name"]].iloc[0]
        cand_row = candidate_features.loc[candidate_features["candidate_id"] == row["candidate_id"]].iloc[0]

        comp = {
            "cvi_name": row["cvi_name"],
            "candidate_id": row["candidate_id"],
            "total_distance": row["total_distance"],
            "violence_distance": row["violence_distance"],
            "structural_distance": row["structural_distance"],
        }

        for col in structural_cols + violence_cols:
            comp[f"cvi_{col}"] = cvi_row[col]
            comp[f"candidate_{col}"] = cand_row[col]
            comp[f"abs_diff_{col}"] = abs(cvi_row[col] - cand_row[col])

        comparison_rows.append(comp)

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_path = out_dir / "cvi_candidate_match_features.csv"
    comparison_df.to_csv(comparison_path, index=False)
    logging.info(f"Wrote: {comparison_path}")

    matched_candidate_ids = matches_df["candidate_id"].tolist()
    matched_candidates = candidate_gdf[candidate_gdf["candidate_id"].isin(matched_candidate_ids)].copy()

    matched_candidates = matched_candidates.merge(
        matches_df[["cvi_name", "candidate_id", "total_distance"]],
        on="candidate_id",
        how="left",
        validate="1:1",
    ).rename(columns={"cvi_name": "matched_cvi_name"})

    matched_geojson_path = out_dir / "matched_candidate_controls.geojson"
    matched_candidates.to_file(matched_geojson_path, driver="GeoJSON")
    logging.info(f"Wrote: {matched_geojson_path}")

    cvis_map = cvis.merge(
        matches_df[["cvi_name", "candidate_id"]],
        on="cvi_name",
        how="left",
        validate="1:1",
    ).rename(columns={"candidate_id": "matched_candidate_id"})

    map_path = out_dir / "matched_candidate_controls_map.html"
    make_html_map(
        boundary=boundary,
        cvis=cvis_map,
        matched_candidates=matched_candidates,
        out_html=map_path,
    )
    logging.info(f"Wrote: {map_path}")

    logging.info("Final matched pairs:")
    for _, row in matches_df.iterrows():
        logging.info(
            f"{row['cvi_name']} -> {row['candidate_id']} "
            f"(total={row['total_distance']:.3f}, "
            f"violence={row['violence_distance']:.3f}, "
            f"structural={row['structural_distance']:.3f})"
        )

    logging.info("Done.")


if __name__ == "__main__":
    main()