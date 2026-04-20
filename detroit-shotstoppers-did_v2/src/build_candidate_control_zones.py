#!/usr/bin/env python3
"""
build_candidate_control_zones.py

Build contiguous non-CVI candidate control zones from Detroit census tracts.

Current approach:
    - Load Detroit census tracts, CVI polygons, and Detroit boundary
    - Restrict tracts to Detroit
    - Exclude any tract that intersects a CVI polygon
    - Build tract adjacency among remaining tracts
    - Generate contiguous tract clusters
    - Keep clusters whose total area is between min_area and max_area
    - Export candidate zone geometry, membership, summary, and an HTML QC map

Outputs:
    - candidate_control_zones.geojson
    - candidate_control_zones.csv
    - candidate_control_zone_membership.csv
    - candidate_control_zones_map.html

Usage:
 python build_candidate_control_zones.py \
   --out-dir candidate_controls \
   --verbose
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import deque
from pathlib import Path
from typing import Dict, List, Set, Tuple

import folium
import geopandas as gpd
import pandas as pd
from shapely.geometry import mapping
from shapely.ops import unary_union

# -----------------------------
# Configuration
# -----------------------------

AREA_CRS = "EPSG:26917"

BASE_DIR = Path(__file__).resolve().parent.parent

DEFAULT_TRACT_FILE = BASE_DIR / "assets" / "Detroit_Census_Tracts.geojson"
DEFAULT_CVI_FILE = BASE_DIR / "assets" / "Community_Violence_Intervention_Areas.geojson"
DEFAULT_BOUNDARY_FILE = BASE_DIR / "assets" / "City_of_Detroit_Boundary.geojson"

DEFAULT_TRACT_GEOID_COL = "GEOID"
DEFAULT_CVI_NAME_COL = "CVI_AREA_NAME"

DEFAULT_MIN_AREA = 3.5
DEFAULT_MAX_AREA = 4.5


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
        description="Build contiguous non-CVI candidate control zones from Detroit census tracts."
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
        help=f"Path to Detroit boundary GeoJSON (default: {DEFAULT_BOUNDARY_FILE}).",
    )
    parser.add_argument(
        "--tract-geoid-col",
        default=DEFAULT_TRACT_GEOID_COL,
        help=f"Tract GEOID column (default: {DEFAULT_TRACT_GEOID_COL}).",
    )
    parser.add_argument(
        "--cvi-name-col",
        default=DEFAULT_CVI_NAME_COL,
        help=f"CVI name column (default: {DEFAULT_CVI_NAME_COL}).",
    )
    parser.add_argument(
        "--min-area",
        type=float,
        default=DEFAULT_MIN_AREA,
        help=f"Minimum candidate zone area in square miles (default: {DEFAULT_MIN_AREA}).",
    )
    parser.add_argument(
        "--max-area",
        type=float,
        default=DEFAULT_MAX_AREA,
        help=f"Maximum candidate zone area in square miles (default: {DEFAULT_MAX_AREA}).",
    )
    parser.add_argument(
        "--max-tracts-per-zone",
        type=int,
        default=5,
        help="Maximum number of tracts allowed in a candidate zone search path (default: 8).",
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


def spatial_filter_to_detroit(
    tracts: gpd.GeoDataFrame,
    boundary: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    boundary_union = (
        boundary.geometry.union_all()
        if hasattr(boundary.geometry, "union_all")
        else boundary.unary_union
    )
    centroids = tracts.geometry.centroid
    mask = centroids.within(boundary_union)
    filtered = tracts.loc[mask].copy()
    logging.info(f"Filtered tracts to Detroit by centroid: {len(filtered):,} retained")
    return filtered


def build_adjacency(tracts: gpd.GeoDataFrame, id_col: str) -> Dict[str, Set[str]]:
    """
    Build adjacency using polygon touches. This is O(n^2), but tract counts are small enough here.
    """
    ids = tracts[id_col].tolist()
    geoms = tracts.geometry.tolist()

    adjacency: Dict[str, Set[str]] = {gid: set() for gid in ids}

    for i in range(len(tracts)):
        gid_i = ids[i]
        geom_i = geoms[i]
        for j in range(i + 1, len(tracts)):
            gid_j = ids[j]
            geom_j = geoms[j]
            if geom_i.touches(geom_j):
                adjacency[gid_i].add(gid_j)
                adjacency[gid_j].add(gid_i)

    return adjacency


def generate_candidate_clusters(
    area_lookup: Dict[str, float],
    adjacency: Dict[str, Set[str]],
    geometry_lookup: Dict[str, object],
    centroid_lookup: Dict[str, object],
    min_area: float,
    max_area: float,
    max_tracts_per_zone: int,
    min_shared_border_m: float = 50.0,
    max_centroid_distance_m: float = 4000.0,
    min_compactness: float = 0.35,
) -> List[Tuple[str, ...]]:
    """
    Generate contiguous candidate clusters using bounded search, with added
    spatial realism constraints.

    Filters applied:
        - area must be within [min_area, max_area]
        - cluster cannot exceed max_tracts_per_zone
        - newly added tract must share at least min_shared_border_m with
          at least one tract already in the cluster
        - max centroid-to-centroid distance within cluster cannot exceed
          max_centroid_distance_m
        - compactness = area / convex_hull_area must be at least min_compactness

    Notes:
        - This substantially reduces long, snake-like, weakly connected zones.
        - Requires projected CRS in meters.
    """
    all_candidates: Set[Tuple[str, ...]] = set()

    sorted_ids = sorted(adjacency.keys())

    for start in sorted_ids:
        queue = deque()
        queue.append((start,))

        while queue:
            cluster = queue.popleft()
            cluster_set = set(cluster)

            cluster_area = sum(area_lookup[g] for g in cluster)

            # Too big -> stop immediately
            if cluster_area > max_area:
                continue

            cluster_key = tuple(sorted(cluster))

            # Build geometry for spatial quality checks
            cluster_geoms = [geometry_lookup[g] for g in cluster]
            cluster_union = unary_union(cluster_geoms)

            # Compactness check
            hull_area = cluster_union.convex_hull.area
            compactness = (cluster_union.area / hull_area) if hull_area > 0 else 0.0

            # Max centroid spread check
            cluster_centroids = [centroid_lookup[g] for g in cluster]
            max_pairwise_centroid_dist = 0.0
            if len(cluster_centroids) > 1:
                for i in range(len(cluster_centroids)):
                    for j in range(i + 1, len(cluster_centroids)):
                        d = cluster_centroids[i].distance(cluster_centroids[j])
                        if d > max_pairwise_centroid_dist:
                            max_pairwise_centroid_dist = d

            # Valid candidate
            if (
                min_area <= cluster_area <= max_area
                and compactness >= min_compactness
                and max_pairwise_centroid_dist <= max_centroid_distance_m
            ):
                all_candidates.add(cluster_key)

            # At max size/area -> don't expand further
            if cluster_area >= max_area:
                continue

            if len(cluster) >= max_tracts_per_zone:
                continue

            # Expand only to neighbors of the existing cluster
            neighbor_pool: Set[str] = set()
            for gid in cluster:
                neighbor_pool.update(adjacency[gid])

            neighbor_pool -= cluster_set

            for nbr in sorted(neighbor_pool):
                nbr_geom = geometry_lookup[nbr]

                # Require meaningful shared boundary with at least one current tract
                shared_border_ok = False
                for gid in cluster:
                    shared_len = geometry_lookup[gid].boundary.intersection(nbr_geom.boundary).length
                    if shared_len >= min_shared_border_m:
                        shared_border_ok = True
                        break

                if not shared_border_ok:
                    continue

                new_cluster = tuple(sorted(cluster_set | {nbr}))
                new_area = sum(area_lookup[g] for g in new_cluster)

                if new_area > max_area:
                    continue

                # Quick centroid spread pruning before queueing
                new_centroids = [centroid_lookup[g] for g in new_cluster]
                too_spread = False
                if len(new_centroids) > 1:
                    for i in range(len(new_centroids)):
                        for j in range(i + 1, len(new_centroids)):
                            if new_centroids[i].distance(new_centroids[j]) > max_centroid_distance_m:
                                too_spread = True
                                break
                        if too_spread:
                            break

                if too_spread:
                    continue

                # Quick compactness pruning before queueing
                new_union = unary_union([geometry_lookup[g] for g in new_cluster])
                new_hull_area = new_union.convex_hull.area
                new_compactness = (new_union.area / new_hull_area) if new_hull_area > 0 else 0.0

                if new_compactness < min_compactness:
                    continue

                queue.append(new_cluster)

    return sorted(all_candidates)


def create_candidate_geometries(
    tracts: gpd.GeoDataFrame,
    id_col: str,
    candidates: List[Tuple[str, ...]],
) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    """
    Build dissolved candidate zone geometries and membership table.
    """
    tract_geom_lookup = dict(zip(tracts[id_col], tracts.geometry))
    tract_area_lookup = dict(zip(tracts[id_col], tracts["tract_area_sq_miles"]))

    candidate_rows = []
    membership_rows = []

    for idx, member_ids in enumerate(candidates, start=1):
        candidate_id = f"CAND_{idx:04d}"
        subset = tracts[tracts[id_col].isin(member_ids)].copy()
        dissolved = subset.dissolve(as_index=False)

        geometry = dissolved.geometry.iloc[0]
        area_sq_miles = sum(tract_area_lookup[g] for g in member_ids)

        candidate_rows.append(
            {
                "candidate_id": candidate_id,
                "area_sq_miles": area_sq_miles,
                "n_source_tracts": len(member_ids),
                "member_geoids": ",".join(member_ids),
                "geometry": geometry,
            }
        )

        for gid in member_ids:
            membership_rows.append(
                {
                    "candidate_id": candidate_id,
                    "GEOID": gid,
                    "tract_area_sq_miles": tract_area_lookup[gid],
                }
            )

    candidate_gdf = gpd.GeoDataFrame(candidate_rows, geometry="geometry", crs=tracts.crs)
    membership_df = pd.DataFrame(membership_rows)

    return candidate_gdf, membership_df


def make_html_map(
    boundary: gpd.GeoDataFrame,
    cvis: gpd.GeoDataFrame,
    candidates: gpd.GeoDataFrame,
    out_html: Path,
) -> None:
    """
    Build a quick Folium QC map.
    """
    boundary_wgs = boundary.to_crs(epsg=4326)[[boundary.geometry.name]].copy()
    cvis_wgs = cvis.to_crs(epsg=4326)[["cvi_name", cvis.geometry.name]].copy()
    candidates_wgs = candidates.to_crs(epsg=4326)[
    ["candidate_id", "area_sq_miles", "n_source_tracts", candidates.geometry.name]
    ].copy()

    center = boundary_wgs.geometry.union_all().centroid if hasattr(boundary_wgs.geometry, "union_all") else boundary_wgs.unary_union.centroid
    m = folium.Map(location=[center.y, center.x], zoom_start=11, tiles="CartoDB positron")

    # Detroit boundary
    folium.GeoJson(
        data=json.loads(boundary_wgs.to_json()),
        name="Detroit Boundary",
        style_function=lambda _: {
            "color": "black",
            "weight": 2,
            "fillOpacity": 0,
        },
    ).add_to(m)

    # CVIs
    folium.GeoJson(
        data=json.loads(cvis_wgs.to_json()),
        name="CVI Areas",
        style_function=lambda _: {
            "color": "#d62728",
            "weight": 2,
            "fillColor": "#d62728",
            "fillOpacity": 0.25,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=["cvi_name"] if "cvi_name" in cvis_wgs.columns else [],
            aliases=["CVI:"],
            sticky=False,
        ),
    ).add_to(m)

    # Candidates
    candidate_tooltip_fields = ["candidate_id", "area_sq_miles", "n_source_tracts"]
    candidate_tooltip_aliases = ["Candidate:", "Area (sq mi):", "# Tracts:"]

    folium.GeoJson(
        data=json.loads(candidates_wgs.to_json()),
        name="Candidate Control Zones",
        style_function=lambda _: {
            "color": "#1f77b4",
            "weight": 1.5,
            "fillColor": "#1f77b4",
            "fillOpacity": 0.20,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=candidate_tooltip_fields,
            aliases=candidate_tooltip_aliases,
            sticky=False,
        ),
    ).add_to(m)

    folium.LayerControl().add_to(m)
    m.save(str(out_html))


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    out_dir = ensure_output_dir(args.out_dir)

    logging.info("Loading geographic files...")
    tracts = load_geodata(args.tract_file)
    cvis = load_geodata(args.cvi_file)
    boundary = load_geodata(args.boundary_file)

    if args.tract_geoid_col not in tracts.columns:
        raise ValueError(
            f"Tract GEOID column '{args.tract_geoid_col}' not found in tract file. "
            f"Available columns: {list(tracts.columns)}"
        )

    if args.cvi_name_col not in cvis.columns:
        raise ValueError(
            f"CVI name column '{args.cvi_name_col}' not found in CVI file. "
            f"Available columns: {list(cvis.columns)}"
        )

    # Standardize tract GEOID and keep only needed columns
    tracts = tracts[[args.tract_geoid_col, tracts.geometry.name]].copy()
    tracts["GEOID"] = tracts[args.tract_geoid_col].map(standardize_geoid)
    tracts = tracts.loc[tracts["GEOID"].notna()].copy()
    tracts = tracts[["GEOID", tracts.geometry.name]].copy()

    # Dissolve duplicate GEOIDs if needed
    dup_counts = tracts["GEOID"].value_counts()
    dups = dup_counts[dup_counts > 1]
    if not dups.empty:
        logging.warning(
            f"{len(dups):,} GEOIDs appear more than once in tract file. "
            "Dissolving tract geometries by GEOID."
        )
        tracts = tracts.dissolve(by="GEOID", as_index=False)

    # Standardize CVI names
    cvis = cvis[[args.cvi_name_col, cvis.geometry.name]].copy()
    cvis["cvi_name"] = cvis[args.cvi_name_col].astype(str).str.strip()
    cvis = cvis[["cvi_name", cvis.geometry.name]].copy()

    # Reproject
    logging.info(f"Projecting all layers to {AREA_CRS}...")
    tracts = tracts.to_crs(AREA_CRS)
    cvis = cvis.to_crs(AREA_CRS)
    boundary = boundary.to_crs(AREA_CRS)

    # Restrict tracts to Detroit
    tracts = spatial_filter_to_detroit(tracts, boundary)

    # Compute tract area
    tracts["tract_area_sq_miles"] = compute_area_sq_miles(tracts)

    # Exclude any tract intersecting a CVI
    cvi_union = cvis.geometry.union_all() if hasattr(cvis.geometry, "union_all") else cvis.unary_union
    intersects_cvi = tracts.intersects(cvi_union)

    excluded = tracts.loc[intersects_cvi].copy()
    candidate_tracts = tracts.loc[~intersects_cvi].copy()

    logging.info(f"Excluded tracts intersecting CVIs: {len(excluded):,}")
    logging.info(f"Remaining non-CVI tracts: {len(candidate_tracts):,}")

    if candidate_tracts.empty:
        raise ValueError("No non-CVI tracts remain after excluding CVI-intersecting tracts.")

    # Build adjacency
    logging.info("Building tract adjacency...")
    adjacency = build_adjacency(candidate_tracts, id_col="GEOID")

    # Remove isolated tracts from adjacency dict only if they truly have no neighbors
    adjacency = {k: v for k, v in adjacency.items()}

    area_lookup = dict(zip(candidate_tracts["GEOID"], candidate_tracts["tract_area_sq_miles"]))

    # Generate contiguous candidate clusters
    logging.info("Generating candidate control zones...")

    geometry_lookup = dict(zip(candidate_tracts["GEOID"], candidate_tracts.geometry))
    centroid_lookup = dict(zip(candidate_tracts["GEOID"], candidate_tracts.geometry.centroid))
    area_lookup = dict(zip(candidate_tracts["GEOID"], candidate_tracts["tract_area_sq_miles"]))

    candidates = generate_candidate_clusters(
        area_lookup=area_lookup,
        adjacency=adjacency,
        geometry_lookup=geometry_lookup,
        centroid_lookup=centroid_lookup,
        min_area=args.min_area,
        max_area=args.max_area,
        max_tracts_per_zone=args.max_tracts_per_zone,
        min_shared_border_m=50.0,
        max_centroid_distance_m=4000.0,
        min_compactness=0.35,
    )

    logging.info(f"Candidate clusters found: {len(candidates):,}")

    if not candidates:
        raise ValueError(
            "No candidate zones found in the requested area range. "
            "Try increasing --max-tracts-per-zone or widening the min/max area."
        )

    # Create candidate geometries + membership
    candidate_gdf, membership_df = create_candidate_geometries(
        tracts=candidate_tracts,
        id_col="GEOID",
        candidates=candidates,
    )

    # Final summary CSV
    candidate_summary = candidate_gdf.drop(columns=[candidate_gdf.geometry.name]).copy()

    # Write outputs
    geojson_path = out_dir / "candidate_control_zones.geojson"
    csv_path = out_dir / "candidate_control_zones.csv"
    membership_path = out_dir / "candidate_control_zone_membership.csv"
    html_path = out_dir / "candidate_control_zones_map.html"

    candidate_gdf.to_file(geojson_path, driver="GeoJSON")
    candidate_summary.to_csv(csv_path, index=False)
    membership_df.to_csv(membership_path, index=False)

    logging.info(f"Wrote: {geojson_path}")
    logging.info(f"Wrote: {csv_path}")
    logging.info(f"Wrote: {membership_path}")

    # HTML QC map
    make_html_map(
        boundary=boundary,
        cvis=cvis,
        candidates=candidate_gdf,
        out_html=html_path,
    )
    logging.info(f"Wrote: {html_path}")

    # Quick sanity checks
    logging.info("Candidate zone area summary:")
    logging.info(
        f"Min area: {candidate_gdf['area_sq_miles'].min():.3f} | "
        f"Max area: {candidate_gdf['area_sq_miles'].max():.3f} | "
        f"Mean area: {candidate_gdf['area_sq_miles'].mean():.3f}"
    )

    logging.info("Done.")


if __name__ == "__main__":
    main()