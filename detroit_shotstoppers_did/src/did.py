#!/usr/bin/env python3
"""
did.py

Basic Difference-in-Differences model using incidents_long_fractional.csv.
Aggregates fractional incidents to monthly zone-level panels and fits:

    crime_count_it = β0
                    + β1 * treated_i
                    + β2 * time_t
                    + β3 * treated_i * time_t
                    + β4 * calls_count_it
                    + ε_it

Outputs:
- did_panel.csv       (panel dataset used for modeling)
- did_results.txt     (model summary)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import statsmodels.formula.api as smf
import argparse


# ---------------------------------------------------------
# Load and preprocess
# ---------------------------------------------------------
def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce")
    df = df.dropna(subset=["event_time"])
    return df

# ---------------------------------------------------------
# Build monthly panel dataset
# ---------------------------------------------------------
def build_panel(df: pd.DataFrame) -> pd.DataFrame:
    # Monthly index
    df["month"] = df["event_time"].dt.to_period("M").dt.to_timestamp()

    # Identify RMS vs 911
    if "source" in df.columns:
        src_lower = df["source"].astype(str).str.lower()
        is_crime = src_lower.str.contains("rms", na=False)
        is_calls = src_lower.str.contains("911", na=False)
    else:
        raise KeyError("Expected a 'source' column (e.g., 'RMS' vs '911') in incidents_long_fractional.csv")

    # Ensure no NaNs before converting to int
    df["is_crime"] = is_crime.fillna(False).astype(int)
    df["is_calls"] = is_calls.fillna(False).astype(int)

    # Weighted aggregation
    panel = df.groupby(["month", "cvi_area"]).agg(
        crime_count=("weight", lambda x: x[df.loc[x.index, "is_crime"] == 1].sum()),
        calls_count=("weight", lambda x: x[df.loc[x.index, "is_calls"] == 1].sum())
    ).reset_index()

    # Replace NaN with zero
    panel[["crime_count", "calls_count"]] = panel[["crime_count", "calls_count"]].fillna(0)

    # ---------------------------------------------------------
    # Build DiD variables
    # ---------------------------------------------------------
    panel["treated"] = (panel["cvi_area"] != "Non-CVI").astype(int)

    program_start = pd.Timestamp("2023-08-01")
    panel["time"] = (panel["month"] >= program_start).astype(int)
    panel["treated_time"] = panel["treated"] * panel["time"]

    return panel


# ---------------------------------------------------------
# Fit basic DiD model
# ---------------------------------------------------------
def run_did(panel: pd.DataFrame):
    model = smf.ols(
        "crime_count ~ treated + time + treated_time + calls_count",
        data=panel
    ).fit(cov_type="HC3")
    return model


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--incidents", type=str, default="incidents_long_fractional.csv")
    parser.add_argument("--out-panel", type=str, default="did_panel.csv")
    parser.add_argument("--out-results", type=str, default="did_results.txt")
    args = parser.parse_args()

    incidents_path = Path(args.incidents)

    print("[did.py] Loading data…")
    df = load_data(incidents_path)

    print("[did.py] Building monthly panel…")
    panel = build_panel(df)
    panel.to_csv(args.out_panel, index=False)
    print(f"[did.py] Saved panel to {args.out_panel}")

    print("[did.py] Running DiD model…")
    model = run_did(panel)

    with open(args.out_results, "w") as f:
        f.write(model.summary().as_text())

    print(f"[did.py] Saved model results to {args.out_results}")
    print("\n=== DiD Estimate (treated_time) ===")
    print(model.params.get("treated_time", "MISSING"))
    print("===================================")


if __name__ == "__main__":
    main()
