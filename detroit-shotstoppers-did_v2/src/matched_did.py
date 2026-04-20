#!/usr/bin/env python3
"""
matched_did.py

Run a standard matched-pair DID on the long-format matched fractional dataset.

Model:
    crime_count ~ treated * post + log_calls + C(pair_id) + C(month)

Where:
    - crime_count is weighted monthly RMS count
    - calls_count is weighted monthly 911 count
    - log_calls = log(calls_count + 1)
    - pair fixed effects compare within matched pairs
    - month fixed effects absorb common time shocks
    - standard errors are clustered by pair_id

Inputs:
    - incidents_long_matched_fractional.csv

Outputs (written to user-specified out-dir):
    - matched_did_panel.csv
    - matched_did_results.txt
    - matched_did_coefficients.csv
    - matched_did_pair_summary.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


# -----------------------------
# Defaults
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DEFAULT_INPUT = BASE_DIR / "src" / "matched_controls" / "incidents_long_matched_fractional.csv"
DEFAULT_PROGRAM_START = "2023-08-01"
DEFAULT_START_DATE = "2021-01-01"


# -----------------------------
# Helpers
# -----------------------------

def parse_exclude_list(exclude_str: str | None) -> list[str]:
    if exclude_str is None:
        return []
    parts = [p.strip() for p in exclude_str.split(",")]
    return [p for p in parts if p]


def ensure_out_dir(path: str | Path) -> Path:
    out_dir = Path(path)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def round_numeric(df: pd.DataFrame, digits: int = 3) -> pd.DataFrame:
    out = df.copy()
    num_cols = out.select_dtypes(include=[np.number]).columns
    out[num_cols] = out[num_cols].round(digits)
    return out


def load_long_file(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)

    required = [
        "source",
        "event_time",
        "month",
        "zone_name",
        "zone_type",
        "pair_id",
        "pair_cvi_name",
        "treated",
        "weight",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            f"Missing required columns in long file: {missing}"
        )

    df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce")
    df["month"] = pd.to_datetime(df["month"], errors="coerce")
    df["treated"] = pd.to_numeric(df["treated"], errors="coerce").fillna(0).astype(int)
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(0.0)

    return df


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run matched-pair DID on incidents_long_matched_fractional.csv"
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help=f"Path to incidents_long_matched_fractional.csv (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Required output directory for DID results.",
    )
    parser.add_argument(
        "--exclude",
        default=None,
        help=(
            "Comma-separated list of CVIs to exclude. "
            "Excluding a CVI removes both the CVI and its matched control pair. "
            'Example: --exclude "Live in Peace,Team Pursuit"'
        ),
    )
    parser.add_argument(
        "--program-start",
        default=DEFAULT_PROGRAM_START,
        help=f"Program start date for standard DID post indicator (default: {DEFAULT_PROGRAM_START})",
    )
    parser.add_argument(
        "--start-date",
        default=DEFAULT_START_DATE,
        help=f"Earliest event date to include (default: {DEFAULT_START_DATE})",
    )

    args = parser.parse_args()

    out_dir = ensure_out_dir(args.out_dir)
    exclude_list = parse_exclude_list(args.exclude)

    print("[matched_did.py] Loading long-format matched file...")
    df = load_long_file(args.input)

    # Filter date range
    start_date = pd.Timestamp(args.start_date)
    df = df.loc[df["event_time"] >= start_date].copy()

    # Optional exclusions: drop whole matched pairs by pair_cvi_name
    if exclude_list:
        print(f"[matched_did.py] Excluding CVIs: {exclude_list}")
        before = len(df)
        df = df.loc[~df["pair_cvi_name"].isin(exclude_list)].copy()
        after = len(df)
        print(f"[matched_did.py] Dropped {before - after:,} rows due to exclusions.")

    if df.empty:
        raise ValueError("No rows remain after filtering/exclusions.")

    # Build monthly panel
    print("[matched_did.py] Building monthly panel...")
    df["is_crime"] = df["source"].astype(str).str.upper().eq("RMS")
    df["is_call"] = ~df["is_crime"]

    panel = (
        df.groupby(
            ["pair_id", "pair_cvi_name", "zone_name", "zone_type", "treated", "month"],
            as_index=False
        )
        .agg(
            crime_count=("weight", lambda s: s[df.loc[s.index, "is_crime"]].sum()),
            calls_count=("weight", lambda s: s[df.loc[s.index, "is_call"]].sum()),
        )
        .copy()
    )

    # Post indicator
    program_start = pd.Timestamp(args.program_start)
    panel["post"] = (panel["month"] >= program_start).astype(int)
    panel["treated_post"] = panel["treated"] * panel["post"]

    # Calls transform
    panel["log_calls"] = np.log1p(panel["calls_count"])

    # Helpful identifiers
    panel["month_str"] = panel["month"].dt.strftime("%Y-%m")
    panel["zone_id"] = panel["zone_name"].astype(str)

    # Sanity summaries
    print("\n[matched_did.py] Pair counts:")
    print(panel["pair_id"].value_counts().sort_index())

    print("\n[matched_did.py] Zones in analysis:")
    print(panel[["pair_id", "pair_cvi_name", "zone_name", "zone_type"]]
          .drop_duplicates()
          .sort_values(["pair_id", "zone_type", "zone_name"]))

    # Export panel
    panel_out = round_numeric(panel, 3)
    panel_path = out_dir / "matched_did_panel.csv"
    panel_out.to_csv(panel_path, index=False)
    print(f"\n[matched_did.py] Wrote panel to {panel_path}")

    # Pair summary
    pair_summary = (
        panel.groupby(["pair_id", "pair_cvi_name", "zone_type"], as_index=False)
        .agg(
            total_crime=("crime_count", "sum"),
            avg_monthly_crime=("crime_count", "mean"),
            total_calls=("calls_count", "sum"),
            avg_monthly_calls=("calls_count", "mean"),
            n_months=("month", "nunique"),
        )
        .sort_values(["pair_id", "zone_type"])
        .copy()
    )
    pair_summary_out = round_numeric(pair_summary, 3)
    pair_summary_path = out_dir / "matched_did_pair_summary.csv"
    pair_summary_out.to_csv(pair_summary_path, index=False)
    print(f"[matched_did.py] Wrote pair summary to {pair_summary_path}")

    # Regression
    print("\n[matched_did.py] Estimating DID model...")
    formula = "crime_count ~ treated * post + log_calls + C(pair_id) + C(month)"

    reg = smf.ols(formula=formula, data=panel).fit(
        cov_type="cluster",
        cov_kwds={"groups": panel["pair_id"]},
    )

    # Coefficient table
    coef_table = pd.DataFrame({
        "term": reg.params.index,
        "coef": reg.params.values,
        "std_err": reg.bse.values,
        "t": reg.tvalues.values,
        "p_value": reg.pvalues.values,
        "ci_low": reg.conf_int()[0].values,
        "ci_high": reg.conf_int()[1].values,
    })
    coef_table_out = round_numeric(coef_table, 3)

    coef_path = out_dir / "matched_did_coefficients.csv"
    coef_table_out.to_csv(coef_path, index=False)
    print(f"[matched_did.py] Wrote coefficient table to {coef_path}")

    # Text summary
    results_path = out_dir / "matched_did_results.txt"
    with open(results_path, "w", encoding="utf-8") as f:
        f.write("=== matched_did.py summary ===\n\n")
        f.write(f"Input file: {args.input}\n")
        f.write(f"Output dir: {out_dir}\n")
        f.write(f"Program start: {args.program_start}\n")
        f.write(f"Start date filter: {args.start_date}\n")
        f.write(f"Excluded CVIs: {exclude_list if exclude_list else 'None'}\n\n")

        f.write("Model formula:\n")
        f.write(f"{formula}\n\n")

        f.write("Key DID coefficient:\n")
        if "treated:post" in reg.params.index:
            f.write(
                f"treated:post = {reg.params['treated:post']:.3f} "
                f"(SE {reg.bse['treated:post']:.3f}, "
                f"p={reg.pvalues['treated:post']:.3f})\n\n"
            )
        else:
            f.write("treated:post term not found.\n\n")

        f.write("Model fit:\n")
        f.write(f"R-squared: {reg.rsquared:.3f}\n")
        f.write(f"Adj. R-squared: {reg.rsquared_adj:.3f}\n")
        f.write(f"N observations: {int(reg.nobs)}\n\n")

        f.write(reg.summary().as_text())

    print(f"[matched_did.py] Wrote text summary to {results_path}")

    # Console highlight
    print("\n[matched_did.py] Key DID result:")
    if "treated:post" in reg.params.index:
        print(
            f"treated:post = {reg.params['treated:post']:.3f} "
            f"(SE {reg.bse['treated:post']:.3f}, "
            f"p={reg.pvalues['treated:post']:.3f})"
        )
    else:
        print("treated:post term not found.")

    print("\n[matched_did.py] Done.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[matched_did.py] ERROR: {e}", file=sys.stderr)
        sys.exit(1)