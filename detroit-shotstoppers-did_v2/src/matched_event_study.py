#!/usr/bin/env python3
"""
matched_event_study.py

Run a clean pooled matched-pair event study on the long-format matched fractional dataset.

Model:
    crime_count ~ sum_k [treated x I(event_time_k)] + log_calls + C(pair_id) + C(month)

where:
    - event_time_k is relative month to treatment start
    - omitted reference period is event_time = -1
    - outcome is weighted monthly RMS count
    - control is log(calls_count + 1)
    - pair fixed effects compare within matched pairs
    - month fixed effects absorb common shocks
    - standard errors are clustered by pair_id

Recommended use:
    Exclude problematic / late-treated CVIs, e.g.
        --exclude "Live in Peace,Team Pursuit"

Outputs:
    - matched_event_study_panel.csv
    - matched_event_study_coefficients.csv
    - matched_event_study_results.txt
    - matched_event_study_plot.html
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import statsmodels.formula.api as smf


# -----------------------------
# Defaults
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DEFAULT_INPUT = BASE_DIR / "src" / "matched_controls" / "incidents_long_matched_fractional.csv"
DEFAULT_PROGRAM_START = "2023-08-01"
DEFAULT_START_DATE = "2021-01-01"

DEFAULT_MIN_EVENT_MONTH = -12
DEFAULT_MAX_EVENT_MONTH = 24


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
        raise KeyError(f"Missing required columns in long file: {missing}")

    df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce")
    df["month"] = pd.to_datetime(df["month"], errors="coerce")
    df["treated"] = pd.to_numeric(df["treated"], errors="coerce").fillna(0).astype(int)
    df["weight"] = pd.to_numeric(df["weight"], errors="coerce").fillna(0.0)

    return df


def month_diff(dates: pd.Series, treatment_start: pd.Timestamp) -> pd.Series:
    return (
        (dates.dt.year - treatment_start.year) * 12
        + (dates.dt.month - treatment_start.month)
    )


def event_label(k: int) -> str:
    if k < 0:
        return f"m{abs(k)}"
    return f"p{k}"


def parse_event_term(term: str) -> int | None:
    """
    Parse terms like:
      treated:C(event_time_cat, Treatment(reference='m1'))[T.m12]
      treated:C(event_time_cat, Treatment(reference='m1'))[T.p0]
    """
    m = re.search(r"\[T\.(m\d+|p\d+)\]$", term)
    if not m:
        return None
    lab = m.group(1)
    if lab.startswith("m"):
        return -int(lab[1:])
    return int(lab[1:])


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run pooled matched-pair event study on incidents_long_matched_fractional.csv"
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help=f"Path to incidents_long_matched_fractional.csv (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Required output directory for event study results.",
    )
    parser.add_argument(
        "--exclude",
        default="Live in Peace,Team Pursuit",
        help=(
            "Comma-separated list of CVIs to exclude. "
            "Default excludes the poor/late-treated matches: "
            '"Live in Peace,Team Pursuit"'
        ),
    )
    parser.add_argument(
        "--program-start",
        default=DEFAULT_PROGRAM_START,
        help=f"Treatment start date used as event time 0 (default: {DEFAULT_PROGRAM_START})",
    )
    parser.add_argument(
        "--start-date",
        default=DEFAULT_START_DATE,
        help=f"Earliest event date to include (default: {DEFAULT_START_DATE})",
    )
    parser.add_argument(
        "--min-event-month",
        type=int,
        default=DEFAULT_MIN_EVENT_MONTH,
        help=f"Minimum relative month to keep (default: {DEFAULT_MIN_EVENT_MONTH})",
    )
    parser.add_argument(
        "--max-event-month",
        type=int,
        default=DEFAULT_MAX_EVENT_MONTH,
        help=f"Maximum relative month to keep (default: {DEFAULT_MAX_EVENT_MONTH})",
    )

    args = parser.parse_args()

    out_dir = ensure_out_dir(args.out_dir)
    exclude_list = parse_exclude_list(args.exclude)

    print("[matched_event_study.py] Loading long-format matched file...")
    df = load_long_file(args.input)

    start_date = pd.Timestamp(args.start_date)
    df = df.loc[df["event_time"] >= start_date].copy()

    if exclude_list:
        print(f"[matched_event_study.py] Excluding CVIs: {exclude_list}")
        before = len(df)
        df = df.loc[~df["pair_cvi_name"].isin(exclude_list)].copy()
        after = len(df)
        print(f"[matched_event_study.py] Dropped {before - after:,} rows due to exclusions.")

    if df.empty:
        raise ValueError("No rows remain after filtering/exclusions.")

    print("[matched_event_study.py] Building monthly panel...")
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

    treatment_start = pd.Timestamp(args.program_start)
    panel["event_time_k"] = month_diff(panel["month"], treatment_start)

    panel = panel.loc[
        (panel["event_time_k"] >= args.min_event_month)
        & (panel["event_time_k"] <= args.max_event_month)
    ].copy()

    if panel.empty:
        raise ValueError("No panel rows remain after applying event-month window.")

    panel["log_calls"] = np.log1p(panel["calls_count"])
    panel["month_str"] = panel["month"].dt.strftime("%Y-%m")

    # Reference period = -1
    ref_k = -1
    panel["event_time_cat"] = panel["event_time_k"].map(event_label)

    if ref_k not in panel["event_time_k"].unique():
        raise ValueError(
            "Reference period event_time = -1 is not present in the data. "
            "Adjust event window or treatment timing."
        )

    panel_out = round_numeric(panel, 3)
    panel_path = out_dir / "matched_event_study_panel.csv"
    panel_out.to_csv(panel_path, index=False)
    print(f"[matched_event_study.py] Wrote panel to {panel_path}")

    print("[matched_event_study.py] Estimating pooled event study...")
    formula = (
        "crime_count ~ treated * C(event_time_cat, Treatment(reference='m1')) "
        "+ log_calls + C(pair_id) + C(month)"
    )

    reg = smf.ols(formula=formula, data=panel).fit(
        cov_type="cluster",
        cov_kwds={"groups": panel["pair_id"]},
    )

    # Extract event-study coefficients
    rows = []
    for term, coef in reg.params.items():
        if not term.startswith("treated:C(event_time_cat"):
            continue

        k = parse_event_term(term)
        if k is None:
            continue

        se = reg.bse.get(term, np.nan)
        p = reg.pvalues.get(term, np.nan)
        ci = reg.conf_int().loc[term]

        rows.append(
            {
                "event_time_k": k,
                "coef": coef,
                "std_err": se,
                "p_value": p,
                "ci_low": ci[0],
                "ci_high": ci[1],
            }
        )

    coef_df = pd.DataFrame(rows).sort_values("event_time_k").reset_index(drop=True)

    # Add omitted reference period manually
    ref_row = pd.DataFrame(
        [{"event_time_k": ref_k, "coef": 0.0, "std_err": np.nan, "p_value": np.nan, "ci_low": 0.0, "ci_high": 0.0}]
    )
    coef_df = pd.concat([coef_df, ref_row], axis=0, ignore_index=True)
    coef_df = coef_df.sort_values("event_time_k").reset_index(drop=True)

    coef_out = round_numeric(coef_df, 3)
    coef_path = out_dir / "matched_event_study_coefficients.csv"
    coef_out.to_csv(coef_path, index=False)
    print(f"[matched_event_study.py] Wrote coefficient table to {coef_path}")

    # Plot
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=coef_df["event_time_k"],
            y=coef_df["coef"],
            mode="lines+markers",
            name="Estimate",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=coef_df["event_time_k"],
            y=coef_df["ci_high"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=coef_df["event_time_k"],
            y=coef_df["ci_low"],
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            name="95% CI",
            hoverinfo="skip",
        )
    )

    fig.add_hline(y=0, line_dash="dash")
    fig.add_vline(x=0, line_dash="dash")

    fig.update_layout(
        title="Matched Event Study (Clean Pooled Sample)",
        xaxis_title="Months Relative to Treatment Start (Aug 2023 = 0)",
        yaxis_title="Estimated Treatment Effect on Monthly RMS Incidents",
        hovermode="x unified",
    )

    plot_path = out_dir / "matched_event_study_plot.html"
    fig.write_html(plot_path)
    print(f"[matched_event_study.py] Wrote plot to {plot_path}")

    # Text summary
    results_path = out_dir / "matched_event_study_results.txt"
    with open(results_path, "w", encoding="utf-8") as f:
        f.write("=== matched_event_study.py summary ===\n\n")
        f.write(f"Input file: {args.input}\n")
        f.write(f"Output dir: {out_dir}\n")
        f.write(f"Program start: {args.program_start}\n")
        f.write(f"Start date filter: {args.start_date}\n")
        f.write(f"Excluded CVIs: {exclude_list if exclude_list else 'None'}\n")
        f.write(f"Event window: [{args.min_event_month}, {args.max_event_month}]\n\n")

        f.write("Model formula:\n")
        f.write(f"{formula}\n\n")

        f.write("Model fit:\n")
        f.write(f"R-squared: {reg.rsquared:.3f}\n")
        f.write(f"Adj. R-squared: {reg.rsquared_adj:.3f}\n")
        f.write(f"N observations: {int(reg.nobs)}\n\n")

        f.write("Event-study coefficients (see CSV for rounded version):\n")
        f.write(coef_df.to_string(index=False))
        f.write("\n\n")
        f.write(reg.summary().as_text())

    print(f"[matched_event_study.py] Wrote text summary to {results_path}")
    print("[matched_event_study.py] Done.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[matched_event_study.py] ERROR: {e}", file=sys.stderr)
        sys.exit(1)