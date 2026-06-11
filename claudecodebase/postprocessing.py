"""
Post-processing utilities: unit conversion, labeling, and summary statistics.
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Column names for the beautified outputs
# ---------------------------------------------------------------------------

TIMESERIES_COLS = [
    "sample_id",
    "param_id",
    "time",
    "out-flow (ADV%)",
    "inventory (ADV%)",
    "impact state (bps)",
    "in-flow (ADV%)",
]

SUMMARY_STAT_COLS = [
    "param_id",
    "sample_id",
    "Total variation of out-flow (ADV%)",
    "Terminal in-flow (ADV%)",
    "Terminal impact state (bps)",
    "Total variation of in-flow (ADV%)",
    "Sqrt QV of in-flow (ADV%)",
    "Spread cost per in-flow (bps)",
    "Impact cost per in-flow (bps)",
    "Total cost per in-flow (bps)",
    "Spread cost",
    "Impact cost",
    "Total cost",
    "Closing trade as a percent of out-flow (%)",
    "Internalization (%)",
    "Out-flow TV over in-flow sqrtQV (%)",
    "Internalization regret (%)",
    "Impact to spread ratio",
    "Continuous trading as a percent of out-flow (%)",
]


def beautify(params, summary_stats, timeseries_df, N):
    """
    Convert raw simulation outputs to human-readable units and labels.

    - Flows expressed as % of ADV.
    - Impact state expressed in basis points (bps).
    - Time axis mapped to NYSE trading hours (09:30 – 16:00).

    Parameters
    ----------
    params : pd.DataFrame
        Parameter table with a ``param_id`` column. May optionally contain
        a ``parameter scans`` column for grouping.
    summary_stats : pd.DataFrame
        Raw output from ``simulate_path``.
    timeseries_df : pd.DataFrame
        Raw path output from ``simulate_path``.
    N : int
        Number of time steps used in the simulation.

    Returns
    -------
    summary_stats : pd.DataFrame
    timeseries_df : pd.DataFrame
    """
    timeseries_df = timeseries_df.copy()
    summary_stats = summary_stats.copy()

    # --- Time series ---
    timeseries_df["out-flow (ADV%)"] = timeseries_df["tradeSum"] * 100
    timeseries_df["inventory (ADV%)"] = timeseries_df["Xs"] * 100
    timeseries_df["impact state (bps)"] = timeseries_df["Ys"] * 1e4
    timeseries_df["in-flow (ADV%)"] = timeseries_df["Zs"] * 100
    timeseries_df["time"] = timeseries_df["time"] / N * 6.5 + 9.5  # hours since midnight

    timeseries_df = timeseries_df[TIMESERIES_COLS].merge(params, on=["param_id"])
    timeseries_df = timeseries_df.drop(columns=["param_id"])

    # Correct impact state at t=0 to reflect the initial jump
    mask = timeseries_df["time"] == 9.5
    timeseries_df.loc[mask, "impact state (bps)"] = timeseries_df.loc[mask, "y"] * 1e4

    # --- Summary statistics ---
    tv_in = summary_stats["TVInFlows"]
    tv_out = summary_stats["TVOutFlows"]
    qv_in = summary_stats["QVInFlows"]

    summary_stats["Total variation of out-flow (ADV%)"] = tv_out * 100
    summary_stats["Terminal in-flow (ADV%)"] = summary_stats["ZT"] * 100
    summary_stats["Terminal impact state (bps)"] = summary_stats["YT"] * 1e4
    summary_stats["Total variation of in-flow (ADV%)"] = tv_in * 100
    summary_stats["Sqrt QV of in-flow (ADV%)"] = np.sqrt(qv_in) * 100
    summary_stats["Spread cost per in-flow (bps)"] = summary_stats["spreadCost"] / tv_in * 1e4
    summary_stats["Impact cost per in-flow (bps)"] = summary_stats["impactCost"] / tv_in * 1e4
    summary_stats["Total cost per in-flow (bps)"] = (
        (summary_stats["spreadCost"] + summary_stats["impactCost"]) / tv_in * 1e4
    )
    summary_stats["Spread cost"] = summary_stats["spreadCost"]
    summary_stats["Impact cost"] = summary_stats["impactCost"]
    summary_stats["Total cost"] = summary_stats["spreadCost"] + summary_stats["impactCost"]
    summary_stats["Closing trade as a percent of out-flow (%)"] = summary_stats["JTProp"] * 100
    summary_stats["Internalization (%)"] = (1 - tv_out / tv_in) * 100
    summary_stats["Out-flow TV over in-flow sqrtQV (%)"] = (tv_out / np.sqrt(qv_in)) * 100
    summary_stats["Internalization regret (%)"] = (
        1 - summary_stats["ZT"] / tv_out
    ) * 100
    summary_stats["Impact to spread ratio"] = (
        summary_stats["impactCost"] / summary_stats["spreadCost"]
    )
    summary_stats["Continuous trading as a percent of out-flow (%)"] = (
        summary_stats["intradayTrds"] / tv_out * 100
    )

    summary_stats = summary_stats[SUMMARY_STAT_COLS]

    if "parameter scans" in params.columns:
        summary_stats = summary_stats.merge(
            params[["param_id", "parameter scans"]], on=["param_id"]
        ).drop(columns=["param_id"])
    else:
        summary_stats = summary_stats.merge(params, on=["param_id"])

    return summary_stats, timeseries_df


def summarize_stats(stats_output: pd.DataFrame, round: int = 1) -> pd.DataFrame:
    """
    Return the mean of each statistic grouped by ``parameter scans``.

    Parameters
    ----------
    stats_output : pd.DataFrame
        Output of ``beautify``.
    round : int
        Number of decimal places.

    Returns
    -------
    pd.DataFrame
        Mean statistics per parameter group.
    """
    summary = stats_output.groupby("parameter scans").mean().round(round)
    summary.drop(columns=["sample_id"], inplace=True, errors="ignore")
    return summary
