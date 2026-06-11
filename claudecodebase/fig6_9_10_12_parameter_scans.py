"""
Parameter scan experiments:
  - Figure 6  : deep scan of in-flow volatility σ
  - Figure 9  : deterministic baseline, sensitivity to spread-cost ε
  - Figure 10 : joint scan of ε and θ
  - Figure 12 : sensitivity to initial impact state y₀
  - Heatmap   : joint initial conditions (not in paper)
"""

import sys
sys.path.insert(0, "..")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

from src import (
    generate_parameters,
    simulate,
    beautify,
    summarize_stats,
    plot_timeseries,
    plot_parameter_scan,
    DEFAULT_COLORS,
    DEFAULT_HUE_ORDER,
    BLUE_HEX, ORANGE_HEX, GREEN_HEX,
)

SPEED_UP_FACTOR = 10

THETAS   = [0.0, 1.0, -1.0]
FLOW_MAP = pd.DataFrame({
    "parameter scans": ["martingale in-flow", "reversal in-flow", "momentum in-flow"],
    "theta": THETAS,
})
COLORS    = DEFAULT_COLORS
HUE_ORDER = DEFAULT_HUE_ORDER


# ===========================================================================
# Figure 9 — Deterministic baseline: sensitivity to ε
# ===========================================================================

def run_figure9():
    eps   = [1e-4, 1e-3, 1e-2, 0.1]
    names = ["(nearly) no spread", "low spread", "medium spread", "high spread"]

    parameters = generate_parameters(eps=eps, sigma=[0.0])
    summary_stats, timeseries_df, df, N = simulate(parameters)

    param_map = pd.DataFrame({"parameter scans": names, "eps": eps})
    params = parameters.merge(param_map, on=["eps"])
    params["parameter scans"] = pd.Categorical(params["parameter scans"], categories=names, ordered=True)

    stats_output, ts_output = beautify(params, summary_stats, timeseries_df, N)

    sns.set_palette("rocket")
    plot_timeseries(ts_output, sample_id=0, hue_order=names)
    sns.set_palette(None)

    display(summarize_stats(stats_output).sort_values("parameter scans"))


# ===========================================================================
# Figure 10 — Joint scan: ε × θ
# ===========================================================================

def run_figure10():
    eps  = np.arange(1, 100, 1 * SPEED_UP_FACTOR) / 1000
    parameters = generate_parameters(eps=eps, theta=THETAS, z=[0.0])
    summary_stats, timeseries_df, df, N = simulate(parameters)
    stats_output, _ = beautify(parameters, summary_stats, timeseries_df, N)

    stats_output = (
        stats_output
        .groupby(["eps", "theta"])
        .mean(numeric_only=True)
        .reset_index()
        .merge(FLOW_MAP, on=["theta"], how="left")
    )

    plot_parameter_scan(
        stats_output,
        x_var="eps",
        variables=[
            "Internalization (%)",
            "Closing trade as a percent of out-flow (%)",
            "Internalization regret (%)",
            "Impact cost per in-flow (bps)",
        ],
        colors=COLORS,
        hue_order=HUE_ORDER,
        x_log_scale=True,
        vline=0.01,
        vline_label="Default ε",
    )


# ===========================================================================
# Figure 6 — Deep scan of σ
# ===========================================================================

def run_figure6():
    sigmas = np.arange(0, 100, 1 * SPEED_UP_FACTOR) / 1000
    parameters = generate_parameters(z=[0.03], theta=THETAS, sigma=sigmas)
    summary_stats, timeseries_df, df, N = simulate(parameters)
    stats_output, _ = beautify(parameters, summary_stats, timeseries_df, N)

    stats_output = (
        stats_output
        .groupby(["sigma", "theta"])
        .mean(numeric_only=True)
        .reset_index()
        .merge(FLOW_MAP, on=["theta"], how="left")
    )

    plot_parameter_scan(
        stats_output,
        x_var="sigma",
        variables=[
            "Internalization (%)",
            "Closing trade as a percent of out-flow (%)",
            "Internalization regret (%)",
            "Total cost per in-flow (bps)",
        ],
        colors=COLORS,
        hue_order=HUE_ORDER,
        vline=0.03,
        vline_label="Initial inventory",
    )


# ===========================================================================
# Figure 12 — Sensitivity to initial impact state y₀
# ===========================================================================

def run_figure12():
    ys    = [-0.003, 0.0, 0.003, 0.01]
    names = [
        "favorable initial impact",
        "zero initial impact",
        "moderate initial impact",
        "high initial impact",
    ]
    colors    = ["LimeGreen", "DodgerBlue", "SandyBrown", "Brown"]
    hue_order = names

    parameters = generate_parameters(y=ys, sigma=[0.01])
    summary_stats, timeseries_df, df, N = simulate(parameters)

    param_map = pd.DataFrame({"parameter scans": names, "y": ys})
    params = parameters.merge(param_map, on=["y"])
    stats_output, ts_output = beautify(params, summary_stats, timeseries_df, N)

    plot_timeseries(
        ts_output,
        sample_id=None,
        colors=colors,
        hue_order=hue_order,
        variables=["inventory (ADV%)", "impact state (bps)"],
        bbox_to_anchor_up_down=-0.15,
    )


# ===========================================================================
# Heatmap — Joint initial conditions (not in paper)
# ===========================================================================

def run_heatmap():
    sigmas = np.arange(10, 150, 2) / 1000
    ys     = np.arange(-30, 100, 2) / 10000

    parameters = generate_parameters(y=ys, sigma=sigmas)
    summary_stats, timeseries_df, df, N = simulate(
        parameters, nSamples=50, N=50
    )
    stats_output, _ = beautify(parameters, summary_stats, timeseries_df, N)

    stats_output = (
        stats_output
        .groupby(["y", "sigma"])
        .mean(numeric_only=True)
        .reset_index()
    )
    stats_output["Initial impact state (bps)"] = round(stats_output["y"] * 10000)
    stats_output["In-flow volatility (%)"]     = round(stats_output["sigma"] * 100, 1)

    fig, ax = plt.subplots(figsize=(5, 5))
    sns.heatmap(
        stats_output.pivot(
            index="Initial impact state (bps)",
            columns="In-flow volatility (%)",
            values="Internalization (%)",
        ),
        ax=ax,
        cmap="RdBu_r",
        center=0,
    )
    ax.set_title("Internalization (%)")
    plt.show()


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    print("=== Figure 9: ε sensitivity (deterministic) ===")
    run_figure9()

    print("=== Figure 10: ε × θ joint scan ===")
    run_figure10()

    print("=== Figure 6: σ deep scan ===")
    run_figure6()

    print("=== Figure 12: y₀ sensitivity ===")
    run_figure12()

    print("=== Heatmap: joint initial conditions ===")
    run_heatmap()
