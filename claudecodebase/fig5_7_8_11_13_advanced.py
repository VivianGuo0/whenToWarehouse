"""
Advanced experiments:
  - Figure 5  : total cost distribution conditional on internalization level
  - Figure 7  : path monotonicity across σ
  - Figure 8  : misspecification costs (three true-θ scenarios)
  - Figure 11 : diffusive limit as number of shocks → ∞
  - Figure 13 : empirical θ from market data (requires empiricalTheta.csv)
"""

import sys
sys.path.insert(0, "..")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src import (
    generate_parameters,
    add_auxiliary_variables,
    add_time_grid,
    add_helper_variables,
    simulate_path,
    simulate,
    beautify,
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
# Figure 5 — Cost distribution conditional on internalization
# ===========================================================================

def run_figure5():
    sigmas = np.arange(1, 15, 2) / 10
    parameters = generate_parameters(sigma=sigmas)
    summary_stats, timeseries_df, df, N = simulate(
        parameters,
        nSamples=int(10000 / SPEED_UP_FACTOR ** 2),
        N=50,
    )
    stats_output, _ = beautify(parameters, summary_stats, timeseries_df, N)
    stats_output["In-flow volatility (%)"] = stats_output["sigma"] * 100

    # Bin samples by internalization level
    for level, center in [("low", 60), ("medium", 75), ("high", 90)]:
        mask = np.abs(stats_output["Internalization (%)"] - center) < 5
        stats_output.loc[mask, "internalization"] = level

    sns.set_palette("mako")
    fig, ax = plt.subplots(figsize=(6, 3))
    sns.kdeplot(
        data=stats_output,
        x="Total cost per in-flow (bps)",
        hue="internalization",
        hue_order=["high", "medium", "low"],
        ax=ax,
        common_norm=False,
    )
    ax.set_title("Total cost distribution conditional on internalization")
    ax.set_xlim([-3, 80])
    plt.show()
    sns.set_palette(None)


# ===========================================================================
# Figure 7 — Path monotonicity
# ===========================================================================

def run_figure7():
    sigmas  = [0.01, 0.07, 0.2]
    mapping = {0.01: "low", 0.07: "medium", 0.2: "high"}
    colors    = ["SkyBlue", "DarkGoldenrod", "Maroon"]
    hue_order = ["high", "medium", "low"]

    parameters = generate_parameters(sigma=sigmas)
    summary_stats, timeseries_df, df, N = simulate(
        parameters,
        nSamples=int(200000 / SPEED_UP_FACTOR ** 2),
        N=50,
    )
    stats_output, _ = beautify(parameters, summary_stats, timeseries_df, N)
    stats_output["In-flow volatility (%)"] = round(stats_output["sigma"] * 100, 1)
    stats_output["Terminal in-flow over TV of in-flow (%)"] = round(
        stats_output["Terminal in-flow (ADV%)"] / stats_output["Total variation of in-flow (ADV%)"] * 100, 1
    )
    stats_output["sigma"] = stats_output["sigma"].replace(mapping)

    x_var = "Terminal in-flow over TV of in-flow (%)"
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(14, 4))
    for ax, variable in zip(axs.flatten(), ["Internalization (%)", "Internalization regret (%)"]):
        sns.lineplot(
            data=stats_output,
            x=x_var,
            y=variable,
            hue="sigma",
            hue_order=hue_order,
            palette=colors,
            ax=ax,
            errorbar="sd",
        )
        ax.set_title(variable)
        ax.get_legend().remove()
    plt.show()

    # Density of x_var across σ
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.kdeplot(
        data=stats_output,
        x=x_var,
        hue="sigma",
        hue_order=hue_order,
        palette=colors,
        ax=ax,
    )
    ax.set_title("Density of terminal in-flow over TV of in-flow across σ")
    legend_lines = [plt.Line2D([0], [0], color=c, linewidth=2) for c in colors]
    ax.legend(legend_lines, hue_order, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
    plt.tight_layout()
    plt.show()


# ===========================================================================
# Figure 8 — Misspecification costs
# ===========================================================================

def _misspecification_scenario(true_theta: float, color: str):
    """Run one misspecification scenario and produce the three-panel plot."""
    thetas = np.arange(-100, 102, 2 * SPEED_UP_FACTOR) / 100 + true_theta
    T, N = 1.0, int(200 / SPEED_UP_FACTOR)
    nSamples, nShocks = int(2000 / SPEED_UP_FACTOR), int(200 / SPEED_UP_FACTOR)

    parameters = generate_parameters(theta=thetas)
    parameters["misspecified theta"] = parameters["theta"].copy()

    # Build helper functions under misspecified θ, then fix true θ for simulation
    parameters = add_auxiliary_variables(parameters)
    df, T, N, dt = add_time_grid(parameters, T, N)
    df, T = add_helper_variables(df, T)
    parameters["theta"] = true_theta
    df["theta"] = true_theta

    summary_stats, timeseries_df = simulate_path(df, dt, N, nSamples, nShocks)
    stats_output, _ = beautify(parameters, summary_stats, timeseries_df, N)
    stats_output["Impact cost per out-flow (bps)"] = (
        stats_output["Total cost per in-flow (bps)"]
        * stats_output["Total variation of in-flow (ADV%)"]
        / stats_output["Total variation of out-flow (ADV%)"]
    )

    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(14, 4))
    for ax, variable in zip(axs.flatten(), [
        "Total cost per in-flow (bps)", "Internalization (%)", "Impact cost per out-flow (bps)"
    ]):
        sns.lineplot(data=stats_output, x="misspecified theta", y=variable, ax=ax, color=color)
        ax.set_title(variable)
        ax.axvline(x=true_theta, linestyle="--", color="black")
        ax.text(
            true_theta + 0.03, ax.get_ylim()[0], "actual θ", verticalalignment="bottom"
        )
    plt.tight_layout()
    plt.show()


def run_figure8():
    for true_theta, color in [(1.0, GREEN_HEX), (0.0, BLUE_HEX), (-1.0, ORANGE_HEX)]:
        print(f"True θ = {true_theta}")
        _misspecification_scenario(true_theta, color)


# ===========================================================================
# Figure 11 — Diffusive limit
# ===========================================================================

def run_figure11():
    parameters = generate_parameters(theta=THETAS)

    def _one_shock_level(N_shocks):
        summary_stats, timeseries_df, df, N = simulate(
            parameters,
            nShocks=N_shocks,
            N=N_shocks,
            nSamples=int(5000 / SPEED_UP_FACTOR ** 2),
        )
        stats_output, _ = beautify(parameters, summary_stats, timeseries_df, N)
        stats_output = stats_output.groupby("theta").mean(numeric_only=True).reset_index()
        stats_output["Number of shocks"] = N_shocks
        return stats_output

    results = pd.concat(
        [_one_shock_level(10 * (2 ** i)) for i in range(1, 11)]
    ).reset_index(drop=True).merge(FLOW_MAP, on=["theta"])

    def _panel(stats, special_ylim=None):
        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(11, 3))
        for i, (ax, variable) in enumerate(zip(axs.flatten(), stats)):
            sns.lineplot(
                data=results,
                x="Number of shocks",
                y=variable,
                hue="parameter scans",
                hue_order=HUE_ORDER,
                palette=COLORS,
                ax=ax,
            )
            ax.set_xscale("log")
            ax.set_title(variable)
            ax.legend().remove()
            if special_ylim is not None and i == 1:
                ax.set_ylim(*special_ylim)
        handles, labels = axs[0].get_legend_handles_labels()
        fig.legend(handles=handles, labels=labels, loc="center",
                   bbox_to_anchor=(0.5, -0.13), ncol=len(handles))
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.13)
        plt.show()

    _panel(
        ["Total variation of in-flow (ADV%)", "Sqrt QV of in-flow (ADV%)", "Internalization (%)"],
        special_ylim=(10, 20),
    )
    _panel(["Terminal impact state (bps)", "Impact cost", "Internalization regret (%)"])
    _panel([
        "Total variation of out-flow (ADV%)",
        "Closing trade as a percent of out-flow (%)",
        "Out-flow TV over in-flow sqrtQV (%)",
    ])


# ===========================================================================
# Figure 13 — Empirical θ (requires empiricalTheta.csv in working directory)
# ===========================================================================

def run_figure13(csv_path="empiricalTheta.csv"):
    x_name  = "log(2)/theta (days)"
    hue_name = "horizon (minutes)"
    hue_order = ["1/6", "1", "5", "30", "60"]

    df = pd.read_csv(csv_path)
    df[x_name] = np.log(2) / df["theta"] / (60 * 6.5)  # minutes → trading days

    mapping = {"0.1666667": "1/6", "1.0": "1", "5.0": "5", "30.0": "30", "60.0": "60"}
    df[hue_name] = df["horizon"].astype(str).replace(mapping)

    sns.set_palette("mako")
    fig = sns.kdeplot(data=df, x=x_name, hue=hue_name, hue_order=hue_order, common_norm=False)
    fig.set_xlim([0, 2])
    plt.show()
    sns.set_palette(None)


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    print("=== Figure 5: cost distribution ===")
    run_figure5()

    print("=== Figure 7: path monotonicity ===")
    run_figure7()

    print("=== Figure 8: misspecification costs ===")
    run_figure8()

    print("=== Figure 11: diffusive limit ===")
    run_figure11()

    # Figure 13 requires external data
    # run_figure13("path/to/empiricalTheta.csv")
