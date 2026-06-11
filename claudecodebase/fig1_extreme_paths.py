"""
Figure 1 — Illustration of two extreme trading paths.

Identifies the sample path closest to pure optimal execution
and the one closest to pure market making, then plots them side by side.
"""

import sys
sys.path.insert(0, "..")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src import (
    generate_parameters,
    simulate,
    beautify,
    OLIVE_HEX,
    RED_HEX,
)

SPEED_UP_FACTOR = 10


def run_figure1():
    parameters = generate_parameters(z=[0.01])
    summary_stats, timeseries_df, df, N = simulate(
        parameters,
        N=int(2000 / SPEED_UP_FACTOR),
        nSamples=int(1000 / SPEED_UP_FACTOR),
        nShocks=200,
    )
    stats_output, ts_output = beautify(parameters, summary_stats, timeseries_df, N)

    # Rank paths by "terminal in-flow as a share of total-variation" to
    # separate market-making paths (high terminal fraction) from
    # optimal-execution paths (low terminal fraction)
    stats_output["Terminal in-flow over TV of in-flow (%)"] = (
        stats_output["Terminal in-flow (ADV%)"]
        / stats_output["Total variation of in-flow (ADV%)"]
        * 100
    )
    stats_output = stats_output.sort_values(
        "Terminal in-flow over TV of in-flow (%)", ascending=False
    )

    idx_exec = int(np.ceil(0.01 * len(stats_output)) - 1)
    idx_mm   = len(stats_output) - 1
    sample_ids = stats_output.iloc[[idx_exec, idx_mm]]["sample_id"].values

    samples = pd.DataFrame({
        "sample": ["Optimal execution path", "Market making path"],
        "sample_id": sample_ids,
    })

    colors    = [OLIVE_HEX, RED_HEX]
    hue_order = ["Market making path", "Optimal execution path"]

    ts_plot = (
        ts_output[ts_output["sample_id"].isin(sample_ids)]
        .merge(samples, on="sample_id")
    )

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(11, 2))
    for ax, variable in zip(axes, ["in-flow (ADV%)", "out-flow (ADV%)", "inventory (ADV%)"]):
        sns.lineplot(
            data=ts_plot,
            x="time",
            y=variable,
            hue="sample",
            hue_order=hue_order,
            palette=colors,
            ax=ax,
        )
        ax.set_title(f"Time series of {variable}")
        ax.set_xlabel("Time (hours)")
        ax.set_ylabel(None)
        ax.get_legend().remove()

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles=handles, labels=labels, loc="lower center",
               bbox_to_anchor=(0.5, -0.25), ncol=2)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    plt.show()


if __name__ == "__main__":
    run_figure1()
