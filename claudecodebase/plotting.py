"""
Plotting utilities for simulation outputs.

All functions follow the same color/hue_order conventions used in the paper.
"""

import math
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Default palette
# ---------------------------------------------------------------------------

_pal = sns.color_palette("tab10")
_hex = _pal.as_hex()
BLUE_HEX   = _hex[0]
ORANGE_HEX = _hex[1]
GREEN_HEX  = _hex[2]
RED_HEX    = _hex[3]
OLIVE_HEX  = _hex[8]

DEFAULT_COLORS    = [ORANGE_HEX, BLUE_HEX, GREEN_HEX]
DEFAULT_HUE_ORDER = ["momentum in-flow", "martingale in-flow", "reversal in-flow"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_colors(colors, n_categories):
    if colors is None:
        return None
    if not isinstance(colors, list):
        raise TypeError("colors must be a list or None")
    if len(colors) != n_categories:
        print(
            f"colors must have length {n_categories}; falling back to defaults."
        )
        return None
    return colors


def _validate_hue_order(hue_order, unique_vals):
    if hue_order is None:
        return None
    if not isinstance(hue_order, list):
        raise TypeError("hue_order must be a list or None")
    if len(hue_order) != len(unique_vals):
        print("hue_order length mismatch; falling back to defaults.")
        return None
    return hue_order


# ---------------------------------------------------------------------------
# Public plotting functions
# ---------------------------------------------------------------------------

def plot_timeseries(
    ts_output: pd.DataFrame,
    sample_id=None,
    colors=None,
    hue_order=None,
    variables=None,
    bbox_to_anchor_up_down=-0.05,
    pathToSave=None,
):
    """
    Plot mean ± 1 s.d. time series for one or more state variables.

    Parameters
    ----------
    ts_output : pd.DataFrame
        Output of ``beautify``.
    sample_id : int or None
        If provided, plot only this Monte Carlo path; otherwise plot the
        cross-sectional mean ± s.d.
    colors : list[str] or None
        One color per ``parameter scans`` category.
    hue_order : list[str] or None
        Display order for ``parameter scans`` categories.
    variables : list[str] or None
        State variables to plot. Defaults to all four state variables.
    bbox_to_anchor_up_down : float
        Vertical position of the shared legend.
    pathToSave : str or None
        If provided, save the figure to this path.
    """
    if variables is None:
        variables = [
            "in-flow (ADV%)",
            "out-flow (ADV%)",
            "inventory (ADV%)",
            "impact state (bps)",
        ]

    if sample_id is not None:
        ts_output = ts_output[ts_output["sample_id"] == sample_id]

    n_categories = ts_output["parameter scans"].nunique()
    colors = _validate_colors(colors, n_categories)
    hue_order = _validate_hue_order(hue_order, ts_output["parameter scans"].unique())

    nrows = len(variables) // 2
    fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=(11, 3.5 * nrows))
    axes = axes.flatten()

    for ax, variable in zip(axes, variables):
        sns.lineplot(
            data=ts_output,
            x="time",
            y=variable,
            hue="parameter scans",
            hue_order=hue_order,
            ax=ax,
            errorbar="sd",
            palette=colors,
        )
        ax.set_title(f"Time series of {variable}")
        ax.set_xlabel("Time (hours)")
        ax.set_ylabel(None)
        ax.legend().remove()

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles=handles,
        labels=labels,
        title=None,
        loc="center",
        bbox_to_anchor=(0.5, bbox_to_anchor_up_down),
        ncol=len(handles),
    )
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.05)
    plt.show()

    if pathToSave is not None:
        fig.savefig(pathToSave)


def plot_distributions(
    stats_output: pd.DataFrame,
    stats=None,
    colors=None,
    hue_order=None,
):
    """
    KDE plots of scalar statistics across parameter groups.

    Parameters
    ----------
    stats_output : pd.DataFrame
        Output of ``beautify``.
    stats : list[str] or None
        Columns to plot. Defaults to key trading metrics.
    colors : list[str] or None
    hue_order : list[str] or None
    """
    if stats is None:
        stats = [
            "Internalization (%)",
            "Closing trade as a percent of out-flow (%)",
            "Total cost per in-flow (bps)",
        ]

    n_categories = stats_output["parameter scans"].nunique()
    colors = _validate_colors(colors, n_categories)
    hue_order = (
        hue_order
        if _validate_hue_order(hue_order, stats_output["parameter scans"].unique()) is not None
        else stats_output["parameter scans"].unique().tolist()
    )

    ncols = 3
    nrows = math.ceil(len(stats) / ncols)
    fig, axs = plt.subplots(nrows, ncols, figsize=(11, 7))

    # Hide unused axes
    for extra in range(len(stats), nrows * ncols):
        fig.delaxes(axs.flatten()[extra])

    axs = axs.flatten()
    for i, stat in enumerate(stats):
        sns.kdeplot(
            data=stats_output,
            x=stat,
            hue="parameter scans",
            hue_order=hue_order,
            palette=colors,
            ax=axs[i],
        )
        axs[i].set_title(stat)
        axs[i].set_ylabel("Density")
        axs[i].set_xlabel(None)
        axs[i].get_legend().remove()

    if colors is not None:
        legend_lines = [
            plt.Line2D([0], [0], color=c, linewidth=2) for c in colors
        ]
        fig.legend(
            legend_lines,
            hue_order,
            title=None,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.05),
            ncol=len(colors),
        )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.05)
    plt.show()


def plot_parameter_scan(
    stats_output: pd.DataFrame,
    x_var: str,
    variables: list,
    colors=None,
    hue_order=None,
    x_log_scale: bool = False,
    vline: float = None,
    vline_label: str = None,
    figsize=(11, 7),
):
    """
    Line plots of statistics as a function of a scalar parameter.

    Parameters
    ----------
    stats_output : pd.DataFrame
        Grouped (averaged) output with ``parameter scans`` and ``x_var`` columns.
    x_var : str
        Column name for the x-axis.
    variables : list[str]
        Statistics to plot (max 4 for a 2×2 layout).
    x_log_scale : bool
        Whether to use a log scale on the x-axis.
    vline : float or None
        Draw a vertical reference line at this value.
    vline_label : str or None
        Label for the vertical reference line.
    """
    n_categories = stats_output["parameter scans"].nunique()
    colors = _validate_colors(colors, n_categories)
    hue_order = _validate_hue_order(hue_order, stats_output["parameter scans"].unique())

    nrows = math.ceil(len(variables) / 2)
    fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=figsize)
    axes = axes.flatten()

    for ax, variable in zip(axes, variables):
        if x_log_scale:
            ax.set_xscale("log")
        sns.lineplot(
            data=stats_output,
            x=x_var,
            y=variable,
            hue="parameter scans",
            hue_order=hue_order,
            palette=colors,
            ax=ax,
        )
        ax.set_title(f"{variable} across {x_var}")
        ax.set_xlabel(x_var)
        ax.set_ylabel(None)
        ax.legend().remove()

        if vline is not None:
            ax.axvline(x=vline, linestyle="--", color="black")
            if vline_label:
                ax.text(
                    vline * 1.06,
                    ax.get_ylim()[1] - 0.4,
                    vline_label,
                    verticalalignment="top",
                )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles=handles,
        labels=labels,
        title=None,
        loc="center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=len(handles),
    )
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.05)
    plt.show()
