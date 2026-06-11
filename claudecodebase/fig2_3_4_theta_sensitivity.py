"""
Figures 2, 3, 4 — Sensitivity to the in-flow mean-reversion parameter θ.

Scans across three flow regimes:
  - θ > 0 : momentum in-flow
  - θ = 0 : martingale in-flow
  - θ < 0 : reversal in-flow
"""

import sys
sys.path.insert(0, "..")

import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display

from src import (
    generate_parameters,
    simulate,
    beautify,
    summarize_stats,
    plot_timeseries,
    plot_distributions,
    DEFAULT_COLORS,
    DEFAULT_HUE_ORDER,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SPEED_UP_FACTOR = 10  # set to 1 for publication-quality plots

THETAS    = [0.0, 1.0, -1.0]
FLOW_MAP  = pd.DataFrame({
    "parameter scans": ["martingale in-flow", "reversal in-flow", "momentum in-flow"],
    "theta": THETAS,
})

STATS_TO_PLOT = [
    "Total variation of in-flow (ADV%)",
    "Internalization (%)",
    "Impact cost per in-flow (bps)",
    "Total variation of out-flow (ADV%)",
    "Internalization regret (%)",
    "Spread cost per in-flow (bps)",
]

COLORS    = DEFAULT_COLORS
HUE_ORDER = DEFAULT_HUE_ORDER

# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

parameters = generate_parameters(theta=THETAS)
summary_stats, timeseries_df, df, N = simulate(
    parameters, nSamples=int(5000 / SPEED_UP_FACTOR)
)
params = parameters.merge(FLOW_MAP, on=["theta"])
stats_output, ts_output = beautify(params, summary_stats, timeseries_df, N)

# ---------------------------------------------------------------------------
# Figure 2 — single representative path
# ---------------------------------------------------------------------------

plot_timeseries(ts_output, sample_id=0, colors=COLORS, hue_order=HUE_ORDER)

# ---------------------------------------------------------------------------
# Figure 3 — cross-sectional mean ± s.d.
# ---------------------------------------------------------------------------

plot_timeseries(ts_output, sample_id=None, colors=COLORS, hue_order=HUE_ORDER)

# ---------------------------------------------------------------------------
# Figure 4 — distribution of key statistics
# ---------------------------------------------------------------------------

plot_distributions(stats_output, stats=STATS_TO_PLOT, colors=COLORS, hue_order=HUE_ORDER)

# ---------------------------------------------------------------------------
# Table 2 — summary statistics
# ---------------------------------------------------------------------------

display(summarize_stats(stats_output).loc[HUE_ORDER, :])

# ---------------------------------------------------------------------------
# Internalization regret quantiles
# ---------------------------------------------------------------------------

stat_name = "Internalization regret (%)"
thresholds = [0, 0.1, 0.2, 0.5, 1, 2, 5, 10]

for threshold in thresholds:
    subset = stats_output[stats_output[stat_name] <= threshold]
    counts = subset.groupby("parameter scans").count().iloc[:, 0]
    total = stats_output.groupby("parameter scans").count().iloc[:, 0]
    prob = counts / total * 100
    print(f"Probability below {threshold}%")
    print(prob[HUE_ORDER])
    print()
