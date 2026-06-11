# Stochastic Inventory Management Simulation

Simulation codebase for a stochastic optimal execution / market-making model with
mean-reverting in-flow.

## Model

The state dynamics are

```
X_{t+dt} = X_t + q_t * dt - (Z_{t+dt} - Z_t)      # inventory
Y_{t+dt} = Y_t + (-beta * Y_t + lambda * q_t) * dt  # impact state
Z_{t+dt} = Z_t - theta * Z_t * dt + sigma * dW_t    # in-flow
```

where `q` is the optimal trading rate derived from the associated HJB equation.

## Project layout

```
simulation_project/
├── src/
│   ├── __init__.py          # Public API
│   ├── simulation.py        # Core: parameters, time grid, path simulation
│   ├── postprocessing.py    # Unit conversion (ADV%, bps) and summary stats
│   └── plotting.py          # Reusable plot functions
├── experiments/
│   ├── fig1_extreme_paths.py              # Figure 1
│   ├── fig2_3_4_theta_sensitivity.py      # Figures 2, 3, 4 + Table 2
│   ├── fig6_9_10_12_parameter_scans.py    # Figures 6, 9, 10, 12 + heatmap
│   └── fig5_7_8_11_13_advanced.py        # Figures 5, 7, 8, 11, 13
└── notebooks/
    └── main.ipynb           # Orchestrator notebook (runs all experiments)
```

## Quick start

```python
from src import generate_parameters, simulate, beautify, plot_timeseries

params = generate_parameters(theta=[0.0, 1.0, -1.0])
summary, ts, df, N = simulate(params, nSamples=1000)
stats, ts_plot = beautify(params, summary, ts, N)
plot_timeseries(ts_plot)
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `y`       | 0.0     | Initial impact state |
| `z`       | 0.1     | Initial in-flow (= initial inventory) |
| `lamb`    | 0.2     | Impact loading coefficient λ |
| `beta`    | 8.0     | Impact decay rate β |
| `eps`     | 0.01    | Spread cost parameter ε |
| `theta`   | 0.0     | In-flow mean-reversion speed θ |
| `sigma`   | 0.1     | In-flow volatility σ |

## Speed-up factor

Set `SPEED_UP_FACTOR = 1` in each experiment file for publication-quality plots
(larger sample sizes). The default `= 10` keeps runtimes fast for exploration.

## External data

Figure 13 requires `empiricalTheta.csv` with columns `theta` and `horizon` (in minutes).
