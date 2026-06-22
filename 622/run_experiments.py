"""
run_experiments.py
==================
完整实验流程 demo（使用合成数据）

在拿到真实市场数据后，替换：
  - load_real_vega_flow()
  - load_historical_positions()
  - compute_vol_impact_from_data()
即可直接运行。
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys, os

sys.path.insert(0, os.path.dirname(__file__))

from vega_model import VegaParams, solve_vega_riccati, simulate_vega_flow, run_optimal_strategy
from calibration import (
    calibrate_ou, flow_replication_diagnostics,
    compute_historical_strategy_cost, cost_comparison_report
)
from plotting import (
    plot_riccati_coefficients, plot_experiment1_flow_replication,
    plot_experiment2_cost_comparison, plot_optimal_path
)

OUTPUT_DIR = "/mnt/user-data/outputs/"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 0. Parameters (replace with calibrated values from data)
# ============================================================
PARAMS = VegaParams(
    lambda_V=0.05,     # vol pts per $M vega executed
    beta=2.0,          # impact decay speed (1/day)
    eps_V=0.01,        # spread/execution cost
    theta=1.5,         # OU mean reversion speed
    sigma=0.3,         # OU volatility
    mu=0.0,            # OU mean level
    phi=0.2,           # gamma/theta running cost
    psi=10.0,          # soft terminal penalty
    T=1.0,             # 1 trading day
    n_steps=500,
)

RNG_SEED = 42
N_PATHS = 500
N_DAYS = 60           # backtest days


# ============================================================
# 1. Solve Riccati ODE
# ============================================================
print("=" * 60)
print("Step 1: Solving Vega Riccati ODE...")
coefs = solve_vega_riccati(PARAMS)

# Check coefficient signs (sanity)
print(f"  A(0) = {coefs.A[0]:.4f}  (should be ≥ 0)")
print(f"  f(0) = {coefs.f[0]:.4f}  (should be < 0: sell when long)")
print(f"  h(0) = {coefs.h[0]:.4f}  (flow sensitivity)")
print(f"  A(T) = {coefs.A[-1]:.4f}  (should = psi = {PARAMS.psi})")

fig_coefs = plot_riccati_coefficients(coefs, PARAMS)
fig_coefs.savefig(OUTPUT_DIR + "riccati_coefficients.png", dpi=150, bbox_inches='tight')
plt.close(fig_coefs)
print("  → Saved riccati_coefficients.png")


# ============================================================
# 2. Simulate a single path + plot
# ============================================================
print("\nStep 2: Simulating single optimal path...")
Z_paths = simulate_vega_flow(PARAMS, n_paths=1, seed=RNG_SEED)
Z_single = Z_paths[0]

result_single = run_optimal_strategy(Z_single, coefs, PARAMS, x0=0.0, y0=0.0)
print(f"  Total cost (optimal): {result_single['total_cost']:.6f}")

fig_path = plot_optimal_path(result_single, Z_single, coefs.t_grid)
fig_path.savefig(OUTPUT_DIR + "optimal_path_single.png", dpi=150, bbox_inches='tight')
plt.close(fig_path)
print("  → Saved optimal_path_single.png")


# ============================================================
# 3. Experiment 1: Flow Replication
# ============================================================
print("\n" + "=" * 60)
print("Experiment 1: Flow Replication Diagnostics")

# Generate "real" flow (in practice: load from market data)
rng = np.random.default_rng(42)
n_steps = PARAMS.n_steps
dt = PARAMS.T / (n_steps - 1)

# Simulate one "ground truth" path (pretend this is real data)
e = np.exp(-PARAMS.theta * dt)
v = np.sqrt(PARAMS.sigma**2 / (2*PARAMS.theta) * (1 - e**2))
Z_real = np.zeros(n_steps)
for i in range(1, n_steps):
    Z_real[i] = PARAMS.mu*(1-e) + Z_real[i-1]*e + v*rng.standard_normal()

# Step 1: Calibrate OU from "real" data
print("  Calibrating OU parameters from real flow...")
cal_result = calibrate_ou(Z_real, dt)
print(f"  theta_true={PARAMS.theta:.3f}, theta_hat={cal_result['theta']:.3f}")
print(f"  sigma_true={PARAMS.sigma:.3f}, sigma_hat={cal_result['sigma']:.3f}")
print(f"  Half-life: {cal_result['half_life']:.3f} days")

# Step 2: Simulate with calibrated params
params_cal = VegaParams(
    lambda_V=PARAMS.lambda_V,
    beta=PARAMS.beta,
    eps_V=PARAMS.eps_V,
    theta=cal_result['theta'],
    sigma=cal_result['sigma'],
    mu=cal_result['mu'],
    phi=PARAMS.phi,
    psi=PARAMS.psi,
    T=PARAMS.T,
    n_steps=PARAMS.n_steps,
)
Z_sim = simulate_vega_flow(params_cal, n_paths=N_PATHS, seed=123)

# Step 3: Diagnostics
diag = flow_replication_diagnostics(Z_real, Z_sim, dt, max_lags=30)
print(f"  KS test: stat={diag['ks_stat']:.4f}, p={diag['ks_pval']:.4f}")
print(f"  Momentum (real): {diag['momentum_real']:.4f}")
print(f"  Momentum (sim):  {diag['momentum_sim_mean']:.4f} ± {diag['momentum_sim_std']:.4f}")

fig_exp1 = plot_experiment1_flow_replication(diag)
fig_exp1.savefig(OUTPUT_DIR + "experiment1_flow_replication.png", dpi=150, bbox_inches='tight')
plt.close(fig_exp1)
print("  → Saved experiment1_flow_replication.png")


# ============================================================
# 4. Experiment 2: Strategy Cost Comparison (multi-day)
# ============================================================
print("\n" + "=" * 60)
print("Experiment 2: Optimal vs Historical Strategy Cost")

# params_fn: callables (use constant params for demo)
params_fn = {
    'lambda_V': lambda t: PARAMS.lambda_V,
    'beta':     lambda t: PARAMS.beta,
    'eps_V':    lambda t: PARAMS.eps_V,
    'phi':      lambda t: PARAMS.phi,
    'dgamma':   lambda t: 0.0,  # constant lambda → dgamma = 0
}

opt_costs_days = []
hist_costs_days = []

rng2 = np.random.default_rng(2024)

for day in range(N_DAYS):
    # Sample a vega flow path for this day
    Z_day = np.zeros(n_steps)
    for i in range(1, n_steps):
        Z_day[i] = PARAMS.mu*(1-e) + Z_day[i-1]*e + v*rng2.standard_normal()

    # Optimal strategy
    res_opt = run_optimal_strategy(Z_day, coefs, PARAMS, x0=0.0, y0=0.0)

    # Historical strategy: simulate a noisy/suboptimal strategy
    # In practice: use actual desk positions from market data
    # Here: we use a simple TWAP-like strategy with noise
    q_hist = np.zeros(n_steps)
    X_hist = np.zeros(n_steps)
    Y_hist = np.zeros(n_steps)

    # TWAP baseline: unwind flow at constant rate
    for i in range(n_steps - 1):
        remaining_time = PARAMS.T - coefs.t_grid[i]
        q_twap = (X_hist[i] + Z_day[i]) / max(remaining_time, 1e-6) * 0.5
        # Add noise to simulate imperfect real execution
        q_hist[i] = q_twap + 0.1 * rng2.standard_normal() * abs(Z_day[i])
        dZ = Z_day[i+1] - Z_day[i]
        X_hist[i+1] = X_hist[i] + q_hist[i] * dt - dZ
        Y_hist[i+1] = Y_hist[i] + (-PARAMS.beta * Y_hist[i] + PARAMS.lambda_V * q_hist[i]) * dt

    hist_cost_result = compute_historical_strategy_cost(
        q_hist, X_hist, Y_hist, coefs.t_grid, params_fn, PARAMS.psi
    )

    opt_costs_days.append(res_opt['total_cost'])
    hist_costs_days.append(hist_cost_result['total_cost'])

    if (day + 1) % 10 == 0:
        print(f"  Day {day+1}/{N_DAYS}: opt={res_opt['total_cost']:.5f}, "
              f"hist={hist_cost_result['total_cost']:.5f}")

opt_costs_days  = np.array(opt_costs_days)
hist_costs_days = np.array(hist_costs_days)

df_comp, summary = cost_comparison_report(
    opt_costs_days, hist_costs_days,
    dates=[f"Day {i+1}" for i in range(N_DAYS)]
)

print("\n  === Cost Comparison Summary ===")
for k, v in summary.items():
    print(f"  {k:30s}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

fig_exp2 = plot_experiment2_cost_comparison(df_comp, summary)
fig_exp2.savefig(OUTPUT_DIR + "experiment2_cost_comparison.png", dpi=150, bbox_inches='tight')
plt.close(fig_exp2)
print("\n  → Saved experiment2_cost_comparison.png")

# Save cost data
df_comp.to_csv(OUTPUT_DIR + "cost_comparison.csv", index=False)
print("  → Saved cost_comparison.csv")

print("\n" + "=" * 60)
print("All experiments complete. Outputs saved to:", OUTPUT_DIR)
print("=" * 60)
