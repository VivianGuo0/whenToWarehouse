"""
demo.py
-------
End-to-end demonstration of the refactored CRB framework.

Covers:
  1. Constant-parameter run (paper's base case, closed-form solver)
  2. Time-varying theta run (ODE solver auto-selected)
  3. Monte Carlo run with metrics aggregation
  4. Full sanity check report
  5. Cross-validation: closed-form vs ODE
"""

import numpy as np

from params import ModelParams, LiquidityParams, FlowParams, InitialConditions
from coefficients import CoefficientEngine, SolverMode
from simulator import OUSimulator, StrategySimulator, MonteCarloEngine
from metrics import MetricsCalculator, aggregate_metrics
from sanity_check import SanityChecker


# ---------------------------------------------------------------------------
# 1. Constant-parameter base case (Section 3.1 defaults)
# ---------------------------------------------------------------------------
print("=" * 60)
print("1. Constant-parameter run (closed-form solver)")
print("=" * 60)

params_const = ModelParams.constant(
    beta=8.0, lam=0.2, eps=0.01,
    theta=-1.0,   # momentum in-flow
    sigma=0.1,
    z0=-0.1, y0=0.0, x0=0.0,
    T=1.0, n_steps=390,
)

engine = CoefficientEngine(verbose=True)
coeffs = engine.compute(params_const)
print(f"  f[0]={coeffs.f[0]:.4f}, g[0]={coeffs.g[0]:.4f}, h[0]={coeffs.h[0]:.4f}")
print(f"  Opening block J_0 = {coeffs.opening_block_trade():.6f}")

# Simulate one path
ou  = OUSimulator(n_shocks=20, rng=np.random.default_rng(42))
Z   = ou.simulate(params_const)
sim = StrategySimulator(coeffs)
res = sim.run(Z)
print(f"  Terminal inventory X_T = {res.X[-1]:.6f}")

calc    = MetricsCalculator(params_const)
metrics = calc.compute(res)
print(f"  Internalization rate = {metrics.internalization_rate*100:.1f}%")
print(f"  Total cost = {metrics.total_cost_bps:.4f} per unit in-flow")


# ---------------------------------------------------------------------------
# 2. Time-varying theta (ODE solver)
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("2. Time-varying theta run (ODE solver)")
print("=" * 60)

n_steps = 390
# theta ramps from -2 (strong momentum at open) to +0.5 (mild reversal at close)
theta_tv = np.linspace(-2.0, 0.5, n_steps)
sigma_tv = 0.1  # constant sigma

params_tv = ModelParams(
    liquidity=LiquidityParams(beta=8.0, lam=0.2, eps=0.01),
    flow=FlowParams(theta=theta_tv, sigma=sigma_tv),
    init=InitialConditions(x0=0.0, y0=0.0, z0=-0.1),
    T=1.0,
    n_steps=n_steps,
)

engine_ode = CoefficientEngine(verbose=True)
coeffs_tv  = engine_ode.compute(params_tv)  # auto-selects ODE
print(f"  f[0]={coeffs_tv.f[0]:.4f}, g[0]={coeffs_tv.g[0]:.4f}, h[0]={coeffs_tv.h[0]:.4f}")
print(f"  Opening block J_0 = {coeffs_tv.opening_block_trade():.6f}")


# ---------------------------------------------------------------------------
# 3. Monte Carlo run
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("3. Monte Carlo run (1000 paths, momentum)")
print("=" * 60)

mc_engine = MonteCarloEngine(
    coeffs=coeffs,
    n_paths=1000,
    seed=0,
)
mc_result = mc_engine.run()
summary   = aggregate_metrics(mc_result, params_const)
print(summary)


# ---------------------------------------------------------------------------
# 4. Full sanity check
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("4. Sanity check report (constant params)")
print("=" * 60)

checker = SanityChecker(tol=1e-4, raise_on_fail=False)
report  = checker.full_check(
    params=params_const,
    coeffs=coeffs,
    sim_result=res,
    cross_validate=True,
)
print(report)


# ---------------------------------------------------------------------------
# 5. Time-varying params sanity check (ODE only, no cross-validation)
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("5. Sanity check report (time-varying params)")
print("=" * 60)

ou_tv   = OUSimulator(n_shocks=20, rng=np.random.default_rng(7))
Z_tv    = ou_tv.simulate(params_tv)
res_tv  = StrategySimulator(coeffs_tv).run(Z_tv)

report_tv = checker.full_check(
    params=params_tv,
    coeffs=coeffs_tv,
    sim_result=res_tv,
    cross_validate=False,   # no closed-form available for time-varying
)
print(report_tv)
