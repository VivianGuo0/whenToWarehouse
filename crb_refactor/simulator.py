"""
simulator.py
------------
Simulation of the OU in-flow process Z and execution of the optimal strategy.

Classes
-------
OUSimulator      : generates Z paths (discrete shock driver, Section 3.1)
StrategySimulator: runs (X, Y, Q) forward given Z path and CoefficientsResult
SimulationResult : frozen dataclass holding one path's state trajectories
MonteCarloEngine : runs N_paths independent paths and aggregates results
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from params import ModelParams
from coefficients import CoefficientsResult


# ---------------------------------------------------------------------------
# OU In-flow Simulator
# ---------------------------------------------------------------------------

class OUSimulator:
    """
    Simulates the in-flow process Z using the discrete shock driver (Section 3.1).

    dZ_t = -θ_t Z_t dt + σ_t dW_t

    Discretised as a sequence of n_shocks i.i.d. Gaussian block shocks,
    spread evenly over [0, T].  Between shocks Z evolves deterministically.

    Parameters
    ----------
    n_shocks : number of intraday block orders (paper default: 20)
    rng      : numpy random Generator (pass for reproducibility)
    """

    def __init__(
        self,
        n_shocks: int = 20,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.n_shocks = n_shocks
        self.rng = rng or np.random.default_rng()

    def simulate(self, params: ModelParams) -> np.ndarray:
        """
        Simulate one Z path on params.time_grid.

        Returns
        -------
        Z : np.ndarray of shape (n_steps + 1,)
        """
        t_grid   = params.time_grid
        n_steps  = params.n_steps
        T        = params.T
        z0       = params.init.z0
        theta_arr, sigma_arr = params.resolved_flow()

        # Shock arrival times (evenly spaced)
        shock_times = np.linspace(0.0, T, self.n_shocks + 1)[1:]  # exclude t=0

        # Standard deviations of shocks: σ·√(T/n_shocks)
        dt_shock = T / self.n_shocks
        # Use sigma at each shock time (piecewise constant interp)
        shock_t_idx = np.searchsorted(t_grid[:-1], shock_times, side="right") - 1
        shock_t_idx = np.clip(shock_t_idx, 0, n_steps - 1)
        sigma_shock = sigma_arr[shock_t_idx]
        shocks = self.rng.normal(0.0, sigma_shock * np.sqrt(dt_shock), self.n_shocks)

        Z = np.zeros(n_steps + 1)
        Z[0] = z0

        shock_ptr = 0
        for k in range(n_steps):
            t_k   = t_grid[k]
            t_k1  = t_grid[k + 1]
            dt    = t_k1 - t_k
            theta = theta_arr[k]

            # Deterministic OU decay
            Z[k + 1] = Z[k] * np.exp(-theta * dt)

            # Inject shock if a shock falls in (t_k, t_{k+1}]
            while shock_ptr < self.n_shocks and shock_times[shock_ptr] <= t_k1:
                Z[k + 1] += shocks[shock_ptr]
                shock_ptr += 1

        return Z


# ---------------------------------------------------------------------------
# Strategy Simulator
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SimulationResult:
    """
    Full state trajectory for one simulation path.

    All arrays have shape (n_steps + 1,).

    X : inventory   X_t = Q_t - Z_t
    Y : impact state Y_t
    Z : in-flow cumulative Z_t
    Q : cumulative unwind trades Q_t
    q : intraday trading speed (rate) q_t  [shape (n_steps,)]
    J0: opening block trade (scalar)
    total_cost : realized total cost C (impact + spread)
    """
    t_grid:      np.ndarray
    X:           np.ndarray
    Y:           np.ndarray
    Z:           np.ndarray
    Q:           np.ndarray
    q:           np.ndarray    # shape (n_steps,)
    J0:          float
    total_cost:  float


class StrategySimulator:
    """
    Executes the optimal feedback strategy q*_t = f_t X_t + g_t Y_t + h_t Z_t
    given a realised Z path, using the pre-computed CoefficientsResult.

    Parameters
    ----------
    coeffs : CoefficientsResult from CoefficientEngine.compute()
    """

    def __init__(self, coeffs: CoefficientsResult) -> None:
        self.coeffs = coeffs

    def run(self, Z: np.ndarray) -> SimulationResult:
        """
        Run one path given Z (shape (n_steps+1,)).

        Returns SimulationResult with full trajectories.
        """
        params  = self.coeffs.params
        liq     = params.liquidity
        init    = params.init
        t_grid  = params.time_grid
        n_steps = params.n_steps
        dt      = params.dt
        f, g, h = self.coeffs.f, self.coeffs.g, self.coeffs.h

        beta = liq.beta
        lam  = liq.lam
        eps  = liq.eps

        # Opening block trade
        J0 = self.coeffs.opening_block_trade()

        # Initialise state (post-open block trade absorbed)
        X = np.zeros(n_steps + 1)
        Y = np.zeros(n_steps + 1)
        Q = np.zeros(n_steps + 1)
        q = np.zeros(n_steps)

        # Post-open: Q[0] = J0 (the opening block trade), Z[0] = z0,
        # so inventory X[0] = Q[0] - Z[0].  init.x0 is only used when J0=0.
        Q[0] = J0
        X[0] = Q[0] - Z[0]   # enforces identity X_t = Q_t - Z_t at t=0
        Y[0] = init.y0

        cost_accum = 0.0

        for k in range(n_steps):
            t_k = t_grid[k]

            # Optimal speed
            q_k = f[k] * X[k] + g[k] * Y[k] + h[k] * Z[k]
            q[k] = q_k

            # State update (Euler discretisation of eqs. 2.14)
            dZ = Z[k + 1] - Z[k]
            X[k + 1] = X[k] + q_k * dt - dZ
            Y[k + 1] = Y[k] * (1.0 - beta * dt) + lam * q_k * dt
            Q[k + 1] = Q[k] + q_k * dt

            # Running cost: ε/2 · q² (spread) + Y · q (impact)  — per unit time
            cost_accum += (0.5 * eps * q_k**2 + Y[k] * q_k) * dt

        return SimulationResult(
            t_grid=t_grid,
            X=X, Y=Y, Z=Z, Q=Q, q=q,
            J0=J0,
            total_cost=cost_accum,
        )


# ---------------------------------------------------------------------------
# Monte Carlo Engine
# ---------------------------------------------------------------------------

@dataclass
class MCResult:
    """Aggregated results from N_paths Monte Carlo runs."""
    paths:          list[SimulationResult] = field(default_factory=list)

    @property
    def n_paths(self) -> int:
        return len(self.paths)

    def costs(self) -> np.ndarray:
        return np.array([p.total_cost for p in self.paths])

    def J0s(self) -> np.ndarray:
        return np.array([p.J0 for p in self.paths])

    def terminal_inventories(self) -> np.ndarray:
        return np.array([p.X[-1] for p in self.paths])


class MonteCarloEngine:
    """
    Runs N_paths independent OU paths and strategy simulations.

    Parameters
    ----------
    coeffs   : CoefficientsResult (pre-computed once for all paths)
    ou_sim   : OUSimulator
    n_paths  : number of Monte Carlo paths
    seed     : master seed for reproducibility
    """

    def __init__(
        self,
        coeffs:  CoefficientsResult,
        ou_sim:  Optional[OUSimulator] = None,
        n_paths: int = 1000,
        seed:    Optional[int] = None,
    ) -> None:
        self.coeffs   = coeffs
        self.n_paths  = n_paths
        rng = np.random.default_rng(seed)
        self.ou_sim   = ou_sim or OUSimulator(rng=rng)
        self.strategy_sim = StrategySimulator(coeffs)

    def run(self) -> MCResult:
        """Execute all paths and return MCResult."""
        result = MCResult()
        params = self.coeffs.params
        for _ in range(self.n_paths):
            Z    = self.ou_sim.simulate(params)
            path = self.strategy_sim.run(Z)
            result.paths.append(path)
        return result
