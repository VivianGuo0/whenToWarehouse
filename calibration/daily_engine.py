"""
Daily execution engine for the CRB unwind model.
Nutz, Webster, Zhao (2025)

Separates three distinct concerns:

  1. DailyStrategy   — pre-market: compute optimal coefficients (analytic, no MC)
  2. IntradayExecutor — intraday: apply feedback rule to real observed states
  3. DailySimulator  — pre-market: Monte Carlo cost distribution for risk/quoting

Multi-day usage:
  engine = MultiDayEngine(beta, lam, eps, theta, sigma)
  for day, (z_today, params_today) in enumerate(calendar):
      engine.start_day(**params_today, z=z_today)
      for t, Z_obs in intraday_feed:
          q = engine.step(t, Z_obs)
          # send q to market
      engine.end_day()
  history = engine.history
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from model import compute_analytic_coefficients, simulate_paths


# ─────────────────────────────────────────────
# Data containers
# ─────────────────────────────────────────────

@dataclass
class DayParams:
    """Calibrated parameters for one trading day."""
    beta:  float = 8.0    # impact decay rate
    lam:   float = 0.2    # Kyle's lambda
    eps:   float = 1e-2   # spread cost
    theta: float = 0.0    # in-flow autocorrelation
    sigma: float = 0.1    # in-flow volatility (only used for MC)
    z:     float = 0.1    # initial in-flow / inventory magnitude
    y:     float = 0.0    # initial impact state (carried from previous day)
    T:     float = 1.0    # horizon (1 trading day)
    N:     int   = 200    # time grid steps


@dataclass
class DayRecord:
    """Everything recorded for one completed trading day."""
    day:         int
    params:      DayParams
    J0:          float          # opening block trade
    JT:          float          # closing block trade
    q_path:      np.ndarray     # intraday trading speed, shape (N+1,)
    X_path:      np.ndarray     # inventory path
    Y_path:      np.ndarray     # impact state path
    Z_path:      np.ndarray     # observed in-flow path
    spread_cost: float
    impact_cost: float
    tv_in:       float
    tv_out:      float
    internalization: float      # %
    Y_terminal:  float          # Y_T before overnight decay → feeds tomorrow's y


# ─────────────────────────────────────────────
# 1. Pre-market: compute optimal strategy
# ─────────────────────────────────────────────

class DailyStrategy:
    """
    Given today's calibrated parameters, pre-compute the full-day
    feedback coefficients.  Pure analytics — runs in milliseconds.
    """

    def __init__(self, params: DayParams):
        self.params = params
        self.coeffs = compute_analytic_coefficients(
            beta=params.beta,
            lam=params.lam,
            eps=params.eps,
            theta=params.theta,
            T=params.T,
            N=params.N,
        )
        self.dt = self.coeffs["ts"][1] - self.coeffs["ts"][0]

        # Opening block trade (deterministic given y, z)
        self.J0 = (
            self.coeffs["J0_coeff_y"] * params.y
            + self.coeffs["J0_coeff_z"] * params.z
        )

    def optimal_q(self, step: int, X: float, Y: float, Z: float) -> float:
        """
        Return the optimal trading speed at time step `step`.

        Parameters
        ----------
        step : int        index into [0, N]
        X    : float      current inventory
        Y    : float      current impact state
        Z    : float      current cumulative in-flow
        """
        f = self.coeffs["f"][step]
        g = self.coeffs["g"][step]
        h = self.coeffs["h"][step]
        return f * X + g * Y + h * Z

    def summary(self) -> dict:
        return {
            "kappa":      self.coeffs["kappa"],
            "eps_tilde":  self.coeffs["eps_tilde"],
            "J0":         self.J0,
            "f0":         self.coeffs["f"][0],
            "g0":         self.coeffs["g"][0],
            "h0":         self.coeffs["h"][0],
        }


# ─────────────────────────────────────────────
# 2. Intraday: apply strategy to real data
# ─────────────────────────────────────────────

class IntradayExecutor:
    """
    Executes the optimal strategy against a live (or back-tested) in-flow.

    Usage
    -----
    executor = IntradayExecutor(strategy)
    executor.open()                          # places J0
    for step, dZ in enumerate(inflow_feed):
        q = executor.step(step, dZ)          # returns q to send to market
    record = executor.close()               # places JT, returns DayRecord
    """

    def __init__(self, strategy: DailyStrategy, day_index: int = 0):
        self.strat = strategy
        self.day   = day_index
        p = strategy.params
        self.N  = p.N
        self.dt = strategy.dt
        self.beta, self.lam, self.eps = p.beta, p.lam, p.eps

        # State arrays (allocated once)
        self.X = np.zeros(p.N + 1)
        self.Y = np.zeros(p.N + 1)
        self.Z = np.zeros(p.N + 1)
        self.q = np.zeros(p.N + 1)

    def open(self) -> float:
        """Set initial states after opening auction. Returns J0."""
        p   = self.strat.params
        J0  = self.strat.J0
        self.X[0] = -p.z          # inventory starts at -z (short the in-flow)
        self.Y[0] = p.y + self.lam * J0
        self.Z[0] = p.z
        self.q[0] = self.strat.optimal_q(0, self.X[0], self.Y[0], self.Z[0])
        self._J0  = J0
        return J0

    def step(self, i: int, dZ_observed: float) -> float:
        """
        Advance one time step.

        Parameters
        ----------
        i            : current step index (0-indexed, range [0, N-1])
        dZ_observed  : observed in-flow increment this step

        Returns
        -------
        q : float   optimal trading speed to execute now
        """
        assert 0 <= i < self.N, f"step index {i} out of range [0, {self.N-1}]"

        self.X[i+1] = self.X[i] + self.q[i] * self.dt - dZ_observed
        self.Y[i+1] = self.Y[i] + (-self.beta * self.Y[i] + self.lam * self.q[i]) * self.dt
        self.Z[i+1] = self.Z[i] + dZ_observed

        self.q[i+1] = self.strat.optimal_q(i+1, self.X[i+1], self.Y[i+1], self.Z[i+1])
        return float(self.q[i+1])

    def close(self) -> DayRecord:
        """
        Execute closing block trade to enforce X_T = 0.
        Returns a complete DayRecord.
        """
        J0 = self._J0
        JT = self.Z[-1] - J0 - self.q.sum() * self.dt
        self.X[-1] = 0.0
        self.Y[-1] = self.Y[-2] + self.lam * JT
        self.q[-1] = 0.0

        # Cost accounting
        spread_cost = 0.5 * self.eps * (self.q**2).sum() * self.dt
        impact_cost = (
            0.5 * (self.Y[0] + self.Y[1]) * J0
            + (self.Y * self.q).sum() * self.dt
            + 0.5 * (self.Y[-2] + self.Y[-1]) * JT
        )
        tv_out = np.abs(self.q).sum() * self.dt + abs(J0) + abs(JT)
        tv_in  = abs(self.Z[0]) + np.abs(np.diff(self.Z)).sum()

        return DayRecord(
            day         = self.day,
            params      = self.strat.params,
            J0          = J0,
            JT          = JT,
            q_path      = self.q.copy(),
            X_path      = self.X.copy(),
            Y_path      = self.Y.copy(),
            Z_path      = self.Z.copy(),
            spread_cost = float(spread_cost),
            impact_cost = float(impact_cost),
            tv_in       = float(tv_in),
            tv_out      = float(tv_out),
            internalization = float((1 - tv_out / tv_in) * 100) if tv_in > 0 else 0.0,
            Y_terminal  = float(self.Y[-1]),
        )


# ─────────────────────────────────────────────
# 3. Pre-market: Monte Carlo risk assessment
# ─────────────────────────────────────────────

class DailySimulator:
    """
    Run Monte Carlo before the market opens to estimate today's
    cost distribution.  Used for client quoting and risk management.
    """

    def __init__(self, strategy: DailyStrategy):
        self.strat = strategy

    def run(self, n_samples: int = 1000, n_shocks: int = 20) -> pd.DataFrame:
        """
        Returns summary statistics DataFrame (one row per MC path).
        Columns: spread_cost, impact_cost, total_cost_per_in,
                 internalization, closing_pct, ...
        """
        p = self.strat.params
        summary, _ = simulate_paths(
            coeffs    = self.strat.coeffs,
            y         = p.y,
            z         = p.z,
            lam       = p.lam,
            beta      = p.beta,
            eps       = p.eps,
            theta     = p.theta,
            sigma     = p.sigma,
            T         = p.T,
            N         = p.N,
            n_samples = n_samples,
            n_shocks  = n_shocks,
        )
        return summary

    def cost_quote(self, n_samples: int = 1000, percentiles=(10, 50, 90)) -> dict:
        """
        Return a simple cost quote dict for client-facing use.
        {
          'expected_cost_bps': ...,
          'p10_cost_bps': ...,
          'p90_cost_bps': ...,
          'expected_internalization_pct': ...,
        }
        """
        summary = self.run(n_samples)
        col = "total_cost_per_in"
        ps  = np.percentile(summary[col], percentiles)
        return {
            "expected_cost_bps":           round(summary[col].mean(), 2),
            f"p{percentiles[0]}_cost_bps": round(ps[0], 2),
            f"p{percentiles[1]}_cost_bps": round(ps[1], 2),
            f"p{percentiles[2]}_cost_bps": round(ps[2], 2),
            "expected_internalization_pct": round(summary["internalization"].mean(), 1),
        }


# ─────────────────────────────────────────────
# 4. Multi-day orchestrator
# ─────────────────────────────────────────────

T_OVERNIGHT = 16.0 / 6.5   # ~2.46 trading-day units (16 off-market hours)


class MultiDayEngine:
    """
    Runs the full daily loop across multiple days, carrying state
    (impact residual y) from one day to the next.

    Usage
    -----
    engine = MultiDayEngine()

    # --- back-test mode (simulated in-flow) ---
    for day, params in enumerate(daily_params):
        record = engine.run_simulated_day(params, n_shocks=20)

    # --- live mode ---
    engine.start_day(params)
    engine.open()
    for step, dZ in live_feed:
        q = engine.step(step, dZ)
    engine.end_day()

    df = engine.history_df()
    """

    def __init__(self):
        self.history:  list[DayRecord] = []
        self._y_carry: float = 0.0     # impact state carried overnight
        self._executor: IntradayExecutor | None = None

    # ── convenience: single-day back-test with synthetic in-flow ──

    def run_simulated_day(
        self,
        params_override: dict | None = None,
        n_shocks: int = 20,
        **kwargs,
    ) -> DayRecord:
        """
        Simulate one day with synthetic Gaussian in-flow shocks.
        `params_override` or keyword args update the default DayParams.
        The overnight impact carry is applied automatically.
        """
        p = self._build_params(params_override, kwargs)
        strategy = DailyStrategy(p)
        executor = IntradayExecutor(strategy, day_index=len(self.history))
        executor.open()

        # Synthetic in-flow: n_shocks equispaced Gaussian shocks
        dZ_sequence = self._synthetic_inflow(p, n_shocks)
        for i, dZ in enumerate(dZ_sequence):
            executor.step(i, dZ)

        record = executor.close()
        self._record_and_carry(record, p)
        return record

    # ── live-mode interface ──

    def start_day(self, params_override: dict | None = None, **kwargs) -> float:
        """Call before market open. Returns J0 (opening block size)."""
        p = self._build_params(params_override, kwargs)
        strategy = DailyStrategy(p)
        self._executor = IntradayExecutor(strategy, day_index=len(self.history))
        return self._executor.open()

    def step(self, i: int, dZ_observed: float) -> float:
        """Feed one observed in-flow increment; returns q to execute."""
        assert self._executor is not None, "call start_day() first"
        return self._executor.step(i, dZ_observed)

    def end_day(self) -> DayRecord:
        """Call at market close. Returns completed DayRecord."""
        assert self._executor is not None, "call start_day() first"
        record = self._executor.close()
        self._record_and_carry(record, self._executor.strat.params)
        self._executor = None
        return record

    # ── pre-market risk quote ──

    def cost_quote(
        self,
        params_override: dict | None = None,
        n_samples: int = 500,
        **kwargs,
    ) -> dict:
        """
        Run MC before open and return a cost quote dict.
        Call after start_day() or with explicit params.
        """
        if self._executor is not None:
            strategy = self._executor.strat
        else:
            p = self._build_params(params_override, kwargs)
            strategy = DailyStrategy(p)
        return DailySimulator(strategy).cost_quote(n_samples)

    # ── results ──

    def history_df(self) -> pd.DataFrame:
        """Return a summary DataFrame across all completed days."""
        if not self.history:
            return pd.DataFrame()
        rows = []
        for r in self.history:
            rows.append({
                "day":               r.day,
                "theta":             r.params.theta,
                "sigma":             r.params.sigma,
                "z":                 r.params.z,
                "y0":                r.params.y,
                "J0":                r.J0,
                "JT":                r.JT,
                "spread_cost":       r.spread_cost,
                "impact_cost":       r.impact_cost,
                "total_cost":        r.spread_cost + r.impact_cost,
                "tv_in":             r.tv_in,
                "tv_out":            r.tv_out,
                "internalization_%": r.internalization,
                "total_cost_per_in_bps": (r.spread_cost + r.impact_cost) / r.tv_in * 1e4
                                          if r.tv_in > 0 else np.nan,
                "Y_terminal":        r.Y_terminal,
                "y_carry_tomorrow":  r.Y_terminal * np.exp(-r.params.beta * T_OVERNIGHT),
            })
        return pd.DataFrame(rows)

    # ── internals ──

    def _build_params(self, override: dict | None, kwargs: dict) -> DayParams:
        """Merge defaults + override dict + kwargs, inject overnight y carry."""
        base = {
            "beta": 8.0, "lam": 0.2, "eps": 1e-2,
            "theta": 0.0, "sigma": 0.1, "z": 0.1,
            "y": self._y_carry,   # ← overnight carry
            "T": 1.0, "N": 200,
        }
        if override:
            base.update(override)
        base.update(kwargs)
        return DayParams(**base)

    def _record_and_carry(self, record: DayRecord, params: DayParams):
        self.history.append(record)
        self._y_carry = record.Y_terminal * np.exp(-params.beta * T_OVERNIGHT)

    @staticmethod
    def _synthetic_inflow(p: DayParams, n_shocks: int) -> np.ndarray:
        """
        Generate N synthetic dZ increments with n_shocks Gaussian shocks.
        Mirrors the shock structure used in simulate_paths().
        """
        dt   = p.T / p.N
        wait = p.N // n_shocks
        dZ   = np.zeros(p.N)
        Z    = p.z
        for i in range(p.N):
            shock = 0.0
            if i % wait == 0:
                shock = p.sigma * np.random.randn() * np.sqrt(wait * dt)
            mean_reversion = -p.theta * Z * dt
            dZ[i] = mean_reversion + shock
            Z    += dZ[i]
        return dZ
