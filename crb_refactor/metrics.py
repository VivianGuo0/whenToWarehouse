"""
metrics.py
----------
Trading metrics defined in Section 3.1.2 (eqs. 3.1–3.2).

All metrics operate on SimulationResult or raw arrays.

Metrics
-------
internalization_rate   : 1 - TV(Q) / TV(Z)
internalization_regret : 1 - |Z_T| / TV(Q)
impact_cost_bps        : impact cost per unit in-flow (bps)
spread_cost_bps        : spread cost per unit in-flow (bps)
total_cost_bps         : impact + spread cost per unit in-flow (bps)

Public API
----------
TradingMetrics       : dataclass holding all metrics for one path
MetricsCalculator    : computes TradingMetrics from SimulationResult
MetricsSummary       : aggregate statistics over Monte Carlo paths
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from params import ModelParams
from simulator import SimulationResult, MCResult


# ---------------------------------------------------------------------------
# Per-path metrics
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TradingMetrics:
    """All core metrics for one simulation path."""
    tv_inflow:             float   # TV(Z)
    tv_outflow:            float   # TV(Q)
    internalization_rate:  float   # 1 - TV(Q)/TV(Z)
    internalization_regret:float   # 1 - |Z_T|/TV(Q)
    impact_cost_bps:       float
    spread_cost_bps:       float
    total_cost_bps:        float
    closing_trade_pct:     float   # closing trade as % of TV(Q)


class MetricsCalculator:
    """
    Computes TradingMetrics from a SimulationResult.

    Parameters
    ----------
    params : ModelParams (needed for eps, beta, lam)
    """

    def __init__(self, params: ModelParams) -> None:
        self.params = params

    def compute(self, result: SimulationResult) -> TradingMetrics:
        """Compute all metrics for one path."""
        liq = self.params.liquidity
        eps  = liq.eps
        lam  = liq.lam
        dt   = self.params.dt
        n    = self.params.n_steps

        Z = result.Z
        Q = result.Q
        q = result.q
        Y = result.Y

        # Total variations (discrete)
        dZ = np.diff(Z)
        dQ = np.diff(Q)
        tv_Z = float(np.sum(np.abs(dZ)))
        tv_Q = float(np.sum(np.abs(dQ)))

        if tv_Z < 1e-15:
            raise ValueError("TV(Z) ≈ 0; metrics undefined (no in-flow).")

        # Internalization rate  (eq. 3.1)
        intern_rate   = 1.0 - tv_Q / tv_Z

        # Internalization regret  (eq. 3.1)
        closing_trade = abs(float(Q[-1]) - float(Q[-2]))   # last dQ step
        if tv_Q > 1e-15:
            intern_regret = 1.0 - abs(float(Z[-1])) / tv_Q
            closing_pct   = closing_trade / tv_Q * 100.0
        else:
            intern_regret = np.nan
            closing_pct   = np.nan

        # Cost decomposition (eq. 3.2)
        # Impact cost: ∫ Y_t dQ_t  ≈  Σ Y_k q_k Δt
        impact_cost = float(np.sum(Y[:-1] * q * dt))
        # Spread cost: ε/2 · ∫ q² dt  ≈  ε/2 · Σ q_k² Δt
        spread_cost = float(0.5 * eps * np.sum(q**2) * dt)

        # Per unit in-flow, convert to bps (×10000 if quantities are in ADV%)
        # Convention: raw units throughout; caller rescales to bps externally
        impact_bps = impact_cost / tv_Z
        spread_bps = spread_cost / tv_Z
        total_bps  = impact_bps + spread_bps

        return TradingMetrics(
            tv_inflow=tv_Z,
            tv_outflow=tv_Q,
            internalization_rate=intern_rate,
            internalization_regret=intern_regret,
            impact_cost_bps=impact_bps,
            spread_cost_bps=spread_bps,
            total_cost_bps=total_bps,
            closing_trade_pct=closing_pct,
        )


# ---------------------------------------------------------------------------
# Aggregate statistics over Monte Carlo paths
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MetricsSummary:
    """Mean ± std of each metric over N_paths."""
    n_paths: int
    internalization_rate:   tuple[float, float]   # (mean, std)
    internalization_regret: tuple[float, float]
    impact_cost_bps:        tuple[float, float]
    spread_cost_bps:        tuple[float, float]
    total_cost_bps:         tuple[float, float]
    closing_trade_pct:      tuple[float, float]
    tv_inflow:              tuple[float, float]

    def __str__(self) -> str:
        lines = [
            f"MetricsSummary over {self.n_paths} paths",
            f"  Internalization rate   : {self.internalization_rate[0]*100:.1f}% "
            f"± {self.internalization_rate[1]*100:.1f}%",
            f"  Internalization regret : {self.internalization_regret[0]*100:.1f}% "
            f"± {self.internalization_regret[1]*100:.1f}%",
            f"  Impact cost            : {self.impact_cost_bps[0]:.4f} "
            f"± {self.impact_cost_bps[1]:.4f} (per unit in-flow)",
            f"  Spread cost            : {self.spread_cost_bps[0]:.4f} "
            f"± {self.spread_cost_bps[1]:.4f}",
            f"  Total cost             : {self.total_cost_bps[0]:.4f} "
            f"± {self.total_cost_bps[1]:.4f}",
            f"  Closing trade          : {self.closing_trade_pct[0]:.1f}% "
            f"± {self.closing_trade_pct[1]:.1f}% of out-flow",
        ]
        return "\n".join(lines)


def aggregate_metrics(
    mc_result: MCResult,
    params: ModelParams,
) -> MetricsSummary:
    """
    Compute MetricsSummary from a MonteCarloEngine result.

    Parameters
    ----------
    mc_result : MCResult from MonteCarloEngine.run()
    params    : ModelParams

    Returns
    -------
    MetricsSummary with per-metric (mean, std).
    """
    calc = MetricsCalculator(params)
    all_metrics = [calc.compute(path) for path in mc_result.paths]

    def _ms(vals: list[float]) -> tuple[float, float]:
        arr = np.array(vals)
        return (float(np.nanmean(arr)), float(np.nanstd(arr)))

    return MetricsSummary(
        n_paths=mc_result.n_paths,
        internalization_rate=_ms([m.internalization_rate for m in all_metrics]),
        internalization_regret=_ms([m.internalization_regret for m in all_metrics]),
        impact_cost_bps=_ms([m.impact_cost_bps for m in all_metrics]),
        spread_cost_bps=_ms([m.spread_cost_bps for m in all_metrics]),
        total_cost_bps=_ms([m.total_cost_bps for m in all_metrics]),
        closing_trade_pct=_ms([m.closing_trade_pct for m in all_metrics]),
        tv_inflow=_ms([m.tv_inflow for m in all_metrics]),
    )
