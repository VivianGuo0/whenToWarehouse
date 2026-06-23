"""
coefficients.py
---------------
Unified entry point that selects ClosedFormSolver (fast) or RiccatiODESolver
(general) based on whether parameters are constant.

Public API
----------
CoefficientEngine : call .compute(params) -> CoefficientsResult
CoefficientsResult: holds RiccatiSolution + convenience accessors
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

import numpy as np

from params import ModelParams
from ode_solver import RiccatiSolution, RiccatiODESolver
from closed_form import ClosedFormSolver


class SolverMode(Enum):
    AUTO        = auto()   # detect from params
    CLOSED_FORM = auto()   # force analytic
    ODE         = auto()   # force numerical


@dataclass(frozen=True)
class CoefficientsResult:
    """
    Holds solved Riccati coefficients and exposes derived quantities.

    Attributes
    ----------
    solution    : RiccatiSolution from either solver
    f, g, h     : feedback coefficient arrays on time_grid, shape (N+1,)
    params      : reference to the ModelParams used
    solver_used : which solver path was taken
    """
    solution:    RiccatiSolution
    f:           np.ndarray
    g:           np.ndarray
    h:           np.ndarray
    params:      ModelParams
    solver_used: str

    @property
    def t_grid(self) -> np.ndarray:
        return self.solution.t_grid

    def opening_block_trade(
        self,
        y0_minus: float | None = None,
        eta0_minus: float = 0.0,
    ) -> float:
        """
        Optimal opening block trade J_0 (Proposition 2.13, eq. 2.22).

        J_0 = [(g_0 + η_{0-})·y_{0-} + (-f_0 + h_0)·z_0] / r
        r   = -f_0 - λ(g_0 + η_{0-})

        y0_minus defaults to params.init.y0 if not supplied.
        """
        init = self.params.init
        lam  = self.params.liquidity.lam
        y_m  = init.y0 if y0_minus is None else y0_minus
        z0   = init.z0

        f0 = self.f[0]
        g0 = self.g[0]
        h0 = self.h[0]

        r = -f0 - lam * (g0 + eta0_minus)
        if abs(r) < 1e-12:
            raise RuntimeError("Degenerate opening-block denominator r ≈ 0.")
        J0 = ((g0 + eta0_minus) * y_m + (-f0 + h0) * z0) / r
        return float(J0)

    def strategy_at(self, t_idx: int, X: float, Y: float, Z: float) -> float:
        """
        Optimal trading speed q* at time index t_idx.

        q*(t) = f_t X_t + g_t Y_t + h_t Z_t
        """
        return float(self.f[t_idx] * X + self.g[t_idx] * Y + self.h[t_idx] * Z)


class CoefficientEngine:
    """
    Computes feedback coefficients (f, g, h) and the opening block J_0.

    Parameters
    ----------
    mode    : SolverMode.AUTO (default) selects closed-form when params
              are constant, ODE otherwise.
    ode_rtol, ode_atol : tolerances forwarded to RiccatiODESolver.
    verbose : if True, prints which solver was chosen.
    """

    def __init__(
        self,
        mode: SolverMode = SolverMode.AUTO,
        ode_rtol: float = 1e-8,
        ode_atol: float = 1e-10,
        verbose: bool = False,
    ) -> None:
        self.mode     = mode
        self.verbose  = verbose
        self._cf_solver  = ClosedFormSolver()
        self._ode_solver = RiccatiODESolver(rtol=ode_rtol, atol=ode_atol)

    def compute(self, params: ModelParams) -> CoefficientsResult:
        """
        Solve for (f, g, h) on params.time_grid.

        Returns CoefficientsResult with all coefficients and metadata.
        """
        use_cf = self._select_solver(params)

        if use_cf:
            sol = self._cf_solver.solve(params)
            solver_name = "ClosedForm"
        else:
            sol = self._ode_solver.solve(params)
            solver_name = "RiccatiODE"

        if self.verbose:
            print(f"[CoefficientEngine] Using {solver_name} solver.")

        f, g, h = sol.feedback(params.liquidity.lam, params.liquidity.eps)

        return CoefficientsResult(
            solution=sol,
            f=f, g=g, h=h,
            params=params,
            solver_used=solver_name,
        )

    def _select_solver(self, params: ModelParams) -> bool:
        """Return True to use closed-form, False for ODE."""
        if self.mode == SolverMode.CLOSED_FORM:
            return True
        if self.mode == SolverMode.ODE:
            return False
        # AUTO: use closed-form only when flow is constant
        return params.is_constant_flow()
