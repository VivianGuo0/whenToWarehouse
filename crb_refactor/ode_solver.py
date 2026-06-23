"""
ode_solver.py
-------------
Numerical solution of the Riccati ODE system (Proposition 2.7, eq. 2.20).

The 7-component ODE (backwards in time):

  Ȧ  = ε⁻¹(A + λB)²                                    A(T) = λ_T
  Ḃ  = ε⁻¹(A + λB)(B + λC) + βB                        B(T) = -1
  Ċ  = ε⁻¹(B + λC)² + 2βC - λ⁻¹(2β + γ̇)               C(T) = λ_T⁻¹
  Ḋ  = ε⁻¹(A + λB)(D + λE) - θ(A - D)                  D(T) = 0
  Ė  = ε⁻¹(B + λC)(D + λE) - θ(B - E) + βE             E(T) = 0
  Ḟ  = ε⁻¹(D + λE)² - 2θ(D - F)                        F(T) = 0
  K̇  = -σ²/2 · (A - 2D + F)                             K(T) = 0

where dots denote d/dt (forward time), i.e. we solve the terminal-value
problem by integrating backwards from t=T to t=0.

For constant liquidity params (β, λ, ε) and constant flow params (θ, σ),
use closed_form.py instead — it is orders of magnitude faster.

All integrations are performed with scipy's RK45.

Public API
----------
RiccatiSolution  : frozen dataclass holding coefficient arrays on the time grid
RiccatiODESolver : solver class; call .solve(params) -> RiccatiSolution
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

from params import LiquidityParams, ModelParams


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RiccatiSolution:
    """
    ODE solution evaluated on the forward time grid [t_0, ..., t_N].

    All arrays have shape (n_steps + 1,).
    Coefficients are indexed so that index k corresponds to time t_k.
    """
    t_grid: np.ndarray   # shape (N+1,)
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    D: np.ndarray
    E: np.ndarray
    F: np.ndarray
    K: np.ndarray

    # ------------------------------------------------------------------
    # Derived feedback coefficients  f, g, h  (eq. 2.19)
    # ------------------------------------------------------------------

    def feedback(
        self, lam: float, eps: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute (f_t, g_t, h_t) arrays on t_grid.

        f_t = -ε⁻¹(A_t + λ B_t)
        g_t = -ε⁻¹(B_t + λ C_t)
        h_t = -ε⁻¹(D_t + λ E_t)
        """
        inv_eps = 1.0 / eps
        f = -inv_eps * (self.A + lam * self.B)
        g = -inv_eps * (self.B + lam * self.C)
        h = -inv_eps * (self.D + lam * self.E)
        return f, g, h

    def opening_block(
        self,
        lam: float,
        eps: float,
        y0_minus: float,
        eta0_minus: float = 0.0,
    ) -> float:
        """
        Compute the optimal opening block trade J_0 (Proposition 2.13, eq. 2.22).

        J_0 = [(g_0 + η_{0-})·y_{0-} + (-f_0 + h_0)·z_0] / r
        r   = -f_0 - λ(g_0 + η_{0-})

        Parameters
        ----------
        y0_minus   : impact state just before the open
        eta0_minus : pre-open impact parameter (0 unless overnight carry modelled)
        """
        f, g, h = self.feedback(lam, eps)
        f0, g0, h0 = f[0], g[0], h[0]
        z0 = 0.0  # caller should pass actual z0 if needed; see simulator
        r = -f0 - lam * (g0 + eta0_minus)
        if abs(r) < 1e-12:
            raise RuntimeError("Degenerate opening-block denominator r ≈ 0.")
        J0 = ((g0 + eta0_minus) * y0_minus) / r
        return J0


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

class RiccatiODESolver:
    """
    Solves Proposition 2.7 Riccati ODE system for (possibly) time-varying
    (β_t, λ_t, ε_t, θ_t, σ_t).

    Liquidity params (β, λ, ε) are fixed scalars (LiquidityParams);
    flow params (θ, σ) may be time-arrays via ModelParams.resolved_flow().

    Usage
    -----
    >>> solver = RiccatiODESolver()
    >>> sol = solver.solve(params)   # params: ModelParams
    >>> f, g, h = sol.feedback(params.liquidity.lam, params.liquidity.eps)
    """

    def __init__(self, rtol: float = 1e-8, atol: float = 1e-10) -> None:
        self.rtol = rtol
        self.atol = atol

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def solve(self, params: ModelParams) -> RiccatiSolution:
        """Solve the Riccati system and return a RiccatiSolution."""
        liq = params.liquidity
        t_grid = params.time_grid  # shape (N+1,), forward time
        theta_arr, sigma_arr = params.resolved_flow()  # shape (N,)

        # Build interpolators for θ and σ on forward time
        # (defined on interval midpoints or step-left values)
        t_steps = t_grid[:-1]  # t_0, ..., t_{N-1}
        theta_fn = self._make_interp(t_steps, theta_arr, params.T)
        sigma_fn = self._make_interp(t_steps, sigma_arr, params.T)

        # ODE is solved BACKWARDS: let τ = T - t, so τ runs 0 → T
        T = params.T

        # Terminal conditions at t=T (τ=0)
        lam_T = liq.lam   # constant λ
        y0_vec = np.array([
            lam_T,      # A(T)
            -1.0,       # B(T)
            1.0/lam_T,  # C(T)
            0.0,        # D(T)
            0.0,        # E(T)
            0.0,        # F(T)
            0.0,        # K(T)
        ])

        def rhs(tau: float, y: np.ndarray) -> np.ndarray:
            """RHS in backward time τ = T - t.  ẏ/dτ = -ẏ/dt."""
            t_fwd = T - tau
            A, B, C, D, E, F, K = y

            beta = liq.beta
            lam  = liq.lam
            eps  = liq.eps
            theta = theta_fn(t_fwd)
            sigma = sigma_fn(t_fwd)
            gamma_dot = 0.0   # constant lam => γ̇ = 0

            inv_eps = 1.0 / eps
            AB = A + lam * B
            BC = B + lam * C
            DE = D + lam * E

            # Forward-time ODEs (2.20); negate for backward integration
            dA_dt = inv_eps * AB**2
            dB_dt = inv_eps * AB * BC + beta * B
            dC_dt = inv_eps * BC**2 + 2*beta*C - (2*beta + gamma_dot)/lam
            dD_dt = inv_eps * AB * DE - theta * (A - D)
            dE_dt = inv_eps * BC * DE - theta * (B - E) + beta * E
            dF_dt = inv_eps * DE**2 - 2*theta * (D - F)
            dK_dt = -0.5 * sigma**2 * (A - 2*D + F)

            # In backward time: dy/dτ = -dy/dt
            return np.array([-dA_dt, -dB_dt, -dC_dt,
                             -dD_dt, -dE_dt, -dF_dt, -dK_dt])

        # Solve over τ ∈ [0, T]
        tau_eval = T - t_grid[::-1]   # 0 → T  (reversed from forward grid)
        sol = solve_ivp(
            rhs,
            t_span=(0.0, T),
            y0=y0_vec,
            method="RK45",
            t_eval=tau_eval,
            rtol=self.rtol,
            atol=self.atol,
            dense_output=False,
        )

        if not sol.success:
            raise RuntimeError(f"Riccati ODE solver failed: {sol.message}")

        # Reverse back to forward-time order
        A_fwd = sol.y[0, ::-1]
        B_fwd = sol.y[1, ::-1]
        C_fwd = sol.y[2, ::-1]
        D_fwd = sol.y[3, ::-1]
        E_fwd = sol.y[4, ::-1]
        F_fwd = sol.y[5, ::-1]
        K_fwd = sol.y[6, ::-1]

        return RiccatiSolution(
            t_grid=t_grid,
            A=A_fwd, B=B_fwd, C=C_fwd,
            D=D_fwd, E=E_fwd, F=F_fwd,
            K=K_fwd,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_interp(
        t_nodes: np.ndarray,
        values: np.ndarray,
        T: float,
    ) -> interp1d:
        """
        Build a piecewise-linear interpolator for a time-varying parameter,
        extending the last value to t=T (right endpoint).
        """
        t_ext = np.append(t_nodes, T)
        v_ext = np.append(values, values[-1])
        return interp1d(t_ext, v_ext, kind="linear", fill_value="extrapolate")
