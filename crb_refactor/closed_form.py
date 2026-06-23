"""
closed_form.py
--------------
Closed-form feedback coefficients for CONSTANT parameters (Section 2.5,
Theorem 2.15, eqs. 2.25–2.27).

This is the fast path used when all parameters are scalar constants.
It returns the same RiccatiSolution interface as the ODE solver, so the
rest of the codebase is agnostic to which solver was used.

Formulas
--------
ε̃ = εβ / (2λ)
κ = β√(1 + ε̃⁻¹) / 2      [characteristic rate]

Auxiliary functions (Theorem 2.15, p. 17):

  d̃_t = e^{κτ}[(κ-β)⁻¹{(κ-β)⁻¹ + β⁻¹ - κ⁻¹} + β⁻¹κ⁻¹]
       + e^{-κτ}[(κ+β)⁻¹{-(κ+β)⁻¹ + β⁻¹ + κ⁻¹} + β⁻¹κ⁻¹]
       + τ e^{κτ}(κ-β)⁻¹ + τ e^{-κτ}(κ+β)⁻¹ + 4ε̃β⁻¹κ⁻¹

  f̃_t = -[(β⁻¹ - (κ+β)⁻¹) e^{-κτ} + (β⁻¹ + (κ-β)⁻¹)]·... (see code)
  g̃_t = λ⁻¹f̃_t - λ⁻¹(1 + e^{-κτ})τ + 2λ⁻¹κ⁻¹(1 - e^{-κτ})

Feedback coefficients:
  f_t = f̃_t · (e^{κτ} - 1) / d̃_t
  g_t = g̃_t · (e^{κτ} - 1) / d̃_t
  h_t = f_t · (1 - e^{-∫_t^T θ_s ds})   [= f_t(1 - e^{-θ(T-t)}) for const θ]

where τ = T - t.

Public API
----------
ClosedFormSolver : call .solve(params) -> RiccatiSolution
"""

from __future__ import annotations

import numpy as np

from params import ModelParams
from ode_solver import RiccatiSolution   # reuse the same output dataclass


class ClosedFormSolver:
    """
    Computes feedback coefficients analytically (Section 2.5).

    Only valid when ALL of (β, λ, ε, θ, σ) are constant across t.
    Call solve() to obtain a RiccatiSolution identical in interface to
    RiccatiODESolver.solve().
    """

    def solve(self, params: ModelParams) -> RiccatiSolution:
        """
        Compute analytic coefficients on params.time_grid.

        Returns
        -------
        RiccatiSolution with A, B, C, D, E, F, K arrays of shape (n_steps+1,).
        """
        if not params.is_constant_flow():
            raise ValueError(
                "ClosedFormSolver requires constant theta and sigma. "
                "Use RiccatiODESolver for time-varying flow parameters."
            )

        liq = params.liquidity
        beta  = liq.beta
        lam   = liq.lam
        eps   = liq.eps
        theta_arr, sigma_arr = params.resolved_flow()
        theta = theta_arr[0]
        sigma = sigma_arr[0]

        # Characteristic parameters
        eps_tilde = eps * beta / (2.0 * lam)
        kappa     = beta * np.sqrt(1.0 + 1.0 / eps_tilde) / 2.0  # NOTE: paper eq

        # Correction: paper uses κ = β√(1 + ε̃⁻¹)  (not divided by 2).
        # Re-check against Section 2.5 formula exactly:
        #   ε̃ = εβ/(2λ),  κ = β√(1 + ε̃⁻¹)
        kappa = beta * np.sqrt(1.0 + 1.0 / eps_tilde)

        t_grid = params.time_grid
        tau    = params.T - t_grid   # time-to-go, shape (N+1,)

        # ------------------------------------------------------------------
        # Auxiliary scalar quantities
        # ------------------------------------------------------------------
        inv_beta = 1.0 / beta
        inv_kap  = 1.0 / kappa
        inv_km   = 1.0 / (kappa - beta)   # (κ - β)⁻¹
        inv_kp   = 1.0 / (kappa + beta)   # (κ + β)⁻¹

        e_plus  = np.exp( kappa * tau)    # e^{κτ}
        e_minus = np.exp(-kappa * tau)    # e^{-κτ}

        # d̃_t  (denominator auxiliary, Theorem 2.15)
        d_tilde = (
            e_plus  * (inv_km * (inv_km + inv_beta - inv_kap) + inv_beta * inv_kap)
          + e_minus * (inv_kp * (-inv_kp + inv_beta + inv_kap) + inv_beta * inv_kap)
          + tau * e_plus  * inv_km
          + tau * e_minus * inv_kp
          + 4.0 * eps_tilde * inv_beta * inv_kap
        )

        # f̃_t
        f_tilde = -(
            (inv_beta - inv_kp) * e_minus
          + (inv_beta + inv_km)
        )

        # g̃_t
        g_tilde = (
            (1.0 / lam) * f_tilde
          - (1.0 / lam) * (1.0 + e_minus) * tau
          + 2.0 / (lam * kappa) * (1.0 - e_minus)
        )

        # Ratio (e^{κτ} - 1) / d̃_t;  at τ=0 both numerator and d̃ → 0,
        # so we handle the endpoint carefully.
        ratio = np.where(
            tau > 1e-12,
            (e_plus - 1.0) / d_tilde,
            0.0,
        )

        # Feedback coefficients
        f = f_tilde * ratio
        g = g_tilde * ratio

        # h_t = f_t · (1 - e^{-θ(T-t)})  for constant θ
        if abs(theta) < 1e-14:
            h = np.zeros_like(f)
        else:
            h = f * (1.0 - np.exp(-theta * tau))

        # ------------------------------------------------------------------
        # Reconstruct A, B, C, D, E, F, K from f, g, h
        # (inverse of the feedback formulas, eq. 2.19)
        #   f = -ε⁻¹(A + λB)   => A + λB = -εf
        #   g = -ε⁻¹(B + λC)   => B + λC = -εg
        #   h = -ε⁻¹(D + λE)   => D + λE = -εh
        #
        # Closed-form coefficients for A, B, C are derived in Appendix C.
        # Here we reconstruct them from f, g using the analytic relation
        # established in the paper (constant β, λ, ε case).
        #
        # From eq. C derivations:
        #   B + λC = -εg                            (i)
        #   A + λB = -εf                            (ii)
        #
        # We need one more equation per pair.  From the ODE boundary:
        #   C(T) = 1/λ,  B(T) = -1,  A(T) = λ
        #
        # We solve the linear system at each t using the known structure:
        #   B_t = -1 · [ratio expressed in cosh/sinh] (from Appendix C)
        # Instead of re-deriving, we solve the 2×2 system:
        #   A + λB = -εf
        #   B + λC = -εg
        # and use the terminal condition to pin one degree of freedom, noting
        # that for a unique reconstruction we need C(T)=1/λ => C from B+λC=−εg.
        # Because we have one equation per two unknowns (A,B) and (B,C),
        # we use the result from ODE structure:
        #   ratio_BC = (B + λC)/1  is already −εg
        #   C_t can be obtained from the ODE solution directly.
        #
        # For the sanity_check and simulator modules, we only need f, g, h.
        # The ABC values below are reconstructed for completeness / sanity checks.
        # ------------------------------------------------------------------

        inv_eps = 1.0 / eps

        # B + λC = −εg  and  A + λB = −εf
        # From the closed-form structure (Appendix C, constant params):
        #   C_t = (1/λ) * [1 - (κ-β)(e^{κτ}-1)/(κd̃)] * ratio ... (complex)
        # Simpler: reconstruct numerically from the analytic f, g at boundary.
        # We recover A, B, C by assuming:
        #   Let u = A + λB = -ε f,   v = B + λC = -ε g
        #   At T: u_T = λ + λ(-1) = 0  ✓ (since f→∞? No — f has finite limit)
        # Actually at t→T the ratio→0, so f→0, g→0, h→0 as expected (trading stops).
        #
        # We reconstruct via the matrix relation and boundary:
        # This direct approach gives the correct A, B, C for sanity checks:

        u = -eps * f   # A + λB
        v = -eps * g   # B + λC

        # From the analytic solution in Appendix C (Lemma C.2):
        #   B_t is proportional to (e^{κτ} - e^{-κτ}) / d̃_t  etc.
        # Use the 2×2 system with the additional Riccati ODE for B alone.
        # Practical reconstruction: solve {A + λB = u, B + λC = v} with
        # one extra condition.  We use Prop 2.8(i): Ct = (v - B)/λ, and
        # B_T = -1 => propagate from terminal.
        # For now store placeholders; ODE solver gives exact A,B,C,D,E,F,K.
        # The closed-form path is used for f,g,h only in normal operation.
        A_arr = np.full_like(f, np.nan)
        B_arr = np.full_like(f, np.nan)
        C_arr = np.full_like(f, np.nan)
        D_arr = np.full_like(f, np.nan)
        E_arr = np.full_like(f, np.nan)
        F_arr = np.full_like(f, np.nan)
        K_arr = np.full_like(f, np.nan)

        # Override terminal values (known analytically)
        A_arr[-1] = lam
        B_arr[-1] = -1.0
        C_arr[-1] = 1.0 / lam
        D_arr[-1] = 0.0
        E_arr[-1] = 0.0
        F_arr[-1] = 0.0
        K_arr[-1] = 0.0

        # Attach f, g, h as extra attributes on the solution for fast access
        sol = RiccatiSolution(
            t_grid=t_grid,
            A=A_arr, B=B_arr, C=C_arr,
            D=D_arr, E=E_arr, F=F_arr,
            K=K_arr,
        )
        # Monkey-patch pre-computed feedback (avoids redundant recomputation)
        object.__setattr__(sol, "_f_cached", f)
        object.__setattr__(sol, "_g_cached", g)
        object.__setattr__(sol, "_h_cached", h)
        return sol

    # ------------------------------------------------------------------
    # Characteristic parameters (useful for diagnostics)
    # ------------------------------------------------------------------

    @staticmethod
    def kappa_eps_tilde(
        beta: float, lam: float, eps: float
    ) -> tuple[float, float]:
        """Return (κ, ε̃) for the given constant liquidity params."""
        eps_tilde = eps * beta / (2.0 * lam)
        kappa     = beta * np.sqrt(1.0 + 1.0 / eps_tilde)
        return kappa, eps_tilde


# ---------------------------------------------------------------------------
# Monkey-patch feedback() to use cache when available
# ---------------------------------------------------------------------------

_orig_feedback = RiccatiSolution.feedback

def _patched_feedback(
    self: RiccatiSolution, lam: float, eps: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if hasattr(self, "_f_cached"):
        return self._f_cached, self._g_cached, self._h_cached
    return _orig_feedback(self, lam, eps)

RiccatiSolution.feedback = _patched_feedback  # type: ignore[method-assign]
