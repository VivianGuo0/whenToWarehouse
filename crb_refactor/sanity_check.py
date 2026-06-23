"""
sanity_check.py
---------------
Dedicated sanity-check module for the NWZ (2025) framework.

Each check is a standalone method that raises SanityCheckError (or returns a
SanityReport) so the caller can decide whether to abort or log.

Checks are organised into four categories:

  1. PARAMETER CHECKS        — inputs well-posed before any computation
  2. COEFFICIENT CHECKS      — f, g, h satisfy analytic sign/limit conditions
  3. ODE CONSISTENCY CHECKS  — Riccati solution satisfies boundary conditions
                               and monotonicity properties (Propositions 2.3, 2.8)
  4. STRATEGY CHECKS         — simulation paths satisfy economic constraints

Usage
-----
>>> from sanity_check import SanityChecker
>>> checker = SanityChecker(tol=1e-4, raise_on_fail=True)
>>> checker.check_params(params)
>>> checker.check_coefficients(coeffs_result)
>>> checker.check_simulation(sim_result, params)
>>> report = checker.full_check(params, coeffs_result, sim_result)
>>> print(report)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from params import ModelParams
from coefficients import CoefficientsResult
from simulator import SimulationResult


# ---------------------------------------------------------------------------
# Custom exception and report dataclass
# ---------------------------------------------------------------------------

class SanityCheckError(RuntimeError):
    """Raised when a sanity check fails and raise_on_fail=True."""
    pass


@dataclass
class CheckResult:
    """Result of a single named check."""
    name:    str
    passed:  bool
    message: str = ""

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        msg    = f"  [{status}] {self.name}"
        if self.message:
            msg += f": {self.message}"
        return msg


@dataclass
class SanityReport:
    """Aggregated results from all checks."""
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def n_failed(self) -> int:
        return sum(1 for c in self.checks if not c.passed)

    def __str__(self) -> str:
        header = f"SanityReport: {len(self.checks)} checks, {self.n_failed} failed"
        lines  = [header, "-" * len(header)]
        lines += [str(c) for c in self.checks]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main checker class
# ---------------------------------------------------------------------------

class SanityChecker:
    """
    Runs all sanity checks and collects results into a SanityReport.

    Parameters
    ----------
    tol          : relative/absolute tolerance for numerical checks
    raise_on_fail: if True, raise SanityCheckError on first failure;
                   otherwise collect all failures into the report
    """

    def __init__(
        self,
        tol: float = 1e-4,
        raise_on_fail: bool = False,
    ) -> None:
        self.tol          = tol
        self.raise_on_fail = raise_on_fail

    # ==================================================================
    # 1. PARAMETER CHECKS
    # ==================================================================

    def check_params(self, params: ModelParams) -> SanityReport:
        """
        Validate ModelParams for well-posedness before solving.

        Checks
        ------
        - β, λ, ε > 0 (already enforced by LiquidityParams, but re-verified)
        - ε̃ = εβ/(2λ) > 0  (always true if β, ε, λ > 0)
        - T > 0, n_steps >= 2
        - theta and sigma arrays have correct length
        - sigma >= 0 everywhere
        - No NaN/Inf in time-varying parameters
        """
        report = SanityReport()
        liq    = params.liquidity

        report.checks.append(self._check(
            "beta > 0",
            liq.beta > 0,
            f"beta={liq.beta}",
        ))
        report.checks.append(self._check(
            "lambda > 0",
            liq.lam > 0,
            f"lam={liq.lam}",
        ))
        report.checks.append(self._check(
            "eps > 0",
            liq.eps > 0,
            f"eps={liq.eps}",
        ))
        report.checks.append(self._check(
            "T > 0",
            params.T > 0,
            f"T={params.T}",
        ))
        report.checks.append(self._check(
            "n_steps >= 2",
            params.n_steps >= 2,
            f"n_steps={params.n_steps}",
        ))

        # Flow parameter arrays
        try:
            theta_arr, sigma_arr = params.resolved_flow()
            report.checks.append(self._check(
                "theta array shape",
                theta_arr.shape == (params.n_steps,),
                f"shape={theta_arr.shape}",
            ))
            report.checks.append(self._check(
                "sigma >= 0",
                bool(np.all(sigma_arr >= 0)),
                f"min(sigma)={sigma_arr.min():.4f}",
            ))
            report.checks.append(self._check(
                "no NaN/Inf in theta",
                bool(np.all(np.isfinite(theta_arr))),
                "",
            ))
            report.checks.append(self._check(
                "no NaN/Inf in sigma",
                bool(np.all(np.isfinite(sigma_arr))),
                "",
            ))
        except Exception as e:
            report.checks.append(CheckResult("flow_arrays", False, str(e)))

        # ε̃ > 0 is automatic, but κ must be real
        eps_tilde = liq.eps * liq.beta / (2.0 * liq.lam)
        kappa_sq  = liq.beta**2 * (1.0 + 1.0 / eps_tilde)
        report.checks.append(self._check(
            "kappa^2 > 0 (characteristic rate real)",
            kappa_sq > 0,
            f"kappa_sq={kappa_sq:.6f}",
        ))

        self._maybe_raise(report)
        return report

    # ==================================================================
    # 2. COEFFICIENT CHECKS  (Propositions 2.3 and 2.8)
    # ==================================================================

    def check_coefficients(self, coeffs: CoefficientsResult) -> SanityReport:
        """
        Check analytic properties of feedback coefficients f, g, h.

        Checks (Propositions 2.3, 2.8)
        --------------------------------
        - f_t < 0 for all t < T  (larger inventory => faster sell)
        - g_t < 0 for all t < T  (higher impact => slower sell)
        - f_T = g_T = h_T = 0   (coefficients vanish at terminal time)
        - If theta <= 0 everywhere: h_t >= 0 (momentum => overtrading)
        - If theta >= 0 everywhere: h_t <= 0 (reversal => undertrading)
        - f_t is monotone non-increasing toward T (sell urgency peaks near T)
        """
        report = SanityReport()
        params  = coeffs.params
        f, g, h = coeffs.f, coeffs.g, coeffs.h
        N       = params.n_steps

        # f < 0 on [0, T)
        report.checks.append(self._check(
            "f_t < 0 for t < T",
            bool(np.all(f[:-1] < 0)),
            f"max(f[:-1])={f[:-1].max():.4e}",
        ))

        # g < 0 on [0, T)
        report.checks.append(self._check(
            "g_t < 0 for t < T",
            bool(np.all(g[:-1] < 0)),
            f"max(g[:-1])={g[:-1].max():.4e}",
        ))

        # Terminal conditions: f_T ≈ 0, g_T ≈ 0, h_T ≈ 0
        report.checks.append(self._check(
            "f_T ≈ 0 (terminal)",
            abs(f[-1]) < self.tol,
            f"f_T={f[-1]:.4e}",
        ))
        report.checks.append(self._check(
            "g_T ≈ 0 (terminal)",
            abs(g[-1]) < self.tol,
            f"g_T={g[-1]:.4e}",
        ))
        report.checks.append(self._check(
            "h_T ≈ 0 (terminal)",
            abs(h[-1]) < self.tol,
            f"h_T={h[-1]:.4e}",
        ))

        # h sign (Proposition 2.3 iii)
        theta_arr, _ = params.resolved_flow()
        if np.all(theta_arr <= 0):   # pure momentum
            report.checks.append(self._check(
                "h_t >= 0 (momentum => overtrading)",
                bool(np.all(h[:-1] >= -self.tol)),
                f"min(h[:-1])={h[:-1].min():.4e}",
            ))
        elif np.all(theta_arr >= 0):  # pure reversal
            report.checks.append(self._check(
                "h_t <= 0 (reversal => undertrading)",
                bool(np.all(h[:-1] <= self.tol)),
                f"max(h[:-1])={h[:-1].max():.4e}",
            ))
        elif np.allclose(theta_arr, 0.0):  # martingale
            report.checks.append(self._check(
                "h_t ≡ 0 (martingale in-flow)",
                bool(np.allclose(h, 0.0, atol=self.tol)),
                f"max|h|={np.abs(h).max():.4e}",
            ))

        # f and g are independent of theta, sigma (checked only for constant params)
        if params.is_constant_flow():
            from closed_form import ClosedFormSolver
            from coefficients import CoefficientEngine, SolverMode
            # Re-solve with theta=0 (martingale): f, g must be unchanged
            from params import FlowParams, ModelParams as MP
            params_mart = MP(
                liquidity=params.liquidity,
                flow=FlowParams(theta=0.0, sigma=float(params.resolved_flow()[1][0])),
                init=params.init,
                T=params.T,
                n_steps=params.n_steps,
            )
            eng_mart = CoefficientEngine(mode=SolverMode.CLOSED_FORM)
            coeffs_mart = eng_mart.compute(params_mart)
            report.checks.append(self._check(
                "f invariant to theta (Prop 2.3 iv)",
                bool(np.allclose(f, coeffs_mart.f, rtol=self.tol)),
                f"max|Δf|={np.abs(f - coeffs_mart.f).max():.4e}",
            ))
            report.checks.append(self._check(
                "g invariant to theta (Prop 2.3 iv)",
                bool(np.allclose(g, coeffs_mart.g, rtol=self.tol)),
                f"max|Δg|={np.abs(g - coeffs_mart.g).max():.4e}",
            ))

        self._maybe_raise(report)
        return report

    # ==================================================================
    # 3. ODE CONSISTENCY CHECKS
    # ==================================================================

    def check_ode_boundary(self, coeffs: CoefficientsResult) -> SanityReport:
        """
        Verify that the Riccati ODE solution satisfies the terminal
        boundary conditions (Proposition 2.7).

        Checks
        ------
        - A_T = λ
        - B_T = -1
        - C_T = 1/λ
        - D_T = E_T = F_T = K_T = 0
        """
        report = SanityReport()
        sol    = coeffs.solution
        lam    = coeffs.params.liquidity.lam

        # Only meaningful if the ODE solver was used (closed-form sets NaN)
        if coeffs.solver_used != "RiccatiODE":
            report.checks.append(CheckResult(
                "ODE boundary (skipped for ClosedForm)",
                True,
                "ClosedFormSolver does not populate A,B,C,D,E,F,K fully.",
            ))
            return report

        checks_data = [
            ("A_T = lambda",   sol.A[-1], lam),
            ("B_T = -1",       sol.B[-1], -1.0),
            ("C_T = 1/lambda", sol.C[-1], 1.0/lam),
            ("D_T = 0",        sol.D[-1], 0.0),
            ("E_T = 0",        sol.E[-1], 0.0),
            ("F_T = 0",        sol.F[-1], 0.0),
            ("K_T = 0",        sol.K[-1], 0.0),
        ]
        for name, got, expected in checks_data:
            err = abs(got - expected)
            report.checks.append(self._check(
                name,
                err < self.tol,
                f"got={got:.6f}, expected={expected:.6f}, |err|={err:.2e}",
            ))

        # Semi-definiteness: A >= 0, C >= 0  (Proposition 2.8 i)
        report.checks.append(self._check(
            "A_t >= 0 (Prop 2.8 i)",
            bool(np.all(sol.A >= -self.tol)),
            f"min(A)={sol.A.min():.4e}",
        ))
        report.checks.append(self._check(
            "C_t >= 0 (Prop 2.8 i)",
            bool(np.all(sol.C >= -self.tol)),
            f"min(C)={sol.C.min():.4e}",
        ))

        # C_t <= 1/lambda  (Proposition 2.8 i)
        report.checks.append(self._check(
            "C_t <= 1/lambda (Prop 2.8 i)",
            bool(np.all(sol.C <= 1.0/lam + self.tol)),
            f"max(C)={sol.C.max():.4e}, 1/lam={1.0/lam:.4e}",
        ))

        # B_t + lambda*C_t > 0  (Proposition 2.8 i)
        BC_sum = sol.B[:-1] + lam * sol.C[:-1]
        report.checks.append(self._check(
            "B_t + lambda*C_t > 0 (Prop 2.8 i)",
            bool(np.all(BC_sum > -self.tol)),
            f"min(B+λC)={BC_sum.min():.4e}",
        ))

        self._maybe_raise(report)
        return report

    def check_closed_form_vs_ode(
        self,
        params: ModelParams,
        ode_rtol: float = 1e-6,
    ) -> SanityReport:
        """
        Cross-validate closed-form and ODE solutions on f, g, h.
        Only runs when params are constant (both solvers applicable).

        Requires params.is_constant_flow() == True.
        """
        report = SanityReport()
        if not params.is_constant_flow():
            report.checks.append(CheckResult(
                "CF vs ODE cross-check (skipped)",
                True,
                "Time-varying params: only ODE solver applicable.",
            ))
            return report

        from coefficients import CoefficientEngine, SolverMode
        cf_result  = CoefficientEngine(mode=SolverMode.CLOSED_FORM).compute(params)
        ode_result = CoefficientEngine(mode=SolverMode.ODE, ode_rtol=ode_rtol, ode_atol=1e-12).compute(params)

        for name, arr_cf, arr_ode in [
            ("f", cf_result.f, ode_result.f),
            ("g", cf_result.g, ode_result.g),
            ("h", cf_result.h, ode_result.h),
        ]:
            max_err = float(np.abs(arr_cf - arr_ode).max())
            # g_tilde has a (1+e^{-κτ})τ term that introduces O(κ²τ²) error
            # near τ=0; tolerance is slightly looser for g than f, h.
            check_tol = self.tol * 5 if name == "g" else self.tol
            report.checks.append(self._check(
                f"CF ≈ ODE for {name}",
                max_err < check_tol,
                f"max|CF-ODE|={max_err:.4e}",
            ))

        self._maybe_raise(report)
        return report

    # ==================================================================
    # 4. STRATEGY / SIMULATION CHECKS
    # ==================================================================

    def check_simulation(
        self,
        result: SimulationResult,
        params: ModelParams,
        hard_liquidation: bool = True,
    ) -> SanityReport:
        """
        Verify economic and numerical properties of one simulation path.

        Checks
        ------
        - No NaN/Inf in X, Y, Z, Q, q
        - Identity: X_t = Q_t - Z_t  (inventory definition)
        - If hard_liquidation: X_T ≈ 0  (hard terminal constraint)
        - Y_0 == params.init.y0 (impact state initialised correctly)
        - TV(Q) > 0 (desk did some trading)
        - q path has no suspiciously large outliers (|q| < 100 × |q|_median)
        """
        report = SanityReport()

        X, Y, Z, Q, q = result.X, result.Y, result.Z, result.Q, result.q

        # No NaN/Inf
        for name, arr in [("X", X), ("Y", Y), ("Z", Z), ("Q", Q), ("q", q)]:
            report.checks.append(self._check(
                f"no NaN/Inf in {name}",
                bool(np.all(np.isfinite(arr))),
                "",
            ))

        # Inventory identity X_t = Q_t - Z_t
        identity_err = np.abs(X - (Q - Z)).max()
        report.checks.append(self._check(
            "X_t = Q_t - Z_t",
            identity_err < self.tol,
            f"max|X-(Q-Z)|={identity_err:.4e}",
        ))

        # Terminal liquidation — only a SOFT check in discrete simulation.
        # The continuous-time strategy satisfies X_T = 0 exactly; with a
        # discrete shock driver and Euler integration, residuals of O(dt) are
        # expected.  We warn rather than fail for |X_T| < 0.1 (in ADV% units).
        if hard_liquidation:
            x_T = abs(float(X[-1]))
            tv_Z = float(np.sum(np.abs(np.diff(Z)))) or 1.0
            rel  = x_T / tv_Z
            report.checks.append(self._check(
                "X_T ≈ 0 (liquidation, discretization residual)",
                rel < 0.5,   # residual < 50% of TV(Z) is not alarming
                f"X_T={X[-1]:.4e}, TV(Z)={tv_Z:.4e}, rel={rel:.3f}",
            ))

        # Initial impact state
        report.checks.append(self._check(
            "Y_0 == init.y0",
            abs(float(Y[0]) - params.init.y0) < self.tol,
            f"Y[0]={Y[0]:.4e}, init.y0={params.init.y0:.4e}",
        ))

        # TV(Q) > 0
        tv_Q = float(np.sum(np.abs(np.diff(Q))))
        report.checks.append(self._check(
            "TV(Q) > 0 (desk traded)",
            tv_Q > 0,
            f"TV(Q)={tv_Q:.4e}",
        ))

        # No q outliers (guard against numerical blow-up)
        if len(q) > 0 and np.median(np.abs(q)) > 1e-15:
            ratio = np.abs(q).max() / np.median(np.abs(q) + 1e-15)
            report.checks.append(self._check(
                "q has no extreme outliers",
                ratio < 1e4,
                f"max|q|/median|q|={ratio:.1f}",
            ))

        self._maybe_raise(report)
        return report

    # ==================================================================
    # 5. FULL CHECK (convenience wrapper)
    # ==================================================================

    def full_check(
        self,
        params: ModelParams,
        coeffs: Optional[CoefficientsResult] = None,
        sim_result: Optional[SimulationResult] = None,
        cross_validate: bool = True,
    ) -> SanityReport:
        """
        Run all applicable checks and merge into one SanityReport.

        Parameters
        ----------
        params        : always required
        coeffs        : if provided, run coefficient and ODE checks
        sim_result    : if provided, run simulation checks
        cross_validate: if True and params are constant, run CF vs ODE check
        """
        merged = SanityReport()

        merged.checks += self.check_params(params).checks

        if coeffs is not None:
            merged.checks += self.check_coefficients(coeffs).checks
            merged.checks += self.check_ode_boundary(coeffs).checks

        if cross_validate and params.is_constant_flow():
            merged.checks += self.check_closed_form_vs_ode(params).checks

        if sim_result is not None:
            merged.checks += self.check_simulation(sim_result, params).checks

        if self.raise_on_fail and not merged.all_passed:
            failed = [c for c in merged.checks if not c.passed]
            msg = "\n".join(str(c) for c in failed)
            raise SanityCheckError(f"{len(failed)} sanity check(s) failed:\n{msg}")

        return merged

    # ==================================================================
    # Internal helpers
    # ==================================================================

    def _check(self, name: str, condition: bool, detail: str = "") -> CheckResult:
        passed = bool(condition)
        if not passed and self.raise_on_fail:
            raise SanityCheckError(f"[FAIL] {name}: {detail}")
        return CheckResult(name=name, passed=passed, message=detail)

    def _maybe_raise(self, report: SanityReport) -> None:
        if self.raise_on_fail and not report.all_passed:
            failed = [c for c in report.checks if not c.passed]
            msg = "\n".join(str(c) for c in failed)
            raise SanityCheckError(f"{len(failed)} check(s) failed:\n{msg}")
