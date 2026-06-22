"""
vega_model.py
=============
Vega Flow 最优做市：Riccati ODE 求解 + 反馈策略
基于 Nutz, Webster, Zhao (2025) 扩展

修改点：
1. A_T = psi（软终端惩罚）
2. A_dot 增加 -2*phi_t 项（Gamma/Theta running cost）
3. B_T = 0, C_T = psi / (lambda_V_T * (psi * lambda_V_T + 1))
4. lambda_V 由 impact function 计算，每日 calibrate
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from dataclasses import dataclass, field
from typing import Callable, Optional
import warnings


@dataclass(frozen=True)
class VegaParams:
    """
    Vega 问题参数（每日 pre-market calibrate）。

    时变参数可传入 callable f(t) -> float，
    也可传入标量（自动转为常数函数）。
    """
    # Market / impact parameters
    lambda_V: float | Callable    # vol impact 系数 (vol pts / $vega)
    beta: float | Callable        # impact 衰减速度 (1/day)
    eps_V: float | Callable       # spread/execution cost

    # Flow parameters (OU)
    theta: float                  # mean-reversion speed
    sigma: float                  # flow volatility
    mu: float = 0.0               # flow mean level

    # Vega-specific
    phi: float | Callable = 0.0   # Gamma/Theta running cost
    psi: float = 0.0              # soft terminal penalty

    # Time horizon
    T: float = 1.0                # day fraction (1.0 = full day)

    # Numerical
    n_steps: int = 1000           # ODE grid points

    def _to_fn(self, p):
        if callable(p):
            return p
        return lambda t: float(p)

    def lambda_fn(self, t): return self._to_fn(self.lambda_V)(t)
    def beta_fn(self, t):   return self._to_fn(self.beta)(t)
    def eps_fn(self, t):    return self._to_fn(self.eps_V)(t)
    def phi_fn(self, t):    return self._to_fn(self.phi)(t)

    def dgamma_fn(self, t, dt=1e-6):
        """d/dt log(lambda_V(t))"""
        lam = self._to_fn(self.lambda_V)
        return (np.log(lam(t + dt)) - np.log(lam(t - dt))) / (2 * dt)


@dataclass
class VegaCoefficients:
    """
    Pre-computed Riccati ODE coefficients on [0, T].
    All arrays are time-indexed [i] ↔ t_grid[i].
    """
    t_grid: np.ndarray
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    D: np.ndarray
    E: np.ndarray
    F: np.ndarray
    K: np.ndarray

    # Feedback coefficients
    f: np.ndarray  # q* = f*x + g*y + h*z
    g: np.ndarray
    h: np.ndarray

    def interp_at(self, t: float):
        """Interpolate all coefficients at time t."""
        idx = np.searchsorted(self.t_grid, t)
        idx = np.clip(idx, 0, len(self.t_grid) - 1)
        return {k: getattr(self, k)[idx] for k in ['A','B','C','D','E','F','K','f','g','h']}


def solve_vega_riccati(params: VegaParams) -> VegaCoefficients:
    """
    Backward solve the Vega Riccati ODE system.

    ODE (in backward time s = T - t, integrated 0 → T):

    dA/ds = -(eps^-1 * (A + lam*B)^2 - 2*phi)   [AT = psi]
    dB/ds = -(eps^-1 * (A+lam*B)*(B+lam*C) + beta*B)   [BT = 0]
    dC/ds = -(eps^-1*(B+lam*C)^2 + 2*beta*C - (2*beta+dgamma)/lam)  [CT]
    dD/ds = -(eps^-1*(A+lam*B)*(D+lam*E) - theta*(A-D))  [DT=0]
    dE/ds = -(eps^-1*(B+lam*C)*(D+lam*E) - theta*(B-E) + beta*E)  [ET=0]
    dF/ds = -(eps^-1*(D+lam*E)^2 - 2*theta*(D-F))  [FT=0]
    dK/ds = -(- sigma^2/2*(A-2D+F))  [KT=0]
    """
    T = params.T
    lam_T = params.lambda_fn(T)
    psi = params.psi

    # Terminal conditions
    A_T = psi
    B_T = 0.0
    C_T = psi / (lam_T * (psi * lam_T + 1)) if psi > 0 else 0.0
    y0 = [A_T, B_T, C_T, 0.0, 0.0, 0.0, 0.0]  # [A,B,C,D,E,F,K]

    def rhs(s, state):
        t = T - s
        A, B, C, D, E, F, K = state

        lam = params.lambda_fn(t)
        eps = params.eps_fn(t)
        bt  = params.beta_fn(t)
        th  = params.theta
        sig = params.sigma
        ph  = params.phi_fn(t)
        dg  = params.dgamma_fn(t)

        AB = A + lam * B
        BC = B + lam * C
        DE = D + lam * E

        dA = -(AB**2 / eps - 2 * ph)
        dB = -(AB * BC / eps + bt * B)
        dC = -(BC**2 / eps + 2 * bt * C - (2 * bt + dg) / lam)
        dD = -(AB * DE / eps - th * (A - D))
        dE = -(BC * DE / eps - th * (B - E) + bt * E)
        dF = -(DE**2 / eps - 2 * th * (D - F))
        dK = -(-sig**2 / 2 * (A - 2*D + F))

        return [dA, dB, dC, dD, dE, dF, dK]

    t_eval = np.linspace(0, T, params.n_steps)
    sol = solve_ivp(
        rhs, [0, T], y0,
        t_eval=t_eval,
        method='RK45',
        rtol=1e-8, atol=1e-10,
        dense_output=False
    )

    if not sol.success:
        warnings.warn(f"ODE solve failed: {sol.message}")

    # Reverse: solution is in backward time s, convert to forward t
    A, B, C, D, E, F, K = [sol.y[i][::-1] for i in range(7)]
    t_grid = T - sol.t[::-1]

    # Feedback coefficients (at each t)
    lam_arr = np.array([params.lambda_fn(t) for t in t_grid])
    eps_arr = np.array([params.eps_fn(t) for t in t_grid])

    f = -(A + lam_arr * B) / eps_arr
    g = -(B + lam_arr * C) / eps_arr
    h = -(D + lam_arr * E) / eps_arr

    return VegaCoefficients(
        t_grid=t_grid,
        A=A, B=B, C=C, D=D, E=E, F=F, K=K,
        f=f, g=g, h=h
    )


def simulate_vega_flow(params: VegaParams, n_paths: int, seed: int = 42,
                        z0: float = 0.0, dt: float = None) -> np.ndarray:
    """
    Simulate Vega flow Z_t ~ OU(theta, sigma, mu) on [0, T].
    Returns shape (n_paths, n_steps).
    """
    rng = np.random.default_rng(seed)
    n = params.n_steps
    T = params.T
    if dt is None:
        dt = T / (n - 1)

    # Exact OU discretization
    e = np.exp(-params.theta * dt)
    v = np.sqrt(params.sigma**2 / (2 * params.theta) * (1 - e**2)) if params.theta > 0 else params.sigma * np.sqrt(dt)

    Z = np.zeros((n_paths, n))
    Z[:, 0] = z0
    for i in range(1, n):
        Z[:, i] = params.mu * (1 - e) + Z[:, i-1] * e + v * rng.standard_normal(n_paths)
    return Z


def run_optimal_strategy(Z_path: np.ndarray, coefs: VegaCoefficients,
                          params: VegaParams,
                          x0: float = 0.0, y0: float = 0.0) -> dict:
    """
    Forward simulate the optimal strategy given a vega flow path Z_path.

    q*_t = f_t * X_t + g_t * Y_t + h_t * Z_t

    Returns dict with arrays X, Y, q, cost_running, total_cost.
    """
    n = len(Z_path)
    t_grid = coefs.t_grid
    dt = np.diff(t_grid, prepend=0.0)
    dt[0] = t_grid[0]  # handle t=0

    X = np.zeros(n)
    Y = np.zeros(n)
    q = np.zeros(n)
    running = np.zeros(n)

    X[0] = x0
    Y[0] = y0

    for i in range(n - 1):
        t = t_grid[i]
        dti = dt[i+1]
        lam = params.lambda_fn(t)
        bt = params.beta_fn(t)
        eps = params.eps_fn(t)
        ph = params.phi_fn(t)
        dg = params.dgamma_fn(t)

        q[i] = coefs.f[i] * X[i] + coefs.g[i] * Y[i] + coefs.h[i] * Z_path[i]

        # Running cost
        running[i] = (
            (2*bt + dg) / lam * Y[i]**2
            + eps * q[i]**2
            + 2 * ph * X[i]**2
        ) * dti / 2

        # State update (Euler)
        dZ = Z_path[i+1] - Z_path[i]
        X[i+1] = X[i] + q[i] * dti - dZ
        Y[i+1] = Y[i] + (-bt * Y[i] + lam * q[i]) * dti

    # Terminal cost
    terminal_cost = params.psi * X[-1]**2

    total_cost = np.sum(running) + terminal_cost

    return dict(X=X, Y=Y, q=q, running_cost=running,
                terminal_cost=terminal_cost, total_cost=total_cost)
