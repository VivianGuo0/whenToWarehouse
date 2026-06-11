"""
Analytic solution for the stochastic inventory management model.
Nutz, Webster, Zhao (2025) — Theorem 2.15 (constant parameters).

Optimal strategy:  q*_t = f_t * X_t + g_t * Y_t + h_t * Z_t
Opening block:     J0 = r^{-1} [ (g0 + η0)*y + (-f0 + h0)*z ]
Closing block:     JT = Z_T - J0 - ∫q dt

State dynamics (SDE discretisation):
  X_{t+dt} = X_t + q_t * dt - dZ_t        (inventory)
  Y_{t+dt} = Y_t + (-β Y_t + λ q_t) dt    (impact state)
  Z_{t+dt} = Z_t - θ Z_t * dt + σ * dW_t  (in-flow, OU process)

All formulas reference Theorem 2.15 / equations (2.25)–(2.27) in the paper.
"""

import numpy as np
import pandas as pd


# ============================================================
# 1. Analytic coefficients  (Theorem 2.15)
# ============================================================

def compute_analytic_coefficients(
    beta: float,
    lam: float,
    eps: float,
    theta: float,
    T: float = 1.0,
    N: int = 200,
) -> dict:
    """
    Pre-compute the feedback coefficients f_t, g_t, h_t on [0, T] in closed form.

    Parameters  (all scalars, constant across time)
    ----------
    beta  : impact decay rate  β
    lam   : impact loading     λ
    eps   : spread cost        ε
    theta : in-flow mean-reversion speed  θ  (can be 0)
    T, N  : horizon and number of time steps

    Returns
    -------
    dict with keys:
        'ts'  – time grid, shape (N+1,)
        'f'   – f_t,  shape (N+1,)
        'g'   – g_t,  shape (N+1,)
        'h'   – h_t,  shape (N+1,)
        'J0_f', 'J0_g' – scalar coefficients s.t. J0 = J0_f*f0 + J0_g*g0 (internal)
        'r'   – scalar normalisation r
        'J0_coeff_y', 'J0_coeff_z' – J0 = J0_coeff_y*y + J0_coeff_z*z
        'kappa', 'eps_tilde'
    """
    eps_tilde = eps * beta / (2 * lam)           # ε̃ = εβ/(2λ)
    kappa     = beta * np.sqrt(1 + 1/eps_tilde)  # κ = β√(1 + ε̃⁻¹)

    ts   = np.linspace(0, T, N + 1)
    taus = T - ts                                 # time-to-go τ = T - t

    # ---- f̃_t, g̃_t, d_t  (paper eq. 2.25–2.27, appendix C.28) ----

    e_neg = np.exp(-kappa * taus)   # e^{-κτ}
    e_pos = np.exp( kappa * taus)   # e^{+κτ}

    f_tilde = (
        - (1/beta - 1/(kappa + beta)) * e_neg
        - (1/beta + 1/(kappa - beta))
    )
    g_tilde = (
        f_tilde / lam
        - (1 + e_neg) * taus / lam
        + 2 / (lam * kappa) * (1 - e_neg)
    )

    # determinant d_t  (paper eq. 2.25 denominator)
    d = (
        e_pos * ((kappa - beta)**-1 * ((kappa - beta)**-1 + beta**-1 - kappa**-1) + beta**-1 * kappa**-1)
        + e_neg * ((kappa + beta)**-1 * (-(kappa + beta)**-1 + beta**-1 + kappa**-1) + beta**-1 * kappa**-1)
        + taus * e_pos / (kappa - beta)
        + taus * e_neg / (kappa + beta)
        + 4 * eps_tilde / (beta * kappa)
    )

    scale = (e_pos - 1) / d   # (e^{κτ} - 1) / d_t

    f = f_tilde * scale        # eq. (2.25)
    g = g_tilde * scale        # eq. (2.26)

    # h_t = f_t * (1 - e^{-∫_t^T θ du}) = f_t * (1 - e^{-θτ})   eq. (2.27)
    h = f * (1 - np.exp(-theta * taus))

    # ---- opening block J0 coefficients ----
    # J0 = r^{-1} [ (g0 + η0)*y  +  (-f0 + h0)*z ]
    # η0 = 0 when λ is constant  (no jump in λ at 0)
    f0, g0, h0 = f[0], g[0], h[0]
    r = -f0 - lam * g0          # r = -f_{0-} - λ(g_{0-} + η_{0-})

    J0_coeff_y =  g0 / r        # coefficient on y in J0
    J0_coeff_z = (-f0 + h0) / r # coefficient on z in J0

    return {
        "ts": ts,
        "f": f,
        "g": g,
        "h": h,
        "r": r,
        "J0_coeff_y": J0_coeff_y,
        "J0_coeff_z": J0_coeff_z,
        "kappa": kappa,
        "eps_tilde": eps_tilde,
    }


# ============================================================
# 2. Monte Carlo path simulation
# ============================================================

def simulate_paths(
    coeffs: dict,
    y: float = 0.0,
    z: float = 0.1,
    lam: float = 0.2,
    beta: float = 8.0,
    eps: float = 1e-2,
    theta: float = 0.0,
    sigma: float = 0.1,
    T: float = 1.0,
    N: int = 200,
    n_samples: int = 1000,
    n_shocks: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simulate MC paths under the optimal analytic strategy.

    Parameters
    ----------
    coeffs    : output of compute_analytic_coefficients
    y, z      : initial impact state and in-flow
    lam, beta, theta, sigma : model parameters
    T, N      : horizon and grid size
    n_samples : number of MC paths
    n_shocks  : shocks per horizon (n_shocks = N gives Brownian driver)

    Returns
    -------
    summary : pd.DataFrame, shape (n_samples,)  — per-path scalar stats
    ts_df   : pd.DataFrame — long-format time series (X, Y, Z, q)
    """
    ts   = coeffs["ts"]
    dt   = ts[1] - ts[0]
    f, g, h = coeffs["f"], coeffs["g"], coeffs["h"]
    J0_y = coeffs["J0_coeff_y"]
    J0_z = coeffs["J0_coeff_z"]

    # ---- opening block ----
    J0 = J0_y * y + J0_z * z

    # ---- in-flow shocks: rescaled BM increments ----
    dW_full = np.sqrt(dt) * np.random.randn(n_samples, N)
    W       = np.hstack([np.zeros((n_samples, 1)), np.cumsum(dW_full, axis=1)])

    wait = N // n_shocks
    dW   = np.zeros((n_samples, N))
    for i in range(N):
        if i % wait == 0:
            dW[:, i] = (W[:, i+1] - W[:, i]) * np.sqrt(wait)
        # else dW[:,i] = 0  (no shock at this step)

    # ---- state arrays  (n_samples × N+1) ----
    X = np.zeros((n_samples, N + 1))
    Y = np.zeros((n_samples, N + 1))
    Z = np.zeros((n_samples, N + 1))
    q = np.zeros((n_samples, N + 1))

    X[:, 0] = J0 - z
    Y[:, 0] = y + lam * J0
    Z[:, 0] = z

    for i in range(N):
        dZ          = -theta * Z[:, i] * dt + sigma * dW[:, i]
        X[:, i+1]  = X[:, i] + q[:, i] * dt - dZ
        Y[:, i+1]  = Y[:, i] + (-beta * Y[:, i] + lam * q[:, i]) * dt
        Z[:, i+1]  = Z[:, i] + dZ
        q[:, i+1]  = f[i+1] * X[:, i+1] + g[i+1] * Y[:, i+1] + h[i+1] * Z[:, i+1]

    # ---- terminal adjustments ----
    q[:, -1]  = 0
    X[:, 0]   = -z
    X[:, -1]  = 0
    JT         = Z[:, -1] - J0 - q.sum(axis=1) * dt
    Y[:, -1]  = Y[:, -2] + lam * JT

    trade_sum = J0 + q.cumsum(axis=1) * dt
    trade_sum[:, -1] += JT
    trade_sum[:, 0]   = 0

    # ---- per-path statistics ----
    spread_cost  = 0.5 * eps * (q**2).sum(axis=1) * dt
    impact_cost  = (
        0.5 * (Y[:, 0] + Y[:, 1]) * J0
        + (Y * q).sum(axis=1) * dt
        + 0.5 * (Y[:, -2] + Y[:, -1]) * JT
    )
    tv_out   = np.abs(q).sum(axis=1) * dt + np.abs(J0) + np.abs(JT)
    tv_in    = np.abs(z) + np.abs(np.diff(Z, axis=1)).sum(axis=1)
    qv_in    = z**2 + (np.diff(Z, axis=1)**2).sum(axis=1)

    summary = pd.DataFrame({
        "spread_cost":  spread_cost,
        "impact_cost":  impact_cost,
        "total_cost":   spread_cost + impact_cost,
        "tv_out":       tv_out,
        "tv_in":        tv_in,
        "qv_in":        qv_in,
        "internalization":    (1 - tv_out / tv_in) * 100,
        "closing_pct":        np.abs(JT) / tv_out * 100,
        "spread_cost_per_in": spread_cost / tv_in * 1e4,
        "impact_cost_per_in": impact_cost / tv_in * 1e4,
        "total_cost_per_in":  (spread_cost + impact_cost) / tv_in * 1e4,
    })

    # ---- tidy time series ----
    n = N + 1
    hours = ts / n * 6.5 + 9.5   # map [0,1] → NYSE hours [9.5, 16.0]
    sample_idx = np.repeat(np.arange(n_samples), n)
    time_idx   = np.tile(np.arange(n), n_samples)

    ts_df = pd.DataFrame({
        "sample_id":         sample_idx,
        "time":              hours[time_idx],
        "in-flow (ADV%)":    Z.flatten() * 100,
        "inventory (ADV%)":  X.flatten() * 100,
        "impact state (bps)":Y.flatten() * 1e4,
        "out-flow (ADV%)":   trade_sum.flatten() * 100,
    })

    return summary, ts_df


# ============================================================
# 3. Convenience wrapper
# ============================================================

def run(
    beta: float  = 8.0,
    lam: float   = 0.2,
    eps: float   = 1e-2,
    theta: float = 0.0,
    sigma: float = 0.1,
    y: float     = 0.0,
    z: float     = 0.1,
    T: float     = 1.0,
    N: int       = 200,
    n_samples: int  = 1000,
    n_shocks: int   = 20,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    One-call interface: compute coefficients and simulate paths.

    Returns
    -------
    summary, ts_df, coeffs
    """
    coeffs  = compute_analytic_coefficients(beta, lam, eps, theta, T, N)
    summary, ts_df = simulate_paths(
        coeffs, y=y, z=z, lam=lam, beta=beta, eps=eps,
        theta=theta, sigma=sigma, T=T, N=N,
        n_samples=n_samples, n_shocks=n_shocks,
    )
    return summary, ts_df, coeffs
