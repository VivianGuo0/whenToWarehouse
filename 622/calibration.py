"""
calibration.py
==============
Calibrate Vega Flow OU 参数 + 有效 vol impact 系数

主要功能：
1. MLE calibration of OU(theta, sigma, mu) from historical vega flow
2. OLS calibration of effective lambda_V from (execution, vol_impact) data
3. Diagnostics: ACF comparison, QQ plot, KS test
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import ks_2samp, norm
import warnings


# ===========================================================
# 1. OU Parameter Calibration (MLE)
# ===========================================================

def ou_loglikelihood(params, Z: np.ndarray, dt: float) -> float:
    """
    Exact discrete OU log-likelihood.
    params = [theta, sigma, mu]
    """
    theta, sigma, mu = params
    if theta <= 0 or sigma <= 0:
        return -1e10

    e = np.exp(-theta * dt)
    v2 = sigma**2 / (2 * theta) * (1 - e**2)
    if v2 <= 0:
        return -1e10

    Z_t = Z[:-1]
    Z_next = Z[1:]
    mean_cond = mu * (1 - e) + Z_t * e
    residuals = Z_next - mean_cond

    n = len(residuals)
    ll = -n/2 * np.log(2 * np.pi * v2) - np.sum(residuals**2) / (2 * v2)
    return ll


def calibrate_ou(Z: np.ndarray, dt: float,
                  theta_init: float = 1.0,
                  sigma_init: float = None,
                  mu_init: float = 0.0) -> dict:
    """
    MLE calibration of OU(theta, sigma, mu) from observed path Z.

    Parameters
    ----------
    Z    : np.ndarray, shape (n,), observed vega flow time series
    dt   : float, time step in same units as theta (e.g., days)

    Returns
    -------
    dict with keys: theta, sigma, mu, half_life, ll, converged
    """
    if sigma_init is None:
        sigma_init = np.std(np.diff(Z)) / np.sqrt(dt)

    # Initial guess via OLS (fast, closed-form)
    Z_t = Z[:-1]
    Z_next = Z[1:]
    e_init, _, _, _ = np.linalg.lstsq(
        np.column_stack([np.ones_like(Z_t), Z_t]), Z_next, rcond=None
    )
    mu_guess = e_init[0] / (1 - e_init[1]) if abs(1 - e_init[1]) > 1e-9 else 0.0
    theta_guess = -np.log(max(e_init[1], 1e-9)) / dt

    x0 = [max(theta_guess, 0.01), sigma_init, mu_guess]

    def neg_ll(p):
        return -ou_loglikelihood(p, Z, dt)

    res = minimize(neg_ll, x0, method='L-BFGS-B',
                   bounds=[(1e-4, None), (1e-8, None), (None, None)],
                   options={'maxiter': 1000, 'ftol': 1e-12})

    theta, sigma, mu = res.x
    return {
        'theta': theta,
        'sigma': sigma,
        'mu': mu,
        'half_life': np.log(2) / theta,  # in same time units as dt
        'll': -res.fun,
        'converged': res.success,
        'message': res.message
    }


def calibrate_ou_panel(flow_df: pd.DataFrame, dt: float,
                        date_col: str = 'date',
                        flow_col: str = 'vega_flow') -> pd.DataFrame:
    """
    Calibrate OU parameters per day (panel calibration).
    Useful for detecting time-varying theta.

    Parameters
    ----------
    flow_df : DataFrame with columns [date_col, flow_col]
    dt      : intraday time step

    Returns
    -------
    DataFrame with calibrated params per day
    """
    results = []
    for date, group in flow_df.groupby(date_col):
        Z = group[flow_col].values
        if len(Z) < 5:
            continue
        try:
            cal = calibrate_ou(Z, dt)
            cal['date'] = date
            cal['n_obs'] = len(Z)
            results.append(cal)
        except Exception as e:
            warnings.warn(f"Calibration failed for {date}: {e}")
    return pd.DataFrame(results)


# ===========================================================
# 2. Effective Vol Impact Coefficient (lambda_V)
# ===========================================================

def calibrate_lambda_V(q_executed: np.ndarray,
                        vol_impact: np.ndarray,
                        Z_level: np.ndarray = None,
                        method: str = 'ols') -> dict:
    """
    Calibrate effective vol impact lambda_V from historical execution data.

    Model: vol_impact = lambda_V * q_executed + noise
    (Optionally: condition on Z_level to remove flow-driven vol changes)

    Parameters
    ----------
    q_executed : np.ndarray, vega executed ($ vega or normalized)
    vol_impact : np.ndarray, resulting vol impact (vol points)
    Z_level    : np.ndarray optional, vega flow level for conditioning
    method     : 'ols' or 'robust' (Huber)

    Returns
    -------
    dict with lambda_V, se, r_squared
    """
    X = q_executed.reshape(-1, 1)

    if Z_level is not None:
        # Control for flow level
        X = np.column_stack([q_executed, Z_level])

    # Add intercept
    X_with_const = np.column_stack([np.ones(len(X)), X])
    y = vol_impact

    if method == 'ols':
        beta, residuals, rank, sv = np.linalg.lstsq(X_with_const, y, rcond=None)
        y_pred = X_with_const @ beta
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        # Standard errors
        n, p = X_with_const.shape
        if n > p:
            s2 = ss_res / (n - p)
            XtX_inv = np.linalg.pinv(X_with_const.T @ X_with_const)
            se = np.sqrt(np.diag(s2 * XtX_inv))
        else:
            se = np.full_like(beta, np.nan)

        return {
            'lambda_V': beta[1],        # coefficient on q_executed
            'intercept': beta[0],
            'beta_Z': beta[2] if Z_level is not None else None,
            'se_lambda_V': se[1],
            'r_squared': r2,
            'n_obs': n
        }
    else:
        raise NotImplementedError(f"method='{method}' not yet implemented")


def calibrate_lambda_V_rolling(df: pd.DataFrame,
                                q_col: str = 'q_executed',
                                impact_col: str = 'vol_impact',
                                window: int = 20,
                                dt: float = None) -> pd.DataFrame:
    """
    Rolling calibration of lambda_V (window = number of trading days).
    """
    results = []
    for i in range(window, len(df) + 1):
        chunk = df.iloc[i-window:i]
        try:
            cal = calibrate_lambda_V(
                chunk[q_col].values,
                chunk[impact_col].values
            )
            cal['end_idx'] = i - 1
            results.append(cal)
        except Exception as e:
            warnings.warn(f"lambda_V calibration failed at idx {i}: {e}")
    return pd.DataFrame(results)


# ===========================================================
# 3. Experiment 1: Flow Replication Diagnostics
# ===========================================================

def flow_replication_diagnostics(Z_real: np.ndarray,
                                   Z_sim: np.ndarray,
                                   dt: float,
                                   max_lags: int = 20) -> dict:
    """
    Compare real vega flow vs simulated paths.

    Parameters
    ----------
    Z_real : np.ndarray (n,), one historical path
    Z_sim  : np.ndarray (n_paths, n), simulated paths
    dt     : time step

    Returns
    -------
    dict with ACF comparison, KS test, moments
    """
    # ACF of real flow
    n = len(Z_real)
    acf_real = np.array([
        np.corrcoef(Z_real[:-k], Z_real[k:])[0, 1] if k > 0 else 1.0
        for k in range(max_lags + 1)
    ])

    # ACF of simulated (average over paths)
    acf_sim_all = []
    for path in Z_sim:
        acf_path = np.array([
            np.corrcoef(path[:-k], path[k:])[0, 1] if k > 0 else 1.0
            for k in range(max_lags + 1)
        ])
        acf_sim_all.append(acf_path)
    acf_sim_mean = np.mean(acf_sim_all, axis=0)
    acf_sim_std  = np.std(acf_sim_all, axis=0)

    # Moments
    Z_sim_flat = Z_sim.flatten()
    moments = {
        'mean_real': np.mean(Z_real),
        'mean_sim':  np.mean(Z_sim_flat),
        'std_real':  np.std(Z_real),
        'std_sim':   np.std(Z_sim_flat),
        'skew_real': _skewness(Z_real),
        'skew_sim':  _skewness(Z_sim_flat),
    }

    # KS test
    ks_stat, ks_pval = ks_2samp(Z_real, Z_sim_flat)

    # Momentum metric |Z_T| / TV(Z) for each path
    Z_T_real = Z_real[-1]
    tv_real = np.sum(np.abs(np.diff(Z_real)))
    mom_real = abs(Z_T_real) / tv_real if tv_real > 0 else 0.0

    mom_sim = []
    for path in Z_sim:
        Z_T = path[-1]
        tv = np.sum(np.abs(np.diff(path)))
        mom_sim.append(abs(Z_T) / tv if tv > 0 else 0.0)

    return {
        'acf_lags': np.arange(max_lags + 1),
        'acf_real': acf_real,
        'acf_sim_mean': acf_sim_mean,
        'acf_sim_std': acf_sim_std,
        'moments': moments,
        'ks_stat': ks_stat,
        'ks_pval': ks_pval,
        'momentum_real': mom_real,
        'momentum_sim_mean': np.mean(mom_sim),
        'momentum_sim_std': np.std(mom_sim),
    }


def _skewness(x):
    mu = np.mean(x)
    s  = np.std(x)
    return np.mean(((x - mu) / s)**3) if s > 0 else 0.0


# ===========================================================
# 4. Experiment 2: Cost Comparison
# ===========================================================

def compute_historical_strategy_cost(
    q_hist: np.ndarray,
    X_hist: np.ndarray,
    Y_hist: np.ndarray,
    t_grid: np.ndarray,
    params_fn: dict,
    psi: float
) -> dict:
    """
    Compute cost of historical trading strategy under Vega cost model.

    Parameters
    ----------
    q_hist   : np.ndarray (n,), historical vega trading speed
    X_hist   : np.ndarray (n,), historical vega inventory
    Y_hist   : np.ndarray (n,), historical vol impact state (estimated)
    t_grid   : np.ndarray (n,)
    params_fn: dict with callables for beta, lambda_V, eps_V, phi, dgamma
    psi      : float, soft terminal penalty

    Returns
    -------
    dict with running_cost, terminal_cost, total_cost
    """
    dt = np.diff(t_grid, prepend=t_grid[0])
    n = len(t_grid)
    running = np.zeros(n)

    for i in range(n):
        t = t_grid[i]
        lam = params_fn['lambda_V'](t)
        bt  = params_fn['beta'](t)
        eps = params_fn['eps_V'](t)
        ph  = params_fn['phi'](t)
        dg  = params_fn['dgamma'](t)

        running[i] = (
            (2*bt + dg) / lam * Y_hist[i]**2
            + eps * q_hist[i]**2
            + 2 * ph * X_hist[i]**2
        ) * dt[i] / 2

    terminal_cost = psi * X_hist[-1]**2
    return {
        'running_cost': running,
        'terminal_cost': terminal_cost,
        'total_cost': np.sum(running) + terminal_cost
    }


def cost_comparison_report(
    opt_costs: np.ndarray,
    hist_costs: np.ndarray,
    dates: list = None
) -> pd.DataFrame:
    """
    Statistical comparison: optimal strategy vs historical strategy.

    Returns a summary DataFrame and prints key stats.
    """
    diff = hist_costs - opt_costs
    ratio = opt_costs / hist_costs

    df = pd.DataFrame({
        'date': dates if dates else range(len(opt_costs)),
        'cost_opt': opt_costs,
        'cost_hist': hist_costs,
        'saving_bps': diff,
        'cost_ratio': ratio,
    })

    # Paired t-test
    from scipy.stats import ttest_rel
    t_stat, p_val = ttest_rel(opt_costs, hist_costs)

    summary = {
        'mean_saving_bps': np.mean(diff),
        'median_saving_bps': np.median(diff),
        'std_saving_bps': np.std(diff),
        'pct_days_opt_cheaper': np.mean(diff > 0) * 100,
        't_stat': t_stat,
        'p_value': p_val,
        'mean_cost_ratio': np.mean(ratio),
    }

    return df, summary
