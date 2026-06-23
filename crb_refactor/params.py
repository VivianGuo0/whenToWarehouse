"""
params.py
---------
Parameter containers for the NWZ (2025) model.

Design rules:
  - LiquidityParams: beta, lam (lambda), eps (epsilon) are SCALAR only.
    These are the market-impact/spread parameters assumed constant intraday
    (or pre-market linearised to effective scalars).
  - FlowParams: theta, sigma are time-varying; accept scalar OR 1-D array of
    length n_steps.  A scalar is broadcast to a constant array at solve time.
  - InitialConditions: x0, y0, z0 are always scalars.
  - ModelParams: top-level container that owns the time grid and delegates to
    the sub-containers above.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Union

import numpy as np

# Type alias: a parameter that may be scalar or a 1-D time array
TimeArray = Union[float, np.ndarray]


# ---------------------------------------------------------------------------
# Sub-containers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LiquidityParams:
    """
    Constant (intraday) market-impact and spread parameters.

    beta  : impact decay rate [day^-1].  Half-life = ln2 / beta.
    lam   : Kyle's lambda (price impact coefficient) [bps / ADV%].
    eps   : bid-ask spread parameter [bps^2 · day / ADV%^2].

    All three must be positive scalars; lam and eps drive the LQ cost.
    """
    beta: float
    lam: float
    eps: float

    def __post_init__(self) -> None:
        for name, val in [("beta", self.beta), ("lam", self.lam), ("eps", self.eps)]:
            if not np.isscalar(val):
                raise TypeError(
                    f"LiquidityParams.{name} must be a scalar; "
                    f"got array of shape {np.asarray(val).shape}. "
                    "Use time-varying theta/sigma in FlowParams instead."
                )
            if val <= 0:
                raise ValueError(f"LiquidityParams.{name} must be positive; got {val}.")

    # Convenience: gamma_t = lam_t (constant lam => gamma_dot = 0)
    @property
    def gamma_dot(self) -> float:
        """d(lambda)/dt under constant-lambda assumption."""
        return 0.0


@dataclass(frozen=True)
class FlowParams:
    """
    In-flow (OU) parameters — may be time-varying.

    theta : mean-reversion speed of Z.  Scalar OR array[n_steps].
            theta < 0  => momentum,  theta = 0 => martingale,  theta > 0 => reversal.
    sigma : volatility of Z innovations.  Scalar OR array[n_steps].
    """
    theta: TimeArray
    sigma: TimeArray

    def resolve(self, n_steps: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (theta_arr, sigma_arr) each of shape (n_steps,)."""
        theta_arr = self._broadcast(self.theta, n_steps, "theta")
        sigma_arr = self._broadcast(self.sigma, n_steps, "sigma")
        return theta_arr, sigma_arr

    @staticmethod
    def _broadcast(val: TimeArray, n: int, name: str) -> np.ndarray:
        arr = np.asarray(val, dtype=float)
        if arr.ndim == 0:
            return np.full(n, arr.item())
        if arr.shape != (n,):
            raise ValueError(
                f"FlowParams.{name}: expected scalar or array of length {n}, "
                f"got shape {arr.shape}."
            )
        return arr


@dataclass(frozen=True)
class InitialConditions:
    """
    State at t = 0 (after the opening block trade has been absorbed).

    x0 : initial inventory  X_0 = Q_0 - Z_0
    y0 : initial impact state  Y_0  (often 0 for a fresh start)
    z0 : initial cumulative in-flow  Z_0
    """
    x0: float = 0.0
    y0: float = 0.0
    z0: float = -0.1

    def __post_init__(self) -> None:
        for name, val in [("x0", self.x0), ("y0", self.y0), ("z0", self.z0)]:
            if not np.isscalar(val):
                raise TypeError(f"InitialConditions.{name} must be a scalar.")


# ---------------------------------------------------------------------------
# Top-level container
# ---------------------------------------------------------------------------

@dataclass
class ModelParams:
    """
    Full parameter set for one trading day.

    Attributes
    ----------
    T          : trading horizon in days (default 1.0)
    n_steps    : number of intraday time steps (default 390 = 1-minute bars)
    liquidity  : LiquidityParams (scalar market-impact params)
    flow       : FlowParams (possibly time-varying theta, sigma)
    init       : InitialConditions
    """
    liquidity: LiquidityParams
    flow: FlowParams
    init: InitialConditions = field(default_factory=InitialConditions)
    T: float = 1.0
    n_steps: int = 390

    def __post_init__(self) -> None:
        if self.T <= 0:
            raise ValueError(f"T must be positive; got {self.T}.")
        if self.n_steps < 2:
            raise ValueError(f"n_steps must be >= 2; got {self.n_steps}.")

    # ------------------------------------------------------------------
    # Derived grid quantities
    # ------------------------------------------------------------------

    @property
    def dt(self) -> float:
        return self.T / self.n_steps

    @property
    def time_grid(self) -> np.ndarray:
        """Array of shape (n_steps + 1,): t_0=0, ..., t_N=T."""
        return np.linspace(0.0, self.T, self.n_steps + 1)

    def resolved_flow(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (theta, sigma) arrays of shape (n_steps,) over [t_0, t_{N-1}]."""
        return self.flow.resolve(self.n_steps)

    # ------------------------------------------------------------------
    # Convenience factory
    # ------------------------------------------------------------------

    @classmethod
    def constant(
        cls,
        beta: float = 8.0,
        lam: float = 0.2,
        eps: float = 0.01,
        theta: float = 0.0,
        sigma: float = 0.1,
        x0: float = 0.0,
        y0: float = 0.0,
        z0: float = -0.1,
        T: float = 1.0,
        n_steps: int = 390,
    ) -> "ModelParams":
        """Construct a fully constant-parameter ModelParams (paper's base case)."""
        return cls(
            liquidity=LiquidityParams(beta=beta, lam=lam, eps=eps),
            flow=FlowParams(theta=theta, sigma=sigma),
            init=InitialConditions(x0=x0, y0=y0, z0=z0),
            T=T,
            n_steps=n_steps,
        )

    def is_constant_flow(self) -> bool:
        """True iff theta and sigma are effectively constant."""
        theta_arr, sigma_arr = self.resolved_flow()
        return (
            np.allclose(theta_arr, theta_arr[0])
            and np.allclose(sigma_arr, sigma_arr[0])
        )
