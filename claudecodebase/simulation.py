"""
Core simulation engine for the stochastic inventory management model.

State dynamics:
    X_{t+dt} = X_t + q_t * dt - (Z_{t+dt} - Z_t)
    Y_{t+dt} = Y_t + (-beta * Y_t + lambda * q_t) * dt
    Z_{t+dt} = Z_t - theta * Z_t * dt + sigma * (W_{t+dt} - W_t)

where:
    X = inventory
    Y = impact state
    Z = in-flow
    q = trading rate (control)
    W = Brownian motion
"""

import numpy as np
import pandas as pd
import itertools


# ---------------------------------------------------------------------------
# Parameter construction
# ---------------------------------------------------------------------------

def generate_parameters(
    name="baseline",
    y=[0.0],
    z=[0.1],
    lamb=[0.2],
    beta=[8.0],
    eps=[1e-2],
    theta=[0.0],
    sigma=[0.1],
) -> pd.DataFrame:
    """
    Generate a DataFrame of all parameter combinations via Cartesian product.

    Parameters
    ----------
    name : str
        Label for this parameter set.
    y, z, lamb, beta, eps, theta, sigma : list
        Parameter grids to sweep over.

    Returns
    -------
    pd.DataFrame
        One row per parameter combination, columns named after the parameters.
    """
    param_lists = {
        "y": y,
        "z": z,
        "lamb": lamb,
        "beta": beta,
        "eps": eps,
        "theta": theta,
        "sigma": sigma,
    }
    combos = list(itertools.product(*param_lists.values()))
    df = pd.DataFrame(combos, columns=param_lists.keys())
    df["name"] = name
    return df


def add_auxiliary_variables(parameters: pd.DataFrame) -> pd.DataFrame:
    """Compute derived parameters teps and kappa."""
    parameters = parameters.copy()
    parameters["teps"] = parameters["eps"] * parameters["beta"] / (2 * parameters["lamb"])
    parameters["kappa"] = parameters["beta"] * np.sqrt(1 + 1 / parameters["teps"])
    return parameters


# ---------------------------------------------------------------------------
# Time grid construction
# ---------------------------------------------------------------------------

def add_time_grid(parameters: pd.DataFrame, T: float = 1.0, N: int = 200):
    """
    Tile the parameter DataFrame over a uniform time grid [0, T] with N steps.

    Returns
    -------
    df : pd.DataFrame
        Long-format DataFrame with one row per (parameter set, time step).
    T, N, dt : float, int, float
        Unchanged horizon, number of steps, and step size.
    """
    parameters = parameters.copy()
    parameters["param_id"] = parameters.index

    ts = np.linspace(0, T, N + 1)
    dt = ts[1] - ts[0]
    n_params = len(parameters)

    numerical_cols = parameters.select_dtypes(include=[np.number]).columns.tolist()
    string_cols = parameters.select_dtypes(include=[object]).columns.tolist()

    num_rows = n_params * len(ts)
    data = np.empty((num_rows, len(parameters.columns) + 1), dtype=object)
    data[:, : len(numerical_cols)] = np.repeat(parameters[numerical_cols].values, len(ts), axis=0)
    data[:, len(numerical_cols) : len(numerical_cols) + len(string_cols)] = np.repeat(
        parameters[string_cols].values, len(ts), axis=0
    )
    data[:, -1] = np.tile(ts, n_params)

    df = pd.DataFrame(data, columns=numerical_cols + string_cols + ["time"])
    df[numerical_cols + ["time"]] = df[numerical_cols + ["time"]].apply(
        pd.to_numeric, errors="coerce"
    )
    df["taus"] = T - df["time"]
    return df, T, N, dt


def add_helper_variables(df: pd.DataFrame, T: float):
    """
    Compute the optimal-control helper functions f, g, h evaluated on the time grid.

    These encode the value-function coefficients used by the optimal trading policy.
    """
    df = df.copy()

    exp_neg = np.exp(-df["kappa"] * df["taus"])
    exp_neg_T = np.exp(-df["kappa"] * T)

    df["tfs"] = (
        -(1 / df["beta"] - 1 / (df["kappa"] + df["beta"])) * exp_neg
        - (1 / df["beta"] + 1 / (df["kappa"] - df["beta"]))
    )
    df["tgs"] = (
        df["tfs"] / df["lamb"]
        - (1 + exp_neg) * df["taus"] / df["lamb"]
        + 2 / (df["lamb"] * df["kappa"]) * (1 - exp_neg)
    )
    df["tfs0"] = (
        -(1 / df["beta"] - 1 / (df["kappa"] + df["beta"])) * exp_neg_T
        - (1 / df["beta"] + 1 / (df["kappa"] - df["beta"]))
    )
    df["tgs0"] = (
        df["tfs0"] / df["lamb"]
        - (1 + exp_neg_T) * T / df["lamb"]
        + 2 / (df["lamb"] * df["kappa"]) * (1 - exp_neg_T)
    )
    df["r"] = -df["tfs0"] - df["lamb"] * df["tgs0"]
    df["J0"] = (
        df["tgs0"] * df["y"]
        - df["tfs0"] * np.exp(-df["theta"] * T) * df["z"] / df["r"]
    )
    df["ds"] = (
        np.exp(df["kappa"] * df["taus"])
        * (
            1 / (df["kappa"] - df["beta"])
            * (1 / (df["kappa"] - df["beta"]) + 1 / df["beta"] - 1 / df["kappa"])
            + 1 / (df["beta"] * df["kappa"])
        )
        + np.exp(-df["kappa"] * df["taus"])
        * (
            1 / (df["kappa"] + df["beta"])
            * (-1 / (df["kappa"] + df["beta"]) + 1 / df["beta"] + 1 / df["kappa"])
            + 1 / (df["beta"] * df["kappa"])
        )
        + df["taus"] * np.exp(df["kappa"] * df["taus"]) / (df["kappa"] - df["beta"])
        + df["taus"] * np.exp(-df["kappa"] * df["taus"]) / (df["kappa"] + df["beta"])
        + 4 * df["teps"] / (df["beta"] * df["kappa"])
    )
    df["fs"] = df["tfs"] * (np.exp(df["kappa"] * df["taus"]) - 1) / df["ds"]
    df["gs"] = df["tgs"] * (np.exp(df["kappa"] * df["taus"]) - 1) / df["ds"]
    df["hs"] = df["fs"] * (1 - np.exp(-df["theta"] * df["taus"]))
    return df, T


# ---------------------------------------------------------------------------
# Tensor reshape helpers
# ---------------------------------------------------------------------------

def _reshape_time_grid(df: pd.DataFrame):
    """
    Reshape long-format (param_id × time) DataFrame into a 3-D NumPy array
    of shape (n_params, n_times, n_columns).

    Returns
    -------
    tensor_array : np.ndarray, shape (n_params, n_times, n_columns)
    column_names : list[str]
    """
    df = df.reset_index(drop=True).select_dtypes(include=np.number)
    grouped = df.groupby("param_id")
    df["index"] = grouped.cumcount()
    df = df.set_index(["index", "param_id"])
    tensor = df.unstack("index")

    n_ids = df.index.get_level_values("param_id").nunique()
    n_indices = df.index.get_level_values("index").nunique()
    n_columns = len(df.columns)

    tensor_array = tensor.values.reshape(n_ids, n_columns, n_indices)
    tensor_array = np.swapaxes(tensor_array, 1, 2)
    return tensor_array, df.columns.tolist()


def _array_to_df(arr: np.ndarray, variable_name: str) -> pd.DataFrame:
    """Flatten a (n_samples, n_params, n_times) array into a tidy DataFrame."""
    n_samples, n_params, n_times = arr.shape
    return pd.DataFrame(
        {
            "sample_id": np.repeat(np.arange(n_samples), n_params * n_times),
            "param_id": np.tile(np.repeat(np.arange(n_params), n_times), n_samples),
            "time": np.tile(np.arange(n_times), n_samples * n_params),
            variable_name: arr.flatten(),
        }
    )


# ---------------------------------------------------------------------------
# Path simulation
# ---------------------------------------------------------------------------

def simulate_path(df: pd.DataFrame, dt: float, N: int, nSamples: int = 1000, nShocks: int = 20):
    """
    Simulate Monte Carlo paths of the state process under the optimal control.

    Shocks arrive at discrete intervals (``nShocks`` times per horizon) to
    approximate a compound-Poisson in-flow; set ``nShocks = N`` for a full
    Brownian driver.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format time grid with precomputed helper variables.
    dt : float
        Time step size.
    N : int
        Number of time steps.
    nSamples : int
        Number of Monte Carlo paths.
    nShocks : int
        Number of shock arrivals per horizon.

    Returns
    -------
    summary_stats : pd.DataFrame
        Per-(sample, param) scalar statistics.
    timeseries_df : pd.DataFrame
        Per-(sample, param, time) state trajectories.
    """
    dWs_full = np.sqrt(dt) * np.random.randn(nSamples, N)
    Ws = np.concatenate(
        (np.zeros((nSamples, 1)), np.cumsum(dWs_full, axis=1)), axis=1
    )

    tensor_array, column_names = _reshape_time_grid(df)
    c2i = {name: idx for idx, name in enumerate(column_names)}
    n_ids, n_indices, _ = tensor_array.shape

    # State arrays
    Xs = np.zeros((nSamples, n_ids, n_indices))
    Ys = np.zeros((nSamples, n_ids, n_indices))
    Zs = np.zeros((nSamples, n_ids, n_indices))
    qs = np.zeros((nSamples, n_ids, n_indices))

    # Initial conditions
    J0 = tensor_array[:, 0, c2i["J0"]].reshape(1, n_ids)
    z0 = tensor_array[:, 0, c2i["z"]].reshape(1, n_ids)
    y0 = tensor_array[:, 0, c2i["y"]].reshape(1, n_ids)
    lamb0 = tensor_array[:, 0, c2i["lamb"]].reshape(1, n_ids)

    Xs[:, :, 0] = J0 - z0
    Ys[:, :, 0] = y0 + lamb0 * J0
    Zs[:, :, 0] = z0

    # Rescale shocks so that total quadratic variation is preserved
    dWs = np.zeros((nSamples, N))
    wait = N // nShocks
    for i in range(N):
        dWs[:, i] = Ws[:, i + 1] - Ws[:, i]
        if i % wait != 0:
            dWs[:, i] = 0
        else:
            dWs[:, i] *= np.sqrt(wait)

    # Forward simulation
    for i in range(N):
        theta_i = tensor_array[:, i, c2i["theta"]].reshape(1, n_ids)
        sigma_i = tensor_array[:, i, c2i["sigma"]].reshape(1, n_ids)
        beta_i = tensor_array[:, i, c2i["beta"]].reshape(1, n_ids)
        lamb_i = tensor_array[:, i, c2i["lamb"]].reshape(1, n_ids)

        dZ = -theta_i * Zs[:, :, i] * dt + sigma_i * dWs[:, i].reshape(nSamples, 1)
        Xs[:, :, i + 1] = Xs[:, :, i] + qs[:, :, i] * dt - dZ
        Ys[:, :, i + 1] = Ys[:, :, i] + (-beta_i * Ys[:, :, i] + lamb_i * qs[:, :, i]) * dt
        Zs[:, :, i + 1] = Zs[:, :, i] + dZ

        # Optimal control at next step
        fs_i1 = tensor_array[:, i + 1, c2i["fs"]].reshape(1, n_ids)
        gs_i1 = tensor_array[:, i + 1, c2i["gs"]].reshape(1, n_ids)
        hs_i1 = tensor_array[:, i + 1, c2i["hs"]].reshape(1, n_ids)
        qs[:, :, i + 1] = (
            fs_i1 * Xs[:, :, i + 1]
            + gs_i1 * Ys[:, :, i + 1]
            + hs_i1 * Zs[:, :, i + 1]
        )

    # Terminal adjustments
    qs[:, :, -1] = 0
    Xs[:, :, 0] = -z0
    Xs[:, :, -1] = 0
    JT = Zs[:, :, -1] - J0 - qs.sum(axis=2) * dt
    Ys[:, :, -1] = Ys[:, :, -2] + lamb0 * JT

    tradesSum = J0 + qs.cumsum(axis=2) * dt
    tradesSum[:, :, -1] += JT
    tradesSum[:, :, 0] = 0

    # Cost and flow statistics
    spreadCost = 0.5 * tensor_array[:, 0, c2i["eps"]].reshape(1, n_ids) * np.sum(qs ** 2, axis=2) * dt
    impactCost = (
        0.5 * (Ys[:, :, 0] + Ys[:, :, 1]) * J0
        + np.sum(Ys * qs, axis=2) * dt
        + 0.5 * (Ys[:, :, -2] + Ys[:, :, -1]) * JT
    )
    intradayTrds = np.sum(np.abs(qs), axis=2) * dt
    TVOutFlows = intradayTrds + np.abs(J0) + np.abs(JT)
    TVInFlows = np.abs(z0) + np.abs(Zs[:, :, :-1] - Zs[:, :, 1:]).sum(axis=2)
    QVInFlows = np.square(z0) + np.square(Zs[:, :, :-1] - Zs[:, :, 1:]).sum(axis=2)
    JTProp = np.abs(JT) / TVOutFlows

    ids = np.arange(n_ids).reshape(1, n_ids).repeat(nSamples, axis=0)
    sample_ids = np.arange(nSamples).reshape(nSamples, 1).repeat(n_ids, axis=1)

    summary_stats = pd.DataFrame(
        {
            "param_id": ids.flatten(),
            "sample_id": sample_ids.flatten(),
            "spreadCost": spreadCost.flatten(),
            "impactCost": impactCost.flatten(),
            "TVOutFlows": TVOutFlows.flatten(),
            "TVInFlows": TVInFlows.flatten(),
            "QVInFlows": QVInFlows.flatten(),
            "JTProp": JTProp.flatten(),
            "intradayTrds": intradayTrds.flatten(),
            "XT": np.abs(Xs[:, :, -1]).flatten(),
            "YT": np.abs(Ys[:, :, -1]).flatten(),
            "ZT": np.abs(Zs[:, :, -1]).flatten(),
        }
    )
    numeric_cols = ["spreadCost", "impactCost", "TVOutFlows", "TVInFlows",
                    "QVInFlows", "JTProp", "intradayTrds", "XT", "YT", "ZT"]
    summary_stats[numeric_cols] = summary_stats[numeric_cols].apply(pd.to_numeric, errors="coerce")

    # Build tidy timeseries DataFrame
    tradesSum_df = _array_to_df(tradesSum, "tradeSum")
    Xs_df = _array_to_df(Xs, "Xs")
    Ys_df = _array_to_df(Ys, "Ys")
    Zs_df = _array_to_df(Zs, "Zs")
    timeseries_df = (
        tradesSum_df
        .merge(Xs_df, on=["sample_id", "param_id", "time"])
        .merge(Ys_df, on=["sample_id", "param_id", "time"])
        .merge(Zs_df, on=["sample_id", "param_id", "time"])
    )
    return summary_stats, timeseries_df


def simulate(
    parameters: pd.DataFrame,
    T: float = 1.0,
    N: int = 200,
    nSamples: int = 1000,
    nShocks: int = 20,
):
    """
    End-to-end simulation pipeline.

    Parameters
    ----------
    parameters : pd.DataFrame
        Output of ``generate_parameters``.
    T : float
        Trading horizon.
    N : int
        Number of time steps.
    nSamples : int
        Number of Monte Carlo paths.
    nShocks : int
        Number of in-flow shock arrivals per horizon.

    Returns
    -------
    summary_stats, timeseries_df, df, N
    """
    parameters = add_auxiliary_variables(parameters)
    df, T, N, dt = add_time_grid(parameters, T, N)
    df, T = add_helper_variables(df, T)
    summary_stats, timeseries_df = simulate_path(df, dt, N, nSamples, nShocks)
    return summary_stats, timeseries_df, df, N
