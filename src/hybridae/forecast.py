"""Latent time-series models: ridge VAR(1) (fertility, simulation) and
AIC-selected VAR(q) on standardized scores via statsmodels (Metro), plus
Monte-Carlo simulation of future latent paths for probabilistic forecasts."""
from __future__ import annotations
import numpy as np
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.api import VAR


# ---------------- ridge VAR(1) (port) ----------------
def fit_var1(H: np.ndarray, ridge: float = 1e-8):
    """Fit h_t = b + A h_{t-1} by ridge least squares. Returns b [K], A [K,K], resid [T-1,K]."""
    T, K = H.shape
    if T < 2:
        raise ValueError("Need at least 2 time points to fit VAR(1).")
    X_aug = np.hstack([np.ones((T - 1, 1)), H[:-1]])
    XtX = X_aug.T @ X_aug
    B = np.linalg.solve(XtX + ridge * np.eye(XtX.shape[0]), X_aug.T @ H[1:])
    b, A = B[0, :], B[1:, :].T
    resid = H[1:] - X_aug @ B
    return b, A, resid


def forecast_var1(h_last: np.ndarray, b: np.ndarray, A: np.ndarray, steps: int) -> np.ndarray:
    out = np.zeros((steps, h_last.shape[0])); h = h_last.copy()
    for i in range(steps):
        h = b + A @ h; out[i] = h
    return out


def simulate_var1(h_init: np.ndarray, b, A, resid, steps: int, rng) -> np.ndarray:
    """h_init: [S, K] starting states. Gaussian innovations with the residual covariance."""
    S, K = h_init.shape
    Sig = np.cov(resid.T, ddof=1).reshape(K, K) + 1e-12 * np.eye(K)
    L = np.linalg.cholesky(Sig)
    out = np.zeros((S, steps, K)); h = h_init.copy()
    for i in range(steps):
        h = b[None] + h @ A.T + rng.standard_normal((S, K)) @ L.T
        out[:, i] = h
    return out


# ---------------- AIC-selected VAR(q) on standardized scores (port) ----------------
def forecast_scores_time_series(scores_train: np.ndarray, horizon: int, maxlags: int, return_model: bool = False):
    """Port of the Metro notebook: standardize scores, AR (K=1) or VAR (K>1) with
    the lag chosen by AIC, forecast `horizon` steps, undo the standardization."""
    scaler = StandardScaler()
    scores_std = scaler.fit_transform(np.asarray(scores_train, dtype=np.float64))
    n, k = scores_std.shape
    if k == 1:
        y = scores_std[:, 0]
        maxl = min(maxlags, max(1, n // 8))
        best, best_aic, best_lag = None, np.inf, None
        for lag in range(1, maxl + 1):
            try:
                r = AutoReg(y, lags=lag, old_names=False).fit()
                if r.aic < best_aic:
                    best, best_aic, best_lag = r, r.aic, lag
            except Exception:
                continue
        fc = scaler.inverse_transform(best.predict(start=n, end=n + horizon - 1).reshape(-1, 1))
        info = {"model": "AR", "lag": best_lag, "aic": best_aic}
        return (fc, info, scaler, best) if return_model else (fc, info, scaler)
    safe = max(1, min(maxlags, max(1, (n - 2) // (k + 1))))
    last = None
    for pmax in range(safe, 0, -1):
        try:
            model = VAR(scores_std)
            res = model.fit(maxlags=pmax, ic="aic")
            if res.k_ar == 0:
                res = model.fit(1)
            p = res.k_ar
            fc = scaler.inverse_transform(res.forecast(scores_std[-p:], steps=horizon))
            info = {"model": "VAR", "lag": p, "aic": res.aic}
            return (fc, info, scaler, res) if return_model else (fc, info, scaler)
        except Exception as err:  # pragma: no cover
            last = err
    raise RuntimeError(f"VAR fitting failed. Last error: {last}")


def simulate_var_paths(res, scaler: StandardScaler, history_raw: np.ndarray, horizon: int,
                       n_paths: int, rng, init_raw: np.ndarray | None = None,
                       innovations: bool = True) -> np.ndarray:
    """Monte-Carlo future latent paths from a fitted statsmodels VARResults.

    history_raw : [T, K] latent scores on the raw (unstandardized) scale.
    init_raw    : optional [S, p, K] sampled values of the last p latent states
                  (e.g. draws from the VHAE posterior); defaults to the observed ones.
    Returns [S, horizon, K] on the raw scale.
    """
    p, K = res.k_ar, res.neqs
    coefs, c = res.coefs, res.intercept
    Sig = np.asarray(res.sigma_u) + 1e-12 * np.eye(K)
    Lc = np.linalg.cholesky(Sig)
    if init_raw is None:
        hist = np.repeat(scaler.transform(history_raw[-p:])[None], n_paths, axis=0)
    else:
        hist = np.stack([scaler.transform(w) for w in init_raw[:, -p:]], 0)
    out = np.zeros((n_paths, horizon, K))
    for step in range(horizon):
        y = np.repeat(c[None], n_paths, 0)
        for i in range(p):
            y = y + hist[:, -1 - i] @ coefs[i].T
        if innovations:
            y = y + rng.standard_normal((n_paths, K)) @ Lc.T
        out[:, step] = y
        hist = np.concatenate([hist[:, 1:], y[:, None]], axis=1)
    return scaler.inverse_transform(out.reshape(-1, K)).reshape(n_paths, horizon, K)
