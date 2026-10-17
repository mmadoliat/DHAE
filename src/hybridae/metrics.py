"""Point and probabilistic accuracy measures."""
from __future__ import annotations
import numpy as np
import torch
from .utils import to_numpy


def mse(pred, truth) -> float:
    return float(np.mean((to_numpy(pred) - to_numpy(truth)) ** 2))


def rel_mse(pred, truth, eps: float = 1e-12) -> float:
    """relMSE = MSE / Var(truth), Var over all entries (population). Eq. (relMSE) of the HAE paper."""
    p, t = to_numpy(pred).astype(np.float64), to_numpy(truth).astype(np.float64)
    return float(np.mean((p - t) ** 2) / max(np.var(t), eps))


def rel_rmse(pred, truth) -> float:
    return float(np.sqrt(rel_mse(pred, truth)))


def rel_mse_meansq(pred, truth, eps: float = 1e-12) -> float:
    """MSE / mean(truth^2): the 'relative_mse' used in the Metro notebook tables."""
    p, t = to_numpy(pred), to_numpy(truth)
    d = float(np.mean(t ** 2))
    return float("nan") if d <= eps else float(np.mean((p - t) ** 2) / d)


# ---------------- probabilistic ----------------
def coverage(samples, truth, level: float = 0.9) -> float:
    """Empirical coverage of pointwise central intervals. samples: [S, ...]."""
    s, t = to_numpy(samples), to_numpy(truth)
    a = (1 - level) / 2
    lo, hi = np.quantile(s, a, axis=0), np.quantile(s, 1 - a, axis=0)
    return float(np.mean((t >= lo) & (t <= hi)))


def mean_width(samples, level: float = 0.9) -> float:
    s = to_numpy(samples); a = (1 - level) / 2
    return float(np.mean(np.quantile(s, 1 - a, axis=0) - np.quantile(s, a, axis=0)))


def interval_score(samples, truth, level: float = 0.9) -> float:
    """Mean interval score of Gneiting & Raftery (2007) for central (1-alpha) intervals."""
    s, t = to_numpy(samples), to_numpy(truth)
    alpha = 1 - level
    lo, hi = np.quantile(s, alpha / 2, axis=0), np.quantile(s, 1 - alpha / 2, axis=0)
    sc = (hi - lo) + (2 / alpha) * (lo - t) * (t < lo) + (2 / alpha) * (t - hi) * (t > hi)
    return float(np.mean(sc))


def crps_samples(samples, truth, max_pairs: int = 400) -> float:
    """Sample-based CRPS, E|X-y| - 0.5 E|X-X'|, averaged over all entries."""
    s, t = to_numpy(samples).astype(np.float64), to_numpy(truth).astype(np.float64)
    s = s[:max_pairs]
    term1 = np.mean(np.abs(s - t[None]), axis=0)
    term2 = np.mean(np.abs(s[:, None] - s[None, :]), axis=(0, 1))
    return float(np.mean(term1 - 0.5 * term2))


def prob_summary(samples, truth, level: float = 0.9) -> dict:
    return {
        f"coverage{int(level*100)}": coverage(samples, truth, level),
        f"width{int(level*100)}": mean_width(samples, level),
        f"interval_score{int(level*100)}": interval_score(samples, truth, level),
        "crps": crps_samples(samples, truth),
    }
