"""Linear baselines: weighted FPCA (centered, robust two-step, uncentered),
hybrid PCA on concatenated standardized data, and functional PCA."""
from __future__ import annotations
import numpy as np
import torch
from sklearn.decomposition import PCA
from .forecast import forecast_scores_time_series


def weighted_fpca_fit(X_train: torch.Tensor, w: torch.Tensor, K: int):
    """Centered FPCA with inner product <f,g> = sum_j f_j g_j w_j.
    Returns mu [P], phi [K,P] (w-orthonormal), scores [T,K], eigvals [K]."""
    T = X_train.shape[0]
    mu = X_train.mean(dim=0, keepdim=True)
    Xc = X_train - mu
    sw = torch.sqrt(w).view(1, -1)
    U, S, Vh = torch.linalg.svd(Xc * sw, full_matrices=False)
    V = Vh.transpose(0, 1)
    K = min(K, V.shape[1])
    phi = (V[:, :K] / sw.flatten()[:, None]).T
    for k in range(K):
        phi[k] = phi[k] / torch.sqrt(torch.sum(phi[k] * phi[k] * w)).clamp_min(1e-12)
    scores = (Xc * w.view(1, -1)) @ phi.T
    return mu.squeeze(0), phi, scores, (S[:K] ** 2) / max(T - 1, 1)


def robust_fpca_two_step(X_train, w, K, zeta: float = 3.0):
    """Two-step robust FPCA in the spirit of Hyndman & Ullah (2007, Sec. 3.3)."""
    mu1, phi1, s1, _ = weighted_fpca_fit(X_train, w, K)
    resid = X_train - (mu1.view(1, -1) + s1 @ phi1)
    v = torch.sum(resid * resid * w.view(1, -1), dim=1)
    s = torch.median(v)
    year_w = (v < s + zeta * torch.sqrt(s.clamp_min(1e-12))).float()
    mu2, phi2, _, ev2 = weighted_fpca_fit(X_train * torch.sqrt(year_w).view(-1, 1), w, K)
    scores2 = ((X_train - mu2.view(1, -1)) * w) @ phi2.T
    return mu2, phi2, scores2, ev2, year_w


def uncentered_weighted_fpca_fit(X_train, w, K):
    """Uncentered weighted PCA (linear simulation notebook)."""
    sw = torch.sqrt(w).view(1, -1)
    U, S, Vh = torch.linalg.svd(X_train * sw, full_matrices=False)
    V = Vh.transpose(0, 1)
    phi = (V[:, :K] / sw.flatten()[:, None]).T
    for k in range(K):
        phi[k] = phi[k] / torch.sqrt(torch.sum(phi[k] * phi[k] * w)).clamp_min(1e-12)
    return phi, (X_train * w.view(1, -1)) @ phi.T


def hpca_forecast(x_train: np.ndarray, y_train: np.ndarray, horizon: int, latent_dim: int,
                  maxlags: int, return_model: bool = False):
    """HPCA baseline: PCA on [x, y] (each already standardized), AIC-VAR on scores."""
    n_grid = x_train.shape[1]
    comb = np.concatenate([x_train, y_train], axis=1)
    pca = PCA(n_components=latent_dim)
    scores = pca.fit_transform(comb)
    out = forecast_scores_time_series(scores, horizon, maxlags, return_model=return_model)
    fc, info, scaler = out[:3]
    cf, cr = pca.inverse_transform(fc), pca.inverse_transform(scores)
    res = dict(x_forecast=cf[:, :n_grid], y_forecast=cf[:, n_grid:],
               x_recon=cr[:, :n_grid], y_recon=cr[:, n_grid:], scores_train=scores,
               scores_forecast=fc, pca=pca, forecast_info=info, score_scaler=scaler)
    if return_model:
        res["var_result"] = out[3]
    return res


def functional_pca_forecast(x_train: np.ndarray, horizon: int, latent_dim: int, maxlags: int):
    """Functional-only PCA on the common grid + AIC-VAR on the scores (Metro FPCA baseline)."""
    k = min(latent_dim, x_train.shape[0] - 1, x_train.shape[1])
    pca = PCA(n_components=k)
    scores = pca.fit_transform(x_train)
    fc, info, scaler = forecast_scores_time_series(scores, horizon, maxlags)
    return dict(x_forecast=pca.inverse_transform(fc), x_recon=pca.inverse_transform(scores),
                scores_train=scores, scores_forecast=fc, pca=pca, forecast_info=info)
