"""Simulation designs of the HAE paper (ports of the simulation notebooks).

linear     : latent stable VAR(1) -> fixed random linear map (d->20->M) ->
             M cubic B-spline coefficients -> curve on [0,1]; Gaussian noise.
             The map, basis and A are fixed across replicates; only the VAR
             innovations and noise change (``Simulation Runs.ipynb``).
nonlinear  : amplitude--phase benchmark (``Simulation Runs_Phase-warped.ipynb``).
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
from .bases import bspline_basis_grid
from .utils import set_seed


@dataclass
class SimCfg:
    scenario: str = "nonlinear"
    seed: int = 123
    T: int = 500
    P: int = 100
    d: int = 5
    Sigma_scale: float = 0.028      # nonlinear default; linear notebook default 0.08
    target_rho: float = 0.85
    gen_n_basis: int = 12           # NB: paper text says 30 (CHECK)
    map_weight_sd: float = 0.8
    meas_noise_sd: float = 0.001    # nonlinear default; linear runs used 0.5
    age_min: float = 15.0
    age_max: float = 49.0
    tail_strength: float = 0.012


def _stabilize(A, target_rho):
    rho = max(abs(np.linalg.eigvals(A)))
    return A * (target_rho / rho) if rho > target_rho else A


def _latent_var1_matrix(cfg: SimCfg):
    rng = np.random.default_rng(cfg.seed)
    A = _stabilize(rng.normal(size=(cfg.d, cfg.d)) * 0.15, cfg.target_rho)
    return A


class _LinearMap(nn.Module):
    def __init__(self, d, hidden, M):
        super().__init__()
        dims = [d] + hidden + [M]
        self.layers = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])

    def forward(self, z):
        for l in self.layers:
            z = l(z)
        return z


def build_linear_decoder(cfg: SimCfg):
    set_seed(cfg.seed)
    u = np.linspace(0.0, 1.0, cfg.P)
    tpts = torch.tensor(u, dtype=torch.float32)
    Bgen = bspline_basis_grid(tpts, cfg.gen_n_basis)
    A = _latent_var1_matrix(cfg)
    torch.manual_seed(cfg.seed)
    mapper = _LinearMap(cfg.d, [20], cfg.gen_n_basis)
    for m in mapper.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, cfg.map_weight_sd); nn.init.zeros_(m.bias)
    mapper.eval()
    return dict(u=u, tpts=tpts, Bgen=Bgen, A=A, mapper=mapper, Sigma=(cfg.Sigma_scale ** 2) * np.eye(cfg.d))


def simulate_linear(fixed, cfg: SimCfg, data_seed: int):
    rng = np.random.default_rng(data_seed)
    Z = np.zeros((cfg.T, cfg.d))
    Z[0] = rng.multivariate_normal(np.zeros(cfg.d), fixed["Sigma"])
    for t in range(1, cfg.T):
        Z[t] = fixed["A"] @ Z[t - 1] + rng.multivariate_normal(np.zeros(cfg.d), fixed["Sigma"])
    with torch.no_grad():
        X_clean = fixed["mapper"](torch.tensor(Z, dtype=torch.float32)) @ fixed["Bgen"].T
        X_noisy = X_clean + cfg.meas_noise_sd * torch.randn_like(X_clean)
    return X_clean, X_noisy, Z


# ---------------- amplitude--phase benchmark ----------------
def _drift(tau, d):
    v = np.zeros(d); v[0] = -0.25 * (tau - 0.5)
    if d >= 2: v[1] = 1.10 * (tau - 0.5)
    if d >= 3: v[2] = 0.25 * np.sin(2 * np.pi * tau)
    if d >= 4: v[3] = 0.20 * np.sin(np.pi * tau)
    if d >= 5: v[4] = 0.15 * np.cos(2 * np.pi * tau)
    return v


def _nonlin(z, d):
    v = np.zeros(d)
    z1 = z[1] if d >= 2 else z[0]
    v[0] = 0.10 * np.tanh(z1) - 0.04 * np.tanh(0.8 * z[0] * z1)
    if d >= 2:
        z2 = z[2] if d >= 3 else z[1]; v[1] = 0.18 * np.tanh(z2) + 0.04 * np.tanh(z[0] ** 2)
    if d >= 3:
        z3 = z[3] if d >= 4 else z[2]; v[2] = -0.10 * np.tanh(z[0]) + 0.08 * np.tanh(0.8 * z3)
    if d >= 4: v[3] = 0.08 * np.sin(np.clip(z[1], -3, 3)) + 0.04 * np.tanh(0.6 * z[2] * z[3])
    if d >= 5: v[4] = 0.06 * np.tanh(0.8 * z[4]) + 0.03 * np.tanh(0.6 * z[1] * z[2])
    return v


_A_NL = np.array([[0.60, 0.05, 0, 0, 0], [0, 0.72, 0.06, 0, 0], [0, 0, 0.66, 0.05, 0],
                  [0, 0, 0, 0.68, 0.06], [0.03, 0, 0, 0, 0.62]], dtype=float)


def latent_amplitude_phase(T, d, Sigma, seed, clip=2.0):
    rng = np.random.default_rng(seed)
    A = _A_NL[:d, :d]
    Z = np.zeros((T, d)); Z[0] = rng.multivariate_normal(np.zeros(d), Sigma)
    for t in range(1, T):
        eps = rng.multivariate_normal(np.zeros(d), Sigma)
        tau = t / max(T - 1, 1)
        Z[t] = np.clip(A @ Z[t - 1] + _nonlin(Z[t - 1], d) + 0.05 * _drift(tau, d) + eps, -clip, clip)
    return Z


def amplitude_phase_decoder(Z: torch.Tensor, cfg: SimCfg) -> torch.Tensor:
    T = Z.shape[0]
    ages = torch.linspace(cfg.age_min, cfg.age_max, cfg.P)
    u = ((ages - cfg.age_min) / (cfg.age_max - cfg.age_min)).view(1, -1)
    zs = [Z[:, i] if Z.shape[1] > i else torch.zeros(T) for i in range(5)]
    trend = torch.linspace(-1.0, 1.0, T)
    z1, z2, z3, z4, z5 = zs[0] - 0.20 * trend, zs[1] + 1.00 * trend, zs[2], zs[3], zs[4]
    amp = 0.07 + 0.07 * torch.sigmoid(z1 - 0.6 * z2)
    sh_amp = 0.16 * torch.sigmoid(1.1 * z4 + 0.5 * z5)
    tail_amp = cfg.tail_strength * torch.sigmoid(1.1 * z5 + 0.5 * z4)
    shift = (0.14 * torch.tanh(1.4 * z2 - 0.7 * z3 + 0.3 * z1 * z4)).view(T, 1)
    dil = (1.0 + 0.18 * torch.tanh(1.1 * z3 + 0.3 * z5)).view(T, 1)
    lw = (0.08 * torch.tanh(1.3 * z4 - 0.6 * z2 + 0.2 * z1 * z5).view(T, 1)
          * torch.exp(-0.5 * ((u - 0.48) / 0.16) ** 2) * torch.sin(math.pi * (u - 0.15)))
    us = (0.50 + (u - shift + lw - 0.50) / dil).clamp(-0.2, 1.2)
    main = torch.exp(-0.5 * ((us - 0.38) / 0.08) ** 2)
    shoulder = torch.exp(-0.5 * ((us - 0.58) / 0.12) ** 2)
    tail = torch.sigmoid((us - 0.62) / 0.04) * torch.exp(-(us - 0.62).clamp_min(0.0) / 0.13)
    X = amp.view(T, 1) * main + sh_amp.view(T, 1) * amp.view(T, 1) * shoulder + tail_amp.view(T, 1) * tail
    return X.clamp_min(0.0)


def simulate_nonlinear(cfg: SimCfg, data_seed: int):
    """Grid: ages 15..49 (P points). Noise generator seeded by data_seed+999 (as in the notebook)."""
    Sigma = (cfg.Sigma_scale ** 2) * np.eye(cfg.d)
    Z = latent_amplitude_phase(cfg.T, cfg.d, Sigma, data_seed)
    with torch.no_grad():
        X_clean = amplitude_phase_decoder(torch.tensor(Z, dtype=torch.float32), cfg)
        g = torch.Generator().manual_seed(data_seed + 999)
        X_noisy = X_clean + cfg.meas_noise_sd * torch.randn(X_clean.shape, generator=g)
    u = np.linspace(cfg.age_min, cfg.age_max, cfg.P)
    return X_clean, X_noisy, Z, u
