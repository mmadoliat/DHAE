"""Basis systems and quadrature weights.

Two B-spline constructions are kept on purpose, because the original
experiments used them:

* ``bspline_basis_grid``: Cox--de Boor recursion on the grid rescaled to [0,1]
  with an open uniform knot vector (fertility and simulation notebooks).
  Returns ``[n_time, n_basis]``.
* ``bspline_basis_scipy``: scipy ``BSpline`` with open uniform knots on
  [min(t), max(t)] (Metro notebook). Returns ``[n_basis, n_time]``.
"""
from __future__ import annotations
import math
import numpy as np
import torch
from scipy.interpolate import BSpline


def _tau(tpts: torch.Tensor) -> torch.Tensor:
    t = tpts.flatten()
    return (t - t.min()) / (t.max() - t.min()).clamp_min(1e-8)


def fourier_basis(tpts: torch.Tensor, n_basis: int) -> torch.Tensor:
    tau = _tau(tpts)
    B = torch.zeros(tau.shape[0], n_basis, dtype=torch.float32, device=tau.device)
    if n_basis > 0:
        B[:, 0] = 1.0
    k, idx = 1, 1
    while idx < n_basis:
        B[:, idx] = torch.sin(2.0 * math.pi * k * tau); idx += 1
        if idx < n_basis:
            B[:, idx] = torch.cos(2.0 * math.pi * k * tau); idx += 1
        k += 1
    return B


def bspline_basis_grid(tpts: torch.Tensor, n_basis: int, degree: int = 3) -> torch.Tensor:
    """Cox--de Boor B-spline basis, shape [n_time, n_basis] (port of BasisFCBuilder)."""
    tau_np = _tau(tpts).detach().cpu().numpy()
    n_time, p = tau_np.shape[0], degree
    if n_basis < p + 1:
        raise ValueError(f"n_basis={n_basis} must be at least degree+1={p+1}.")
    n_int = max(n_basis - p - 1, 0)
    if n_int > 0:
        interior = np.linspace(0.0, 1.0, n_int + 2)[1:-1]
        knots = np.concatenate((np.zeros(p + 1), interior, np.ones(p + 1)))
    else:
        knots = np.concatenate((np.zeros(p + 1), np.ones(p + 1)))
    N = np.zeros((n_basis, n_time), dtype=np.float64)
    for i in range(n_basis):
        N[i, :] = np.where((tau_np >= knots[i]) & (tau_np < knots[i + 1]), 1.0, 0.0)
    N[-1, tau_np == 1.0] = 1.0
    for k in range(1, p + 1):
        N_next = np.zeros_like(N)
        for i in range(n_basis):
            dl = knots[i + k] - knots[i]
            left = (tau_np - knots[i]) / dl * N[i, :] if dl > 0 else 0.0
            dr = knots[i + k + 1] - knots[i + 1] if (i + 1) < n_basis else 0.0
            right = ((knots[i + k + 1] - tau_np) / dr * N[i + 1, :]
                     if (dr > 0 and (i + 1) < n_basis) else 0.0)
            N_next[i, :] = left + right
        N = N_next
    return torch.tensor(N.T, dtype=torch.float32, device=tpts.device)


def make_basis(tpts: torch.Tensor, n_basis: int, basis_type: str = "bspline", degree: int = 3):
    bt = basis_type.lower().replace("-", "").replace("_", "")
    if bt == "fourier":
        return fourier_basis(tpts, n_basis)
    if bt == "bspline":
        return bspline_basis_grid(tpts, n_basis, degree)
    raise ValueError("basis_type must be 'fourier' or 'bspline'")


def bspline_basis_scipy(t_grid, n_basis: int, degree: int = 3) -> torch.Tensor:
    """scipy B-spline basis on [min t, max t], shape [n_basis, n_time] (Metro notebook)."""
    t_grid = np.asarray(t_grid, dtype=float)
    a, b = float(t_grid.min()), float(t_grid.max())
    if n_basis <= degree:
        raise ValueError("n_basis must be larger than degree.")
    n_int = n_basis - degree - 1
    interior = np.linspace(a, b, n_int + 2)[1:-1] if n_int > 0 else np.array([])
    knots = np.concatenate([np.repeat(a, degree + 1), interior, np.repeat(b, degree + 1)])
    rows = []
    for i in range(n_basis):
        c = np.zeros(n_basis); c[i] = 1.0
        v = BSpline(knots, c, degree, extrapolate=False)(t_grid)
        rows.append(np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0))
    return torch.tensor(np.vstack(rows), dtype=torch.float32)


def trapezoid_weights(x) -> torch.Tensor:
    """Trapezoid quadrature weights for an increasing grid."""
    x = np.asarray(x, dtype=float)
    dx = np.diff(x)
    w = np.zeros_like(x)
    w[0], w[-1] = dx[0] / 2.0, dx[-1] / 2.0
    if len(x) > 2:
        w[1:-1] = (x[2:] - x[:-2]) / 2.0
    return torch.tensor(w, dtype=torch.float32)


def trapezoid_weights_torch(tpts: torch.Tensor) -> torch.Tensor:
    """Same weights computed in torch, as in FAEVanilla.project."""
    t = tpts.flatten()
    dt = t[1:] - t[:-1]
    zero = torch.zeros(1, device=t.device, dtype=t.dtype)
    return 0.5 * torch.cat([zero, dt]) + 0.5 * torch.cat([dt, zero])
