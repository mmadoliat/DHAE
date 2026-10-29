"""U.S. fertility pipeline shared by HAE/run_fertility.py and VHAE/run_fertility.py."""
from __future__ import annotations
from functools import partial
import numpy as np
import torch
from statsmodels.tsa.api import VAR
from .data import load_usa_fertility
from .bases import make_basis, trapezoid_weights
from .models import FAEVanilla, VFAEVanilla
from .train import FAETrainCfg, train_fae_random_split
from .forecast import fit_var1, forecast_var1
from .baselines import weighted_fpca_fit


def prepare_fertility(csv, train_end=2018, test_years=(2019, 2020, 2021, 2022, 2023), center="all", standardize=False,
                      smooth_s=50.0):
    """center='all' reproduces the notebook (mean over ALL years, incl. test years);
    center='train' uses the training-year mean only (no look-ahead).
    smooth_s: per-year UnivariateSpline smoothing factor (notebook 50); None = raw rates."""
    ages, years, raw, sm = load_usa_fertility(csv, smooth_s=smooth_s)
    tr, te = years <= train_end, np.isin(years, test_years)
    mu = sm.mean(0) if center == "all" else sm[tr].mean(0)
    x = sm - mu
    scale = float(x[tr].std()) if standardize else 1.0      # one global SD from training years
    x = torch.tensor(x / scale, dtype=torch.float32)
    t = (ages - ages.min()) / (ages.max() - ages.min())
    return dict(ages=ages, years=years, raw=raw, smoothed=sm, mean=mu, x_tr=x[tr], x_te=x[te],
                years_tr=years[tr], years_te=years[te], scale=scale, tpts=torch.tensor(t, dtype=torch.float32))


def fit_fae_fertility(D, K=3, B=25, variational=False, cfg: FAETrainCfg | None = None, hidden=100, logvar_init=-6.0,
                      bias=False):
    cfg = cfg or FAETrainCfg()
    Bp = make_basis(D["tpts"], B); Br = make_basis(D["tpts"], B)
    cls = VFAEVanilla if variational else FAEVanilla
    kw = dict(logvar_init=logvar_init) if variational else {}
    factory = partial(cls, B, K, B, hidden=hidden, activation="relu", bias=bias, init_weight_sd=0.5, **kw)
    model, _ = train_fae_random_split(factory, D["x_tr"], D["tpts"], Bp, Br, cfg)
    return model, Bp, Br


@torch.no_grad()
def fae_var_forecast(model, D, Bp, Br, ridge=1e-8):
    x_hat, H, _, _ = model(D["x_tr"], D["tpts"], Bp, Br)
    Hn = H.numpy().astype(np.float64)
    b, A, resid = fit_var1(Hn, ridge)
    Hf = forecast_var1(Hn[-1], b, A, len(D["years_te"]))
    xf, _ = model.decode_from_rep(torch.tensor(Hf, dtype=torch.float32), Br)
    return dict(x_recon=x_hat.numpy(), H=Hn, H_fore=Hf, x_fore=xf.numpy(), b=b, A=A, resid=resid,
                aic_lag=aic_lag(Hn))


def fpca_var_forecast(D, K=5, ridge=1e-8):
    w = trapezoid_weights(D["ages"])
    mu, phi, sc, _ = weighted_fpca_fit(D["x_tr"], w, K)
    H = sc.numpy().astype(np.float64)
    b, A, resid = fit_var1(H, ridge)
    Hf = forecast_var1(H[-1], b, A, len(D["years_te"]))
    xf = (mu.unsqueeze(0) + torch.tensor(Hf, dtype=torch.float32) @ phi).numpy()
    xr = (mu.unsqueeze(0) + sc @ phi).numpy()
    return dict(x_recon=xr, x_fore=xf, H=H, H_fore=Hf, mu=mu, phi=phi, b=b, A=A, resid=resid, aic_lag=aic_lag(H))


def aic_lag(H, maxlags=8):
    try:
        return int(VAR(H).select_order(maxlags=maxlags).aic)
    except Exception:
        return None


def var_order_table(H, maxlags=8):
    """AIC/BIC/HQIC-selected VAR orders (statsmodels select_order, common estimation sample)."""
    try:
        sel = VAR(np.asarray(H, dtype=np.float64)).select_order(maxlags=maxlags)
        return {k: int(v) for k, v in sel.selected_orders.items()}
    except Exception:
        return None


def gcv_basis_dimension(X, tpts, M_range=range(6, 41)):
    """GCV for the number M of cubic B-splines of a least-squares regression spline, pooled
    over the curves (rows of X): GCV(M) = mean squared residual / (1 - M/J)^2."""
    X = np.asarray(X, dtype=np.float64); J = X.shape[1]; out = {}
    for M in M_range:
        Bm = make_basis(tpts, M).numpy().astype(np.float64)
        Hm = Bm @ np.linalg.pinv(Bm)
        out[int(M)] = float(np.mean((X - X @ Hm.T) ** 2) / (1 - np.trace(Hm) / J) ** 2)
    return min(out, key=out.get), out
