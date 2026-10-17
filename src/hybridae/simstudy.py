"""One replicate of the simulation study: FPCA+VAR, FAE+VAR and VFAE+VAR (point and
probabilistic forecasts). Used by HAE/run_simulation.py and VHAE/run_simulation.py."""
from __future__ import annotations
from functools import partial
import numpy as np
import torch
from .simulate import SimCfg, build_linear_decoder, simulate_linear, simulate_nonlinear
from .bases import make_basis, trapezoid_weights
from .baselines import weighted_fpca_fit, uncentered_weighted_fpca_fit
from .forecast import fit_var1, forecast_var1, simulate_var1
from .models import FAEVanilla, VFAEVanilla
from .train import FAETrainCfg, PhaseFAECfg, train_fae_random_split, train_fae_phase
from .metrics import rel_mse, prob_summary

L = 5


def make_data(scenario, rep, seed_base=30000, fixed=None):
    if scenario == "linear":
        cfg = SimCfg(scenario="linear", Sigma_scale=0.08, meas_noise_sd=0.5)
        fixed = fixed or build_linear_decoder(cfg)
        Xc, Xn, _ = simulate_linear(fixed, cfg, seed_base + rep)
        return Xc, Xn, fixed["u"], fixed
    cfg = SimCfg(scenario="nonlinear")
    Xc, Xn, _, u = simulate_nonlinear(cfg, seed_base + rep)
    return Xc, Xn, u, None


def _sample_forecasts(model, H, b, A, resid, Br, x_res_sd, n_paths, rng, mu=None, logvar=None):
    """Monte-Carlo predictive curves: (optional posterior draw of h_T) -> VAR(1) paths -> decoder -> + obs noise."""
    if mu is not None:
        hT = mu[-1][None] + np.exp(0.5 * logvar[-1])[None] * rng.standard_normal((n_paths, H.shape[1]))
    else:
        hT = np.repeat(H[-1][None], n_paths, 0)
    paths = simulate_var1(hT, b, A, resid, L, rng)                         # [S, L, K]
    with torch.no_grad():
        xs, _ = model.decode_from_rep(torch.tensor(paths.reshape(-1, H.shape[1]), dtype=torch.float32), Br)
    xs = xs.numpy().reshape(n_paths, L, -1)
    return xs + x_res_sd[None, None] * rng.standard_normal(xs.shape)


def run_replicate(scenario, rep, K, B, methods=("fpca", "fae"), fae_mode="notebook", beta=1e-3,
                  seed_base=30000, n_paths=500, fixed=None):
    Xc, Xn, u, fixed = make_data(scenario, rep, seed_base, fixed)
    Xc_tr, Xc_te, Xn_tr, Xn_te = Xc[:-L], Xc[-L:], Xn[:-L], Xn[-L:]
    tpts = torch.tensor(u, dtype=torch.float32)
    out = dict(scenario=scenario, rep=rep, data_seed=seed_base + rep, K=K, B=B, fae_mode=fae_mode)
    rng = np.random.default_rng(seed_base + rep)
    w = trapezoid_weights(u)
    ridge = 1e-8 if scenario == "linear" else 1e-6
    if "fpca" in methods:
        if scenario == "linear":
            phi, sc = uncentered_weighted_fpca_fit(Xn_tr, w, K); mu0 = torch.zeros(Xn_tr.shape[1])
        else:
            mu0, phi, sc, _ = weighted_fpca_fit(Xn_tr, w, K)
        H = sc.numpy().astype(np.float64); b, A, resid = fit_var1(H, ridge)
        Hf = forecast_var1(H[-1], b, A, L)
        xr = mu0[None] + sc @ phi; xf = (mu0[None] + torch.tensor(Hf, dtype=torch.float32) @ phi)
        out |= dict(fpca_recon=rel_mse(xr, Xc_tr), fpca_fore=rel_mse(xf, Xc_te), fpca_fore_noisy=rel_mse(xf, Xn_te))
        rsd = (Xn_tr - xr).std(0).numpy()
        paths = simulate_var1(np.repeat(H[-1][None], n_paths, 0), b, A, resid, L, rng)
        xs = mu0.numpy()[None, None] + paths @ phi.numpy()
        xs = xs + rsd[None, None] * rng.standard_normal(xs.shape)
        out |= {f"fpca_{k}": v for k, v in prob_summary(xs, Xn_te.numpy()).items()}
    for name in [m for m in methods if m in ("fae", "vfae")]:
        var = name == "vfae"
        std = fae_mode == "twostage"          # unit-scale curves for the two-stage FAE and VFAE
        sc = float(Xn_tr.std()) if std else 1.0
        Xn_tr_s, Xc_tr_s = Xn_tr / sc, Xc_tr / sc
        Bp, Br = make_basis(tpts, B), make_basis(tpts, B)
        if scenario == "linear":
            cls = VFAEVanilla if var else FAEVanilla
            fac = partial(cls, B, K, B, hidden=100, activation="relu", bias=False, init_weight_sd=0.5, **({"logvar_init": -6.0} if var else {}))
            cfg = FAETrainCfg(seed=743, epochs=2000, lr=3e-4, weight_decay=1e-4, lamb=1e-3,
                              beta=beta if var else 0.0, kl_warmup=200 if var else 0)
            model, _ = train_fae_random_split(fac, Xn_tr_s, tpts, Bp, Br, cfg)
        else:
            cls = VFAEVanilla if var else FAEVanilla
            fac = partial(cls, B, K, B, hidden=128, activation="tanh", bias=True, init_weight_sd=0.06, **({"logvar_init": -6.0} if var else {}))
            nb = (fae_mode == "notebook") and not var
            cfg = PhaseFAECfg(clean_target=nb, dyn_lambda=8.0 if nb else 0.0,
                              beta=beta if var else 0.0, kl_warmup=150 if var else 0)
            model = train_fae_phase(fac, Xn_tr_s, Xc_tr_s, tpts, Bp, Br, cfg)
        with torch.no_grad():
            xr, Ht, _, _ = model(Xn_tr_s, tpts, Bp, Br); xr = xr * sc
            if var:
                mu_, lv_ = model.posterior(model.project(Xn_tr_s, tpts, Bp)); mu_, lv_ = mu_.numpy(), lv_.numpy()
        H = Ht.numpy().astype(np.float64); b, A, resid = fit_var1(H, ridge)
        Hf = forecast_var1(H[-1], b, A, L)
        with torch.no_grad():
            xf, _ = model.decode_from_rep(torch.tensor(Hf, dtype=torch.float32), Br); xf = xf * sc
        out |= {f"{name}_recon": rel_mse(xr, Xc_tr), f"{name}_fore": rel_mse(xf, Xc_te), f"{name}_fore_noisy": rel_mse(xf, Xn_te)}
        rsd = (Xn_tr - xr).std(0).numpy()
        xs = sc * _sample_forecasts(model, H, b, A, resid, Br, rsd / sc, n_paths, rng,
                               *( (mu_, lv_) if var else (None, None) ))
        out |= {f"{name}_{k}": v for k, v in prob_summary(xs, Xn_te.numpy()).items()}
        if var:
            out[f"{name}_mean_post_sd"] = float(np.exp(0.5 * lv_).mean()); out[f"{name}_H_sd"] = float(H.std(0).mean())
    return out, fixed
