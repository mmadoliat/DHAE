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


# =====================================================================
# Design exactly as described in Section 3 of the HAE paper
# (used by HAE/run_simulation.py; the functions above are kept for VHAE/
#  and for the legacy notebook procedure)
# =====================================================================
from dataclasses import dataclass, asdict
from .simulate import _latent_var1_matrix
from .utils import set_seed

PAPER_KS = (3, 5, 7)
PAPER_BS = (10, 20, 30, 40, 50)


@dataclass
class PaperSimCfg:
    # data (paper Sec. 3 and Appendix Table 'nonlinear_sim_notation')
    N: int = 500                    # T + L curves
    J: int = 100                    # grid points on [0, 1]
    d: int = 5                      # dimension of the data-generating latent state
    L: int = 5                      # held-out curves (forecast horizon)
    gen_n_basis: int = 30           # B-splines of the linear generating map
    sigma_h_nonlinear: float = 0.028
    sigma_eta_nonlinear: float = 0.001
    sigma_h_linear: float = 0.08    # NOT stated in the paper (value of the linear notebook)
    sigma_eta_linear: float = 0.5   # NOT stated in the paper (value of the linear notebook)
    seed_base: int = 30000          # data seed of replicate r = seed_base + r
    # FAE (paper: B-128-K / K-128-B, tanh, Adam lr 7e-4, batch 16, 1400 epochs, lambda 2e-4)
    hidden: int = 128
    activation: str = "tanh"
    lr: float = 7e-4
    batch_size: int = 16
    epochs: int = 1400
    smooth_lambda: float = 2e-4
    # FAE details NOT stated in the paper (values of the amplitude--phase notebook)
    weight_decay: float = 1e-5
    split_rate: float = 0.85        # chronological 85/15 split, best-validation checkpoint
    init_sd: float = 0.06
    fae_seed: int = 743
    standardize: bool = True        # centre + scale curves before the FAE (training fails on the raw scale)
    var_ridge: float = 1e-6         # VAR(1) by (essentially) least squares


def simulate_linear_paper(c: PaperSimCfg, data_seed: int):
    """Linear benchmark: stable VAR(1) latent state -> fixed linear map (d -> 20 -> gen_n_basis)
    -> B-spline coefficients -> curve on [0, 1]; Gaussian noise. The map and A are fixed across
    replicates (seed 123); innovations and noise depend on data_seed only."""
    cfg = SimCfg(scenario="linear", T=c.N, P=c.J, d=c.d, Sigma_scale=c.sigma_h_linear,
                 meas_noise_sd=c.sigma_eta_linear, gen_n_basis=c.gen_n_basis)
    fixed = build_linear_decoder(cfg)
    rng = np.random.default_rng(data_seed)
    Z = np.zeros((c.N, c.d)); S = fixed["Sigma"]
    Z[0] = rng.multivariate_normal(np.zeros(c.d), S)
    for t in range(1, c.N):
        Z[t] = fixed["A"] @ Z[t - 1] + rng.multivariate_normal(np.zeros(c.d), S)
    with torch.no_grad():
        Xc = fixed["mapper"](torch.tensor(Z, dtype=torch.float32)) @ fixed["Bgen"].T
        g = torch.Generator().manual_seed(data_seed + 999)
        Xn = Xc + c.sigma_eta_linear * torch.randn(Xc.shape, generator=g)
    return Xc, Xn


def make_data_paper(scenario: str, rep: int, c: PaperSimCfg):
    if scenario == "linear":
        Xc, Xn = simulate_linear_paper(c, c.seed_base + rep)
    else:
        cfg = SimCfg(scenario="nonlinear", T=c.N, P=c.J, d=c.d, Sigma_scale=c.sigma_h_nonlinear,
                     meas_noise_sd=c.sigma_eta_nonlinear)
        Xc, Xn, _, _ = simulate_nonlinear(cfg, c.seed_base + rep)
    return Xc.float(), Xn.float(), np.linspace(0.0, 1.0, c.J)


def fpca_on_basis(X_tr: torch.Tensor, u: np.ndarray, B: int, K: int):
    """FPCA of the B-dimensional cubic B-spline representation of the training curves
    (least-squares coefficients), centred, L2 inner product by trapezoid weights."""
    Bm = make_basis(torch.tensor(u, dtype=torch.float32), B).double()          # [J, B]
    coef = torch.linalg.lstsq(Bm, X_tr.double().T).solution                     # [B, T]
    Xs = (Bm @ coef).T.float()
    mu, phi, sc, _ = weighted_fpca_fit(Xs, trapezoid_weights(u), K)
    return mu, phi, sc


def run_replicate_paper(scenario: str, rep: int, Ks=PAPER_KS, Bs=PAPER_BS, c: PaperSimCfg | None = None):
    """One simulated data set, all (K, B) cells: FPCA+VAR(1) and FAE+VAR(1).
    relMSE (eq. relmse) against the noise-free curves; reconstruction on the T training
    curves, forecasting on the L held-out curves."""
    import time
    c = c or PaperSimCfg()
    Xc, Xn, u = make_data_paper(scenario, rep, c)
    Lh = c.L
    Xc_tr, Xc_te, Xn_tr, Xn_te = Xc[:-Lh], Xc[-Lh:], Xn[:-Lh], Xn[-Lh:]
    tpts = torch.tensor(u, dtype=torch.float32)
    if c.standardize:
        m0 = Xn_tr.mean(0, keepdim=True); s0 = float((Xn_tr - m0).std())
    else:
        m0 = torch.zeros(1, Xn_tr.shape[1]); s0 = 1.0
    Z = (Xn_tr - m0) / s0
    rows = []
    for K in Ks:
        for B in Bs:
            t0 = time.time()
            row = dict(scenario=scenario, rep=rep, data_seed=c.seed_base + rep, K=K, B=B)
            # ---- FPCA + VAR(1)
            mu, phi, sc = fpca_on_basis(Xn_tr, u, B, K)
            H = sc.numpy().astype(np.float64); b, A, _ = fit_var1(H, c.var_ridge)
            Hf = forecast_var1(H[-1], b, A, Lh)
            xr = mu[None] + sc @ phi; xf = mu[None] + torch.tensor(Hf, dtype=torch.float32) @ phi
            row |= dict(fpca_recon=rel_mse(xr, Xc_tr), fpca_fore=rel_mse(xf, Xc_te),
                        fpca_recon_noisy=rel_mse(xr, Xn_tr), fpca_fore_noisy=rel_mse(xf, Xn_te))
            # ---- FAE + VAR(1) (two-stage: AE first, then VAR on the fitted latents)
            Bp, Br = make_basis(tpts, B), make_basis(tpts, B)
            fac = partial(FAEVanilla, B, K, B, hidden=c.hidden, activation=c.activation, bias=True,
                          init_weight_sd=c.init_sd)
            tc = PhaseFAECfg(seed=c.fae_seed, epochs=c.epochs, batch_size=c.batch_size, lr=c.lr,
                             weight_decay=c.weight_decay, split_rate=c.split_rate,
                             smooth_lambda=c.smooth_lambda, dyn_lambda=0.0, clean_target=False)
            model = train_fae_phase(fac, Z, Z, tpts, Bp, Br, tc)
            with torch.no_grad():
                zr, Ht, _, _ = model(Z, tpts, Bp, Br)
                H = Ht.numpy().astype(np.float64); b, A, _ = fit_var1(H, c.var_ridge)
                Hf = forecast_var1(H[-1], b, A, Lh)
                zf, _ = model.decode_from_rep(torch.tensor(Hf, dtype=torch.float32), Br)
            xr, xf = zr * s0 + m0, zf * s0 + m0
            row |= dict(fae_recon=rel_mse(xr, Xc_tr), fae_fore=rel_mse(xf, Xc_te),
                        fae_recon_noisy=rel_mse(xr, Xn_tr), fae_fore_noisy=rel_mse(xf, Xn_te),
                        sec=time.time() - t0)
            rows.append(row)
    return rows
