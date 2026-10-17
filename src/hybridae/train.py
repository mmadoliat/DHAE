"""Training loops (ports of the notebooks, extended with an optional KL term
for the variational models)."""
from __future__ import annotations
import copy, os
from dataclasses import dataclass, field
from typing import Callable, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from .models import kl_standard_normal, HybridAE, VariationalHybridAE, MetroFAE
from .bases import bspline_basis_scipy
from .utils import set_seed


def _epochs(n):
    """Cap epochs with HYBRIDAE_MAX_EPOCHS (smoke tests only)."""
    return min(n, int(os.environ.get("HYBRIDAE_MAX_EPOCHS", n)))


def diff_penalty(coef):
    """Squared second differences of the output basis coefficients."""
    d = coef[:, 2:] - 2 * coef[:, 1:-1] + coef[:, :-2]
    return torch.mean(torch.sum(d ** 2, dim=1))


def _kl_term(model):
    if getattr(model, "variational", False) and model.last_mu is not None:
        return torch.mean(kl_standard_normal(model.last_mu, model.last_logvar))
    return None


# =====================================================================
# FAE with random within-training split (fertility; linear simulation)
# =====================================================================
@dataclass
class FAETrainCfg:
    seed: int = 743
    epochs: int = 5000
    batch_size: int = 16
    lr: float = 3e-4
    weight_decay: float = 1e-6
    lamb: float = 1e-2
    split_rate: float = 0.85
    beta: float = 0.0          # KL weight (variational models only)
    kl_warmup: int = 0         # epochs of linear KL warm-up
    log_every: int = 0


def train_fae_random_split(model_factory: Callable[[], nn.Module], X_train_all: torch.Tensor,
                           tpts, Bp, Br, cfg: FAETrainCfg, verbose: bool = False):
    """Port of run_fae_experiment / train_fae_on_noisy_train. Seeding order is
    preserved: set_seed -> split -> model init -> shuffled minibatches."""
    set_seed(cfg.seed)
    idx = np.arange(X_train_all.shape[0])
    idx_tr, idx_va = train_test_split(idx, train_size=cfg.split_rate, random_state=cfg.seed, shuffle=True)
    Xtr, Xva = X_train_all[idx_tr].float(), X_train_all[idx_va].float()
    loader = DataLoader(Xtr, batch_size=cfg.batch_size, shuffle=True)
    model = model_factory()
    opt = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    mse = nn.MSELoss()
    for ep in range(1, _epochs(cfg.epochs) + 1):
        model.train()
        beta_t = cfg.beta * (min(1.0, ep / cfg.kl_warmup) if cfg.kl_warmup else 1.0)
        for xb in loader:
            opt.zero_grad()
            x_hat, rep, feat, coef = model(xb, tpts, Bp, Br)
            loss = mse(x_hat, xb)
            if cfg.lamb > 0:
                loss = loss + cfg.lamb * diff_penalty(coef)
            kl = _kl_term(model)
            if kl is not None and beta_t > 0:
                loss = loss + beta_t * kl
            loss.backward(); opt.step()
        if verbose and cfg.log_every and ep % cfg.log_every == 0:
            model.eval()
            with torch.no_grad():
                tr = mse(model(Xtr, tpts, Bp, Br)[0], Xtr).item(); va = mse(model(Xva, tpts, Bp, Br)[0], Xva).item()
            print(f"  epoch {ep:5d} | train {tr:.6f} | val {va:.6f}")
    model.eval()
    return model, dict(idx_train=idx_tr, idx_val=idx_va)


# =====================================================================
# FAE as in the amplitude--phase notebook (chronological split, best-val
# checkpoint, optional clean target and latent-VAR penalty)
# =====================================================================
@dataclass
class PhaseFAECfg:
    seed: int = 743
    epochs: int = 1400
    batch_size: int = 16
    lr: float = 7e-4
    weight_decay: float = 1e-5
    split_rate: float = 0.85
    smooth_lambda: float = 2e-4
    dyn_lambda: float = 8.0     # notebook value; 0 = two-stage FAE as described in the paper
    clean_target: bool = True   # notebook value; False = train on observed (noisy) curves
    var_ridge: float = 1e-6
    beta: float = 0.0
    kl_warmup: int = 0


def latent_var_penalty(H, ridge: float = 1e-6):
    """Mean squared one-step residual of a ridge VAR(1) fitted to H (differentiable)."""
    if H.shape[0] < 3:
        return torch.tensor(0.0)
    X = torch.cat([torch.ones(H.shape[0] - 1, 1), H[:-1]], dim=1); Y = H[1:]
    XtX = X.T @ X
    B = torch.linalg.solve(XtX + ridge * torch.eye(XtX.shape[0]), X.T @ Y)
    return torch.mean((Y - X @ B) ** 2)


def train_fae_phase(model_factory, Xn_train, Xc_train, tpts, Bp, Br, cfg: PhaseFAECfg):
    set_seed(cfg.seed)
    mse = nn.MSELoss()
    n = Xn_train.shape[0]
    n_tr = min(max(8, int(cfg.split_rate * n)), n - 1)
    target = Xc_train if cfg.clean_target else Xn_train
    Xn_tr, Xt_tr, Xn_va, Xt_va = Xn_train[:n_tr], target[:n_tr], Xn_train[n_tr:], target[n_tr:]
    model = model_factory()
    opt = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best_state, best_val = copy.deepcopy(model.state_dict()), float("inf")
    for ep in range(1, _epochs(cfg.epochs) + 1):
        model.train()
        beta_t = cfg.beta * (min(1.0, ep / cfg.kl_warmup) if cfg.kl_warmup else 1.0)
        for s in range(0, n_tr, cfg.batch_size):
            opt.zero_grad()
            xh, rep, _, coef = model(Xn_tr[s:s + cfg.batch_size], tpts, Bp, Br)
            loss = mse(xh, Xt_tr[s:s + cfg.batch_size]) + cfg.smooth_lambda * diff_penalty(coef)
            kl = _kl_term(model)
            if kl is not None and beta_t > 0:
                loss = loss + beta_t * kl
            loss.backward(); opt.step()
        if cfg.dyn_lambda > 0:   # full-batch dynamics step (notebook)
            opt.zero_grad()
            xh, H, _, coef = model(Xn_tr, tpts, Bp, Br)
            loss = mse(xh, Xt_tr) + cfg.smooth_lambda * diff_penalty(coef) + cfg.dyn_lambda * latent_var_penalty(H, cfg.var_ridge)
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            va_hat, H_va, _, coef_va = model(Xn_va, tpts, Bp, Br)
            va = mse(va_hat, Xt_va) + cfg.smooth_lambda * diff_penalty(coef_va)
            if cfg.dyn_lambda > 0:
                va = va + cfg.dyn_lambda * latent_var_penalty(H_va, cfg.var_ridge)
        if float(va) < best_val:
            best_val, best_state = float(va), copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state); model.eval()
    return model


# =====================================================================
# Metro: HAE / VHAE / functional-only FAE (port of the Metro notebook)
# =====================================================================
@dataclass
class MetroCfg:
    forecast_horizon: int = 7
    latent_dim: int = 5
    functional_feature_dim: int = 16
    n_basis: int = 16
    hidden_width: int = 128
    smooth_lambda: float = 2.0e-4
    latent_smooth_lambda: float = 1.5e-3
    nonfunctional_loss_weight: float = 0.70
    checkpoint_start_epoch: int = 50
    early_stopping_patience: int = 180
    early_stopping_metric: str = "hybrid_unweighted"
    epochs: int = 900
    batch_size: int = 16
    learning_rate: float = 6.5e-4
    weight_decay: float = 2.5e-5
    validation_fraction: float = 0.15
    early_stopping_min_delta: float = 1.0e-7
    gradient_clip_norm: float = 5.0
    shuffle_batches: bool = False
    max_ar_lags: int = 21
    seed: int = 776
    # VHAE only
    beta: float = 1e-3
    kl_warmup: int = 100
    logvar_init: float = -4.0


def build_hybrid(cfg: MetroCfg, n_grid, p, s, variational=False, n_meta=0):
    basis = bspline_basis_scipy(s, cfg.n_basis)
    cls = VariationalHybridAE if variational else HybridAE
    kw = dict(logvar_init=cfg.logvar_init) if variational else {}
    return cls(n_grid=n_grid, n_nonfunctional=p, latent_dim=cfg.latent_dim,
               functional_feature_dim=cfg.functional_feature_dim, hidden_width=cfg.hidden_width,
               n_basis=cfg.n_basis, basis=basis, n_metadata=n_meta, **kw)


def hybrid_batch_losses(model, xb, yb, cfg: MetroCfg, mb=None, beta_t: float = 0.0):
    mse = nn.MSELoss()
    x_hat, y_hat, h, coef = model(xb, yb, mb)
    f_rec, n_rec = mse(x_hat, xb), mse(y_hat, yb)
    cpen = diff_penalty(coef)
    h_s = model.last_mu if getattr(model, "variational", False) else h
    if h_s.shape[0] >= 3 and cfg.latent_smooth_lambda > 0:
        lpen = torch.mean((h_s[2:] - 2.0 * h_s[1:-1] + h_s[:-2]) ** 2)
    else:
        lpen = torch.tensor(0.0)
    total = f_rec + cfg.nonfunctional_loss_weight * n_rec + cfg.smooth_lambda * cpen + cfg.latent_smooth_lambda * lpen
    kl = _kl_term(model)
    kl_val = 0.0
    if kl is not None:
        kl_val = kl.item()
        if beta_t > 0:
            total = total + beta_t * kl
    return total, f_rec, n_rec, kl_val


def _es_score(f, n, cfg):
    if cfg.early_stopping_metric == "functional": return f
    if cfg.early_stopping_metric == "nonfunctional": return n
    return f + n


def train_hybrid(cfg: MetroCfg, x_full, y_full, s, variational=False, m_full=None, verbose=True):
    """Chronological train/validation split, AdamW, gradient clipping, best-validation
    checkpoint after `checkpoint_start_epoch`, early stopping (port of
    train_one_tuned_hybrid_model with refit_final_on_full_preforecast=False)."""
    set_seed(cfg.seed, single_thread=True)
    n = x_full.shape[0]
    n_val = min(max(1, int(round(cfg.validation_fraction * n))), n - 1)
    T = lambda a: torch.tensor(a, dtype=torch.float32)
    xtr, xva, ytr, yva = T(x_full[:-n_val]), T(x_full[-n_val:]), T(y_full[:-n_val]), T(y_full[-n_val:])
    has_m = m_full is not None and m_full.shape[1] > 0
    mtr, mva = (T(m_full[:-n_val]), T(m_full[-n_val:])) if has_m else (None, None)
    model = build_hybrid(cfg, x_full.shape[1], y_full.shape[1], s, variational, m_full.shape[1] if has_m else 0)
    ds = TensorDataset(xtr, ytr, mtr) if has_m else TensorDataset(xtr, ytr)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=cfg.shuffle_batches, drop_last=False)
    opt = optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    hist, best, best_state, best_ep, bad = [], np.inf, None, None, 0
    for ep in range(1, _epochs(cfg.epochs) + 1):
        model.train()
        beta_t = cfg.beta * (min(1.0, ep / cfg.kl_warmup) if cfg.kl_warmup else 1.0) if variational else 0.0
        sums, n_seen = np.zeros(4), 0
        for batch in loader:
            xb, yb = batch[0], batch[1]; mb = batch[2] if has_m else None
            opt.zero_grad(set_to_none=True)
            tot, f, nn_, kl = hybrid_batch_losses(model, xb, yb, cfg, mb, beta_t)
            tot.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.gradient_clip_norm)
            opt.step()
            bs = xb.shape[0]; sums += np.array([tot.item(), f.item(), nn_.item(), kl]) * bs; n_seen += bs
        # validation, batched exactly as in the notebook (affects the latent penalty only)
        model.eval(); vs, vn = np.zeros(3), 0
        with torch.no_grad():
            vds = TensorDataset(xva, yva, mva) if has_m else TensorDataset(xva, yva)
            for batch in DataLoader(vds, batch_size=cfg.batch_size, shuffle=False):
                xb, yb = batch[0], batch[1]; mb = batch[2] if has_m else None
                tot, f, nn_, _ = hybrid_batch_losses(model, xb, yb, cfg, mb, 0.0)
                bs = xb.shape[0]; vs += np.array([tot.item(), f.item(), nn_.item()]) * bs; vn += bs
        vs /= vn; tr = sums / n_seen
        score = _es_score(vs[1], vs[2], cfg)
        hist.append(dict(epoch=ep, train_total=tr[0], train_func=tr[1], train_nonfunc=tr[2], train_kl=tr[3],
                         val_total=vs[0], val_func=vs[1], val_nonfunc=vs[2], score=score))
        ok = ep >= cfg.checkpoint_start_epoch
        if ok and score < best - cfg.early_stopping_min_delta:
            best, best_state, best_ep, bad = score, copy.deepcopy(model.state_dict()), ep, 0
        elif ok:
            bad += 1
        if verbose and (ep == 1 or ep % 100 == 0):
            print(f"  epoch {ep:4d} | train_func={tr[1]:.4f} val_func={vs[1]:.4f} "
                  f"train_nonfunc={tr[2]:.4f} val_nonfunc={vs[2]:.4f}" + (f" kl={tr[3]:.3f}" if variational else ""))
        if ok and bad >= cfg.early_stopping_patience:
            if verbose: print(f"  early stop at {ep}; best epoch {best_ep}, score {best:.6f}")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    h = pd.DataFrame(hist); h.attrs["best_epoch"] = best_ep; h.attrs["best_score"] = best
    return model, h


def train_metro_fae(cfg: MetroCfg, x_full, s, verbose=True):
    """Port of train_fae_model (functional-only Metro baseline). The notebook calls it
    right after the HAE without reseeding, so pass the same global RNG state."""
    n = x_full.shape[0]
    n_val = max(1, min(int(round(cfg.validation_fraction * n)), n - 1))
    xtr = torch.tensor(x_full[:-n_val], dtype=torch.float32); xva = torch.tensor(x_full[-n_val:], dtype=torch.float32)
    model = MetroFAE(x_full.shape[1], cfg.latent_dim, cfg.functional_feature_dim, cfg.hidden_width,
                     cfg.n_basis, bspline_basis_scipy(s, cfg.n_basis))
    loader = DataLoader(TensorDataset(xtr), batch_size=cfg.batch_size, shuffle=cfg.shuffle_batches)
    opt = optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    mse = nn.MSELoss(); best, best_state, best_ep, bad = np.inf, None, None, 0
    for ep in range(1, _epochs(cfg.epochs) + 1):
        model.train()
        for (xb,) in loader:
            opt.zero_grad(set_to_none=True)
            x_hat, h, coef = model(xb)
            lpen = torch.mean((h[2:] - 2.0 * h[1:-1] + h[:-2]) ** 2) if (h.shape[0] >= 3 and cfg.latent_smooth_lambda > 0) else torch.tensor(0.0)
            loss = mse(x_hat, xb) + cfg.smooth_lambda * diff_penalty(coef) + cfg.latent_smooth_lambda * lpen
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.gradient_clip_norm); opt.step()
        model.eval()
        with torch.no_grad():
            v = mse(model(xva)[0], xva).item()
        ok = ep >= cfg.checkpoint_start_epoch
        if ok and v < best - cfg.early_stopping_min_delta:
            best, best_state, best_ep, bad = v, copy.deepcopy(model.state_dict()), ep, 0
        elif ok:
            bad += 1
        if ok and bad >= cfg.early_stopping_patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    if verbose: print(f"  FAE best epoch {best_ep}, val MSE {best:.6f}")
    model.eval()
    return model
