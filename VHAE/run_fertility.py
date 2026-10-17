"""VHAE paper, U.S. fertility application (functional-only special case, VFAE).

    python VHAE/run_fertility.py [--seeds 743 744 745 746 747] [--betas 1e-4 1e-3 1e-2]

Centering uses the training-year mean only (no look-ahead; the HAE notebook used
all years). Curves are divided by one global SD of the training curves (for FAE and VFAE alike),
so beta multiplies the KL against a reconstruction term on unit scale.
Outputs: VHAE/results/fertility/{results.csv, summary.csv, fertility_forecasts.pdf}
"""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
import torch
from hybridae.fertility import prepare_fertility, fit_fae_fertility, fae_var_forecast, fpca_var_forecast
from hybridae.train import FAETrainCfg
from hybridae.forecast import simulate_var1
from hybridae.metrics import rel_mse, prob_summary
from hybridae.plotting import plot_curve_forecasts

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=str(ROOT / "data" / "USAfertility.csv"))
ap.add_argument("--seeds", type=int, nargs="+", default=[743, 744, 745, 746, 747])
ap.add_argument("--betas", type=float, nargs="+", default=[1e-4, 1e-3, 1e-2])
ap.add_argument("--main-beta", type=float, default=1e-3)
ap.add_argument("--K", type=int, default=3); ap.add_argument("--paths", type=int, default=1000)
ap.add_argument("--out", default=str(ROOT / "VHAE" / "results" / "fertility"))
a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
D = prepare_fertility(a.data, center="train", standardize=True); Lh = len(D["years_te"]); sc = D["scale"]
xte = D["x_te"].numpy() * sc; xtr = D["x_tr"].numpy() * sc
rows = []; keep = {}


def probs(x_res_sd, H, b, A, resid, decode, rng, init=None):
    hT = np.repeat(H[-1][None], a.paths, 0) if init is None else init
    paths = simulate_var1(hT, b, A, resid, Lh, rng)
    xs = decode(paths.reshape(-1, H.shape[1])).reshape(a.paths, Lh, -1)
    return xs + x_res_sd[None, None] * rng.standard_normal(xs.shape)


for K in sorted({3, 5, a.K}):
    F = fpca_var_forecast(D, K); rng = np.random.default_rng(1)
    phi, mu0 = F["phi"].numpy(), F["mu"].numpy()
    xs = sc * probs((D["x_tr"].numpy() - F["x_recon"]).std(0), F["H"], F["b"], F["A"], F["resid"], lambda h: mu0 + h @ phi, rng)
    F["x_recon"] = F["x_recon"] * sc; F["x_fore"] = F["x_fore"] * sc
    rows.append(dict(method=f"FPCA(K={K})", seed=None, beta=None, recon=rel_mse(F["x_recon"], xtr),
                     fore=rel_mse(F["x_fore"], xte), aic_lag=F["aic_lag"]) | prob_summary(xs, xte))
    if K == a.K: keep["FPCA"] = (F["x_fore"], xs)
for sd in a.seeds:
    for name, beta in [("FAE", 0.0)] + [("VFAE", b) for b in a.betas]:
        cfg = FAETrainCfg(seed=sd, beta=beta, kl_warmup=1000 if beta else 0)
        model, Bp, Br = fit_fae_fertility(D, K=a.K, B=25, variational=beta > 0, cfg=cfg)
        R = fae_var_forecast(model, D, Bp, Br); rng = np.random.default_rng(sd)
        dec = lambda h: model.decode_from_rep(torch.tensor(h, dtype=torch.float32), Br)[0].detach().numpy()
        init, extra = None, {}
        if beta > 0:
            with torch.no_grad():
                mu, lv = model.posterior(model.project(D["x_tr"], D["tpts"], Bp))
            s_T = np.exp(0.5 * lv.numpy()[-1])
            init = mu.numpy()[-1][None] + s_T[None] * rng.standard_normal((a.paths, a.K))
            extra = dict(post_sd=float(np.exp(0.5 * lv.numpy()).mean()), latent_sd=float(R["H"].std(0).mean()))
        xs = sc * probs((D["x_tr"].numpy() - R["x_recon"]).std(0), R["H"], R["b"], R["A"], R["resid"], dec, rng, init)
        R["x_recon"] = R["x_recon"] * sc; R["x_fore"] = R["x_fore"] * sc
        r = dict(method=name, seed=sd, beta=beta, recon=rel_mse(R["x_recon"], xtr), fore=rel_mse(R["x_fore"], xte),
                 aic_lag=R["aic_lag"]) | prob_summary(xs, xte) | extra
        rows.append(r); print({k: (round(x, 4) if isinstance(x, float) else x) for k, x in r.items()}, flush=True)
        if sd == a.seeds[0] and (beta == 0 or beta == a.main_beta):
            keep["FAE" if beta == 0 else "VFAE"] = (R["x_fore"], xs)
df = pd.DataFrame(rows); df.to_csv(out / "results.csv", index=False)
g = df.assign(beta=df.beta.fillna(-1)).groupby(["method", "beta"]).agg(["mean", "std"]).drop(columns=["seed"], errors="ignore")
g.to_csv(out / "summary.csv"); print(g.to_string())
mu = D["mean"]; lo, hi = np.quantile(keep["VFAE"][1], 0.05, 0) + mu, np.quantile(keep["VFAE"][1], 0.95, 0) + mu
plot_curve_forecasts(D["ages"], xte + mu, {k: keep[k][0] + mu for k in ("FPCA", "FAE", "VFAE")}, D["years_te"],
                     out / "fertility_forecasts.pdf", "Age", "Fertility rate (per woman)", bands=dict(method="VFAE", lo=lo, hi=hi))
np.savez(out / "forecasts.npz", ages=D["ages"], years=D["years_te"], truth=xte + mu, VFAE_lo=lo, VFAE_hi=hi,
         **{k: keep[k][0] + mu for k in ("FPCA", "FAE", "VFAE")})
