"""U.S. fertility application of the HAE paper (Section 4.1; Figure 'fertility_forecasts').

    python HAE/run_fertility.py                        # settings as written in the paper
    python HAE/run_fertility.py --legacy-notebook      # 'FAE + VAR - nonlinear - USA.ipynb' exactly

Paper settings (Sec. 4.1): curves X_t(s) on the observed age grid (13..54), 1933-2018 for
training and 2019-2023 for evaluation; 25 cubic B-splines in the input and output layers;
encoder 25-100-3 and decoder 3-100-25 with ReLU hidden layers and linear latent/output layers
(with bias terms, as in the encoder equation of Sec. 2.1); second-difference penalty
lambda = 0.01; Adam (lr 3e-4, weight decay 1e-6), batch 16, 5000 epochs; N(0, 0.5^2) weight
initialization, seed 743; random 85/15 split of the training years (training/validation);
VAR(1) on the latent coordinates and on the FPCA scores; relMSE over the held-out curves.

Not stated in the paper (defaults here, all switchable): no pre-smoothing (--presmooth S
applies UnivariateSpline(s=S) per year, as the notebook did with S=50); centring by the
training-year mean curve (--center all = the notebook's mean over all years); FPCA with
K = 3 (matched to the FAE) and K = 5 (notebook).  The GCV curve of the basis dimension and
the AIC/BIC-selected VAR orders are reported as diagnostics.
Outputs: HAE/results/fertility/results_<tag>.json and fertility_forecasts_<tag>.pdf.
"""
import argparse, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from hybridae.fertility import (prepare_fertility, fit_fae_fertility, fae_var_forecast, fpca_var_forecast,
                                var_order_table, gcv_basis_dimension)
from hybridae.train import FAETrainCfg
from hybridae.metrics import rel_mse, rel_rmse
from hybridae.utils import save_json, environment_info
from hybridae.plotting import plot_paper_panels

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=str(ROOT / "data" / "USAfertility.csv"))
ap.add_argument("--legacy-notebook", action="store_true", help="s=50 pre-smoothing, all-year centring, no biases")
ap.add_argument("--presmooth", type=float, default=None, help="UnivariateSpline smoothing factor per year (default: none)")
ap.add_argument("--center", default="train", choices=["all", "train"])
ap.add_argument("--no-bias", action="store_true")
ap.add_argument("--n-basis", type=int, default=25)
ap.add_argument("--seeds", type=int, nargs="+", default=[743])
ap.add_argument("--fpca-K", type=int, nargs="+", default=[3, 5])
ap.add_argument("--out", default=str(ROOT / "HAE" / "results" / "fertility"))
a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
if a.legacy_notebook:
    a.presmooth, a.center, a.no_bias = 50.0, "all", True
tag = "legacy-notebook" if a.legacy_notebook else f"paper_center-{a.center}" + (f"_s{a.presmooth:g}" if a.presmooth else "")
D = prepare_fertility(a.data, center=a.center, smooth_s=a.presmooth)
mu = D["mean"]
settings = dict(presmooth=a.presmooth, center=a.center, bias=not a.no_bias, n_basis=a.n_basis, K_fae=3,
                fpca_K=a.fpca_K, var="VAR(1), ridge 1e-8", seeds=a.seeds)
res = dict(settings=settings, env=environment_info(), FPCA={}, FAE=[])

# diagnostics: GCV for the basis dimension (training curves, original scale)
M_gcv, gcv = gcv_basis_dimension(D["smoothed"][D["years"] <= 2018], D["tpts"])
res["gcv"] = dict(argmin=M_gcv, curve=gcv)
print(f"GCV-selected number of cubic B-splines: {M_gcv} (paper uses {a.n_basis}); GCV({a.n_basis})/GCV(min) = "
      f"{gcv[a.n_basis] / gcv[M_gcv]:.3f}")


def scores(xf, xr):
    xte, xtr = D["x_te"].numpy(), D["x_tr"].numpy()
    return dict(fore_relMSE=rel_mse(xf, xte), fore_relRMSE=rel_rmse(xf, xte),
                fore_relMSE_original_scale=rel_mse(xf + mu, xte + mu),
                recon_relMSE=rel_mse(xr, xtr))


for K in a.fpca_K:
    F = fpca_var_forecast(D, K)
    res["FPCA"][f"K{K}"] = scores(F["x_fore"], F["x_recon"]) | dict(var_orders=var_order_table(F["H"]))
    print(f"FPCA K={K}: forecast relMSE={res['FPCA'][f'K{K}']['fore_relMSE']:.4f} "
          f"(original scale {res['FPCA'][f'K{K}']['fore_relMSE_original_scale']:.4f}); VAR orders {res['FPCA'][f'K{K}']['var_orders']}")
    if K == a.fpca_K[0]:
        F0 = F
for sd in a.seeds:
    model, Bp, Br = fit_fae_fertility(D, K=3, B=a.n_basis, cfg=FAETrainCfg(seed=sd), bias=not a.no_bias)
    R = fae_var_forecast(model, D, Bp, Br)
    r = dict(seed=sd) | scores(R["x_fore"], R["x_recon"]) | dict(var_orders=var_order_table(R["H"]))
    res["FAE"].append(r)
    print(f"FAE seed={sd}: forecast relMSE={r['fore_relMSE']:.4f} (original scale {r['fore_relMSE_original_scale']:.4f}); "
          f"VAR orders {r['var_orders']}")
    if sd == a.seeds[0]:
        R0 = R
if len(a.seeds) > 1:
    v = np.array([r["fore_relMSE"] for r in res["FAE"]])
    res["FAE_summary"] = dict(mean=float(v.mean()), sd=float(v.std(ddof=1)), n=len(v))
save_json(res, out / f"results_{tag}.json")
plot_paper_panels(D["ages"], D["x_te"].numpy() + mu, {"FPCA": F0["x_fore"] + mu, "FAE": R0["x_fore"] + mu},
                  D["years_te"], out / f"fertility_forecasts_{tag}.pdf", "Age", "Fertility rate")
print(f"-> {out}")
