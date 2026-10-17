"""Reproduce the U.S. fertility application of the HAE paper (FAE+VAR vs FPCA+VAR;
Figure 'fertility_forecasts').

    python HAE/run_fertility.py [--center all|train] [--seeds 743]

Default settings are those of 'FAE + VAR - nonlinear - USA.ipynb': 25 cubic
B-splines in and out, encoder 25-100-3, decoder 3-100-25 (ReLU, no bias),
N(0,0.5^2) init, Adam(3e-4, wd 1e-6), batch 16, lambda=1e-2, 5000 epochs,
random 85/15 split of the training years, ridge VAR(1) on the latents.
FPCA uses K=5 as in the notebook; K=3 (matched to FAE) is also reported.
"""
import argparse, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from hybridae.fertility import prepare_fertility, fit_fae_fertility, fae_var_forecast, fpca_var_forecast
from hybridae.metrics import rel_mse, rel_rmse
from hybridae.utils import save_json, environment_info
from hybridae.plotting import plot_curve_forecasts

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=str(ROOT / "data" / "USAfertility.csv"))
ap.add_argument("--center", default="all", choices=["all", "train"])
ap.add_argument("--seeds", type=int, nargs="+", default=[743])
ap.add_argument("--out", default=str(ROOT / "HAE" / "results" / "fertility"))
a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
D = prepare_fertility(a.data, center=a.center)
res = dict(center=a.center, env=environment_info(), FPCA={}, FAE=[])
for K in (5, 3):
    F = fpca_var_forecast(D, K)
    res["FPCA"][f"K{K}"] = dict(fore_relMSE=rel_mse(F["x_fore"], D["x_te"]), fore_relRMSE=rel_rmse(F["x_fore"], D["x_te"]),
                               recon_relMSE=rel_mse(F["x_recon"], D["x_tr"]), aic_lag=F["aic_lag"])
    print(f"FPCA K={K}: forecast relMSE={res['FPCA'][f'K{K}']['fore_relMSE']:.4f} relRMSE={res['FPCA'][f'K{K}']['fore_relRMSE']:.4f}")
    if K == 5: F5 = F
from hybridae.train import FAETrainCfg
for sd in a.seeds:
    model, Bp, Br = fit_fae_fertility(D, K=3, B=25, cfg=FAETrainCfg(seed=sd))
    R = fae_var_forecast(model, D, Bp, Br)
    r = dict(seed=sd, fore_relMSE=rel_mse(R["x_fore"], D["x_te"]), fore_relRMSE=rel_rmse(R["x_fore"], D["x_te"]),
             recon_relMSE=rel_mse(R["x_recon"], D["x_tr"]), aic_lag=R["aic_lag"])
    res["FAE"].append(r); print(f"FAE seed={sd}: forecast relMSE={r['fore_relMSE']:.4f} relRMSE={r['fore_relRMSE']:.4f} (AIC lag {r['aic_lag']})")
    if sd == a.seeds[0]: R0 = R
save_json(res, out / f"results_center-{a.center}.json")
mu = D["mean"]
plot_curve_forecasts(D["ages"], D["x_te"].numpy() + mu, {"FPCA": F5["x_fore"] + mu, "FAE": R0["x_fore"] + mu},
                     D["years_te"], out / f"fertility_forecasts_center-{a.center}.pdf", "Age", "Fertility rate per 1,000 women")
