"""Reproduce the Metro Interstate Traffic Volume results of the HAE paper
(Table 'traffic_hybrid', the FAE-vs-FPCA and weather-only comparisons, and
Figure 'hae_vs_hpca').

    python HAE/run_metro.py [--data data/metro_interstate_traffic_volume.zip] [--seed 776]

Outputs go to HAE/results/metro/ (results.json, forecasts.npz, hae_vs_hpca.pdf).
"""
import argparse, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
import torch
from hybridae.train import MetroCfg, train_hybrid, train_metro_fae
from hybridae.metro import prepare_metro, hae_recon_and_forecast, point_table, to_counts
from hybridae.baselines import hpca_forecast, functional_pca_forecast
from hybridae.forecast import forecast_scores_time_series
from hybridae.metrics import mse
from hybridae.utils import save_json, environment_info
from hybridae.plotting import plot_metro_forecasts

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=str(ROOT / "data" / "metro_interstate_traffic_volume.zip"))
ap.add_argument("--seed", type=int, default=776)
ap.add_argument("--out", default=str(ROOT / "HAE" / "results" / "metro"))
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
cfg = MetroCfg(seed=a.seed); t0 = time.time()
D = prepare_metro(a.data, cfg.forecast_horizon)
print(f"days={D['x_counts'].shape[0]}, pre-forecast={D['x_tr'].shape[0]}, p={D['y_tr'].shape[1]}, "
      f"test={D['dates'][-D['L']].date()}..{D['dates'][-1].date()}", flush=True)

print("[1] HAE + VAR", flush=True); hae, hist = train_hybrid(cfg, D["x_tr"], D["y_tr"], D["s"])
H = hae_recon_and_forecast(hae, D, cfg); print("   ", H["info"])
print("[2] HPCA + VAR"); P = hpca_forecast(D["x_tr"], D["y_tr"], D["L"], cfg.latent_dim, cfg.max_ar_lags); print("   ", P["forecast_info"])
print("[3] FAE + VAR (functional only; no reseed, as in the notebook)", flush=True)
fae = train_metro_fae(cfg, D["x_tr"], D["s"])
with torch.no_grad():
    xr, Hf, _ = fae(torch.tensor(D["x_tr"]))
Hff, finfo, _ = forecast_scores_time_series(Hf.numpy(), D["L"], cfg.max_ar_lags)
with torch.no_grad():
    xf_fae = fae.decode_latent(torch.tensor(Hff, dtype=torch.float32))[0].numpy()
print("    ", finfo)
print("[4] FPCA + VAR (functional only)"); F = functional_pca_forecast(D["x_tr"], D["L"], cfg.latent_dim, cfg.max_ar_lags); print("   ", F["forecast_info"])

res = dict(
    HAE=point_table(D, H["x_recon"], H["y_recon"], H["x_fore"], H["y_fore"]) | dict(var=H["info"], best_epoch=hist.attrs["best_epoch"]),
    HPCA=point_table(D, P["x_recon"], P["y_recon"], P["x_forecast"], P["y_forecast"]) | dict(var=P["forecast_info"],
         explained_var=float(np.sum(P["pca"].explained_variance_ratio_))),
    FAE=dict(recon_func=mse(xr.numpy(), D["x_tr"]), fore_func=mse(xf_fae, D["x_te"]),
             fore_func_counts=mse(to_counts(xf_fae, D["x_scaler"]), D["x_te_counts"]), var=finfo),
    FPCA=dict(recon_func=mse(F["x_recon"], D["x_tr"]), fore_func=mse(F["x_forecast"], D["x_te"]),
              fore_func_counts=mse(to_counts(F["x_forecast"], D["x_scaler"]), D["x_te_counts"]), var=F["forecast_info"]),
    seed=a.seed, runtime_sec=time.time() - t0, env=environment_info())
red = lambda b, h: 100 * (b - h) / b
print("\nTable (HAE paper, Metro): reconstruction / forecasting MSE")
for comp in ["func", "nonfunc", "hybrid"]:
    for kind in ["recon", "fore"]:
        b, h = res["HPCA"][f"{kind}_{comp}"], res["HAE"][f"{kind}_{comp}"]
        print(f"  {kind:5s} {comp:8s} HPCA={b:.4f} HAE={h:.4f} reduction={red(b,h):6.2f}%")
for k in ["recon_func", "fore_func", "fore_func_counts"]:
    print(f"  FAE vs FPCA {k:16s}: {res['FPCA'][k]:.4f} -> {res['FAE'][k]:.4f} ({red(res['FPCA'][k], res['FAE'][k]):.2f}%)")
print(f"  weather (29 vars) recon HPCA={res['HPCA']['recon_weather']:.4f} HAE={res['HAE']['recon_weather']:.4f}; "
      f"fore HPCA={res['HPCA']['fore_weather']:.4f} HAE={res['HAE']['fore_weather']:.4f}")
save_json(res, out / "results.json")
np.savez(out / "forecasts.npz", truth=D["x_te_counts"], HAE=to_counts(H["x_fore"], D["x_scaler"]),
         HPCA=to_counts(P["x_forecast"], D["x_scaler"]), FAE=to_counts(xf_fae, D["x_scaler"]),
         FPCA=to_counts(F["x_forecast"], D["x_scaler"]), dates=np.array([str(d.date()) for d in D["dates"][-D["L"]:]]))
plot_metro_forecasts(out / "forecasts.npz", ["HPCA", "HAE"], out / "hae_vs_hpca.pdf")
print(f"done in {time.time()-t0:.0f}s -> {out}")
