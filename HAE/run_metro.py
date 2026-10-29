"""Metro Interstate Traffic Volume application of the HAE paper (Section 4.2):
Table 'traffic_hybrid', the weather-only contributions reported after Figure
'metro_hae_hpca', Figure 'metro_hae_hpca' (hae_vs_hpca.pdf), and the per-variable
nonfunctional output used by HAE/make_nonfunctional_figures.py.

    python HAE/run_metro.py [--seed 776] [--optimizer adamw|adam] [--functional-only]

Settings (paper, Sec. 4.2; see HAE/README.md for the details the paper does not state):
1,214 complete days, last 7 (2018-09-24..30) held out; log1p + standardization of the 24
hourly values and of the 49 nonfunctional variables with pre-forecast statistics;
16 fixed cubic B-spline features; encoder 65-128-128-5, decoder 5-128-128-65 (GELU +
LayerNorm, linear outputs); lr 6.5e-4, weight decay 2.5e-5, batch 16, <= 900 epochs,
last 15% of the pre-forecast days for chronological validation; AIC-selected VAR(q) on the
five latent coordinates; HPCA = PCA (5 components) of the concatenated standardized data
with the same VAR step.

--functional-only also fits the functional-only FAE / FPCA pair of the original notebook
(no longer reported in the paper).
Outputs (HAE/results/metro/ by default): results.json, forecasts.npz, nonfunctional.npz,
hae_vs_hpca.pdf.
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
from hybridae.plotting import plot_paper_panels

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=str(ROOT / "data" / "metro_interstate_traffic_volume.zip"))
ap.add_argument("--seed", type=int, default=776)
ap.add_argument("--optimizer", default="adamw", choices=["adamw", "adam"])
ap.add_argument("--functional-only", action="store_true", help="also fit the functional-only FAE/FPCA pair")
ap.add_argument("--no-plot", action="store_true")
ap.add_argument("--out", default=str(ROOT / "HAE" / "results" / "metro"))
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
cfg = MetroCfg(seed=a.seed, optimizer=a.optimizer); t0 = time.time()
D = prepare_metro(a.data, cfg.forecast_horizon)
print(f"days={D['x_counts'].shape[0]}, pre-forecast={D['x_tr'].shape[0]}, p={D['y_tr'].shape[1]}, "
      f"test={D['dates'][-D['L']].date()}..{D['dates'][-1].date()}", flush=True)

print("[1] HAE + VAR", flush=True); hae, hist = train_hybrid(cfg, D["x_tr"], D["y_tr"], D["s"])
H = hae_recon_and_forecast(hae, D, cfg); print("   ", H["info"])
print("[2] HPCA + VAR"); P = hpca_forecast(D["x_tr"], D["y_tr"], D["L"], cfg.latent_dim, cfg.max_ar_lags); print("   ", P["forecast_info"])

res = dict(
    HAE=point_table(D, H["x_recon"], H["y_recon"], H["x_fore"], H["y_fore"]) | dict(var=H["info"], best_epoch=hist.attrs["best_epoch"]),
    HPCA=point_table(D, P["x_recon"], P["y_recon"], P["x_forecast"], P["y_forecast"]) | dict(var=P["forecast_info"],
         explained_var=float(np.sum(P["pca"].explained_variance_ratio_))),
    seed=a.seed, optimizer=a.optimizer)

if a.functional_only:
    print("[3] FAE + VAR (functional only; no reseed, as in the notebook)", flush=True)
    fae = train_metro_fae(cfg, D["x_tr"], D["s"])
    with torch.no_grad():
        xr, Hf, _ = fae(torch.tensor(D["x_tr"]))
    Hff, finfo, _ = forecast_scores_time_series(Hf.numpy(), D["L"], cfg.max_ar_lags)
    with torch.no_grad():
        xf_fae = fae.decode_latent(torch.tensor(Hff, dtype=torch.float32))[0].numpy()
    print("[4] FPCA + VAR (functional only)"); F = functional_pca_forecast(D["x_tr"], D["L"], cfg.latent_dim, cfg.max_ar_lags)
    res["FAE"] = dict(recon_func=mse(xr.numpy(), D["x_tr"]), fore_func=mse(xf_fae, D["x_te"]),
                      fore_func_counts=mse(to_counts(xf_fae, D["x_scaler"]), D["x_te_counts"]), var=finfo)
    res["FPCA"] = dict(recon_func=mse(F["x_recon"], D["x_tr"]), fore_func=mse(F["x_forecast"], D["x_te"]),
                       fore_func_counts=mse(to_counts(F["x_forecast"], D["x_scaler"]), D["x_te_counts"]), var=F["forecast_info"])
res |= dict(runtime_sec=time.time() - t0, env=environment_info())

red = lambda b, h: 100 * (b - h) / b
print("\nTable 'traffic_hybrid': reconstruction / forecasting MSE (standardized scale)")
for comp in ["func", "nonfunc", "hybrid"]:
    for kind in ["recon", "fore"]:
        b, h = res["HPCA"][f"{kind}_{comp}"], res["HAE"][f"{kind}_{comp}"]
        print(f"  {kind:5s} {comp:8s} HPCA={b:.4f} HAE={h:.4f} reduction={red(b,h):6.2f}%")
for kind in ["recon", "fore"]:
    print(f"  weather contribution ({kind}): HPCA={res['HPCA'][f'contrib_weather_{kind}']:.4f} "
          f"HAE={res['HAE'][f'contrib_weather_{kind}']:.4f}")
if a.functional_only:
    for k in ["recon_func", "fore_func", "fore_func_counts"]:
        print(f"  FAE vs FPCA {k:16s}: {res['FPCA'][k]:.4f} -> {res['FAE'][k]:.4f} ({red(res['FPCA'][k], res['FAE'][k]):.2f}%)")
save_json(res, out / "results.json")

dates_te = D["dates"][-D["L"]:]
fc = dict(truth=D["x_te_counts"], HAE=to_counts(H["x_fore"], D["x_scaler"]),
          HPCA=to_counts(P["x_forecast"], D["x_scaler"]), dates=np.array([str(d.date()) for d in dates_te]))
if a.functional_only:
    fc |= dict(FAE=to_counts(xf_fae, D["x_scaler"]), FPCA=to_counts(F["x_forecast"], D["x_scaler"]))
np.savez(out / "forecasts.npz", **fc)
# everything needed for the nonfunctional diagnostics (standardized scale + scaler to undo it)
np.savez(out / "nonfunctional.npz", names=np.array(D["names"]),
         dates_tr=np.array([str(d.date()) for d in D["dates"][:-D["L"]]]), dates_te=fc["dates"],
         y_tr=D["y_tr"], y_te=D["y_te"], y_raw=D["y_raw"],
         HAE_recon=H["y_recon"], HAE_fore=H["y_fore"], HPCA_recon=P["y_recon"], HPCA_fore=P["y_forecast"],
         HAE_latent=H["H"], HAE_latent_fore=H["H_fore"], HPCA_latent=P["scores_train"], HPCA_latent_fore=P["scores_forecast"],
         y_mean=D["y_scaler"].mean_, y_scale=D["y_scaler"].scale_)
if not a.no_plot:
    titles = [d.strftime("%a, %b %d").replace(" 0", " ") for d in dates_te]
    plot_paper_panels(np.arange(24), fc["truth"], {"HPCA": fc["HPCA"], "HAE": fc["HAE"]}, titles,
                      out / "hae_vs_hpca.pdf", "Hour of day", "Traffic volume", xticks=[0, 4, 8, 12, 16, 20, 23])
print(f"done in {time.time()-t0:.0f}s -> {out}")
