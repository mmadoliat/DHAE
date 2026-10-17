"""VHAE paper, Metro application: HPCA+VAR, HAE+VAR and VHAE+VAR point and
probabilistic forecasts of the hybrid daily observation (7-day horizon).

    python VHAE/run_metro.py [--seeds 776 777 778] [--beta 1e-3] [--paths 1000]

Probabilistic forecasts (all methods): Monte-Carlo latent paths from the fitted
VAR(q) with Gaussian innovations, decoded, plus Gaussian observation noise with
per-coordinate variance estimated from pre-forecast reconstruction residuals.
VHAE additionally draws the last q latent states from the encoder posterior.
Outputs: VHAE/results/metro/{results.csv, summary.json, forecasts.npz, vhae_metro_forecasts.pdf}
"""
import argparse, sys, time
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
import torch
from hybridae.train import MetroCfg, train_hybrid
from hybridae.metro import prepare_metro, hae_recon_and_forecast, point_table, to_counts
from hybridae.baselines import hpca_forecast
from hybridae.forecast import simulate_var_paths
from hybridae.metrics import prob_summary
from hybridae.utils import save_json, environment_info
from hybridae.plotting import plot_metro_forecasts

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=str(ROOT / "data" / "metro_interstate_traffic_volume.zip"))
ap.add_argument("--seeds", type=int, nargs="+", default=[776, 777, 778])
ap.add_argument("--beta", type=float, default=1e-3)
ap.add_argument("--paths", type=int, default=1000)
ap.add_argument("--methods", nargs="+", default=["HPCA", "HAE", "VHAE"])
ap.add_argument("--tag", default="")
ap.add_argument("--out", default=str(ROOT / "VHAE" / "results" / "metro"))
a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
D = prepare_metro(a.data, 7); L, w = D["L"], D["weather_idx"]
rows, keep = [], {}


def prob_rows(xs, ys):
    r = {f"func_{k}": v for k, v in prob_summary(xs, D["x_te"]).items()}
    r |= {f"weather_{k}": v for k, v in prob_summary(ys[..., w], D["y_te"][:, w]).items()}
    return r


def decode_paths(model, paths):
    S = paths.shape[0]
    with torch.no_grad():
        xs, _, ys = model.decode_latent(torch.tensor(paths.reshape(S * L, -1), dtype=torch.float32))
    return xs.numpy().reshape(S, L, -1), ys.numpy().reshape(S, L, -1)


def add_noise(xs, ys, xr, yr, rng):
    sx, sy = (D["x_tr"] - xr).std(0), (D["y_tr"] - yr).std(0)
    return xs + sx * rng.standard_normal(xs.shape), ys + sy * rng.standard_normal(ys.shape)


if "HPCA" in a.methods:
    P = hpca_forecast(D["x_tr"], D["y_tr"], L, 5, 21, return_model=True)
    rng = np.random.default_rng(0)
    sp = simulate_var_paths(P["var_result"], P["score_scaler"], P["scores_train"], L, a.paths, rng)
    comb = P["pca"].inverse_transform(sp.reshape(-1, 5)).reshape(a.paths, L, -1)
    xs, ys = add_noise(comb[..., :24], comb[..., 24:], P["x_recon"], P["y_recon"], rng)
    r = dict(method="HPCA", seed=None, lag=P["forecast_info"]["lag"]) | point_table(D, P["x_recon"], P["y_recon"], P["x_forecast"], P["y_forecast"]) | prob_rows(xs, ys)
    rows.append(r); keep["HPCA"] = (P["x_forecast"], xs)
for sd in a.seeds:
    for m in [m for m in a.methods if m in ("HAE", "VHAE")]:
        t0 = time.time(); var = m == "VHAE"
        cfg = MetroCfg(seed=sd, beta=a.beta)
        model, hist = train_hybrid(cfg, D["x_tr"], D["y_tr"], D["s"], variational=var, verbose=False)
        R = hae_recon_and_forecast(model, D, cfg)
        rng = np.random.default_rng(sd); p = R["var_result"].k_ar
        init = None
        if var:
            with torch.no_grad():
                mu, lv = model.posterior(torch.tensor(D["x_tr"]), torch.tensor(D["y_tr"]))
            mu, sdv = mu.numpy()[-p:], np.exp(0.5 * lv.numpy()[-p:])
            init = mu[None] + sdv[None] * rng.standard_normal((a.paths, p, mu.shape[1]))
        paths = simulate_var_paths(R["var_result"], R["scaler"], R["H"], L, a.paths, rng, init_raw=init)
        xs, ys = decode_paths(model, paths)
        xs, ys = add_noise(xs, ys, R["x_recon"], R["y_recon"], rng)
        r = dict(method=m, seed=sd, lag=R["info"]["lag"], best_epoch=hist.attrs["best_epoch"]) | \
            point_table(D, R["x_recon"], R["y_recon"], R["x_fore"], R["y_fore"]) | prob_rows(xs, ys)
        if var:
            r |= dict(post_sd_mean=float(sdv.mean()), latent_sd=float(R["H"].std(0).mean()), beta=a.beta)
        rows.append(r); print(m, sd, {k: round(v, 4) for k, v in r.items() if isinstance(v, float)}, f"{time.time()-t0:.0f}s", flush=True)
        if sd == a.seeds[0]:
            keep[m] = (R["x_fore"], xs)
df = pd.DataFrame(rows); df.to_csv(out / f"results{a.tag}.csv", index=False)
num = df.select_dtypes("number").columns.drop(["seed"], errors="ignore")
summ = df.groupby("method")[num].agg(["mean", "std"]); print(summ.T.to_string())
summ.to_csv(out / f"summary{a.tag}.csv")
save_json(dict(env=environment_info(), args=vars(a)), out / f"run_info{a.tag}.json")
if all(k in keep for k in ("HPCA", "HAE", "VHAE")):
    xs_c = to_counts(keep["VHAE"][1].reshape(-1, 24), D["x_scaler"]).reshape(keep["VHAE"][1].shape)
    lo, hi = np.quantile(xs_c, 0.05, 0), np.quantile(xs_c, 0.95, 0)
    np.savez(out / f"forecasts{a.tag}.npz", truth=D["x_te_counts"], dates=np.array([str(d.date()) for d in D["dates"][-L:]]),
             **{k: to_counts(v[0], D["x_scaler"]) for k, v in keep.items()}, VHAE_lo=lo, VHAE_hi=hi)
    plot_metro_forecasts(out / f"forecasts{a.tag}.npz", ["HPCA", "HAE", "VHAE"], out / f"vhae_metro_forecasts{a.tag}.pdf",
                         bands=dict(method="VHAE", lo=lo, hi=hi))
