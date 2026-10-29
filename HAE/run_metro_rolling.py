"""Rolling-origin evaluation of the Metro hybrid forecasts (HAE vs HPCA), per horizon and per
block of the nonfunctional component.

    python HAE/run_metro_rolling.py [--weeks 12] [--seed 776]

The single 7-day hold-out of Table 'traffic_hybrid' is one forecast origin. Here the HAE and
the HPCA loadings are estimated ONCE on the days before the first origin (scalers too), and
the latent VAR (AIC, q <= 21) is re-estimated at each of `weeks + 1` weekly origins on all
days observed up to that origin; the last origin is the paper's test week (2018-09-24..30).
Encoder/loadings stay fixed, so no information after an origin is used for its forecast.
A seasonal-naive reference (same weekday one week earlier) is included for context.
Standardized errors of rare variables are heavy-tailed (e.g. rain_1h_min is non-zero on only one of the 1,123
training days before the first origin, so its SD is ~0.007; named-holiday indicators), hence the figure shows
medians over origins with inter-quartile bands; means are in summary.json.

Outputs (HAE/results/metro_rolling/): rolling.csv (origin x horizon x method),
summary.json, metro_rolling_origin.pdf
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
import torch
from sklearn.decomposition import PCA
from hybridae.train import MetroCfg, train_hybrid
from hybridae.metro import prepare_metro, nonfunctional_blocks
from hybridae.forecast import forecast_scores_time_series
from hybridae.plotting import PAPER
from hybridae.utils import environment_info

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=str(ROOT / "data" / "metro_interstate_traffic_volume.zip"))
ap.add_argument("--weeks", type=int, default=12, help="weekly origins before the paper's test week")
ap.add_argument("--seed", type=int, default=776)
ap.add_argument("--out", default=str(ROOT / "HAE" / "results" / "metro_rolling"))
a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
t0 = time.time(); h = 7; n_or = a.weeks + 1
cfg = MetroCfg(seed=a.seed)
D = prepare_metro(a.data, horizon=h * n_or)          # scalers from the days before the first origin
x_tr, y_tr, x_ev, y_ev = D["x_tr"], D["y_tr"], D["x_te"], D["y_te"]
dates_ev = D["dates"][-h * n_or:]; p = y_tr.shape[1]
blocks = nonfunctional_blocks(D["names"])
print(f"training days {x_tr.shape[0]} (to {D['dates'][-h*n_or-1].date()}), origins {n_or}, "
      f"evaluation {dates_ev[0].date()}..{dates_ev[-1].date()}", flush=True)

hae, hist = train_hybrid(cfg, x_tr, y_tr, D["s"], verbose=False)
pca = PCA(n_components=cfg.latent_dim).fit(np.concatenate([x_tr, y_tr], 1))
print(f"HAE trained (best epoch {hist.attrs['best_epoch']}), {time.time()-t0:.0f}s", flush=True)

rows, lags = [], []
for k in range(n_or):
    xo = np.concatenate([x_tr, x_ev[:h * k]]); yo = np.concatenate([y_tr, y_ev[:h * k]])
    xt, yt = x_ev[h * k:h * (k + 1)], y_ev[h * k:h * (k + 1)]
    with torch.no_grad():
        H = hae.encode(torch.tensor(xo), torch.tensor(yo)).numpy()
    Hf, info_h, _ = forecast_scores_time_series(H, h, cfg.max_ar_lags)
    with torch.no_grad():
        xf_h, _, yf_h = hae.decode_latent(torch.tensor(Hf, dtype=torch.float32))
    S = pca.transform(np.concatenate([xo, yo], 1))
    Sf, info_p, _ = forecast_scores_time_series(S, h, cfg.max_ar_lags)
    cf = pca.inverse_transform(Sf)
    preds = {"HAE": (xf_h.numpy(), yf_h.numpy()), "HPCA": (cf[:, :x_tr.shape[1]], cf[:, x_tr.shape[1]:]),
             "Seasonal naive": (xo[-h:], yo[-h:])}
    lags.append(dict(origin=str(dates_ev[h * k].date()), HAE=info_h["lag"], HPCA=info_p["lag"]))
    for m, (xf, yf) in preds.items():
        ex, ey = (xf - xt) ** 2, (yf - yt) ** 2
        for i in range(h):
            r = dict(origin=str(dates_ev[h * k].date()), origin_index=k, horizon=i + 1, method=m,
                     func=float(ex[i].mean()), nonfunc=float(ey[i].mean()))
            for b in ("weather", "calendar", "holiday"):
                r[f"contrib_{b}"] = float(ey[i, blocks[b]].sum() / p)
            rows.append(r)
R = pd.DataFrame(rows); R.to_csv(out / "rolling.csv", index=False)

metrics = [("func", "Functional MSE"), ("nonfunc", "Nonfunctional MSE"),
           ("contrib_weather", "Weather contribution"), ("contrib_calendar", "Calendar + holiday contribution")]
R["contrib_calendar"] = R["contrib_calendar"] + R["contrib_holiday"]
per_origin = R.groupby(["method", "origin_index"])[[m for m, _ in metrics]].mean()
summ = {}
for m in ("HAE", "HPCA", "Seasonal naive"):
    po = per_origin.loc[m]
    summ[m] = {k: dict(mean=float(po[k].mean()), se=float(po[k].std(ddof=1) / np.sqrt(len(po))),
                       median=float(po[k].median())) for k, _ in metrics}
win = {k: int((per_origin.loc["HAE"][k] < per_origin.loc["HPCA"][k]).sum()) for k, _ in metrics}
summary = dict(n_origins=n_or, training_days=int(x_tr.shape[0]), seed=a.seed, by_method=summ,
               HAE_better_than_HPCA_origins=win, var_lags=lags, env=environment_info(), runtime_sec=time.time() - t0)
(out / "summary.json").write_text(json.dumps(summary, indent=2))

STY = {"HAE": PAPER["HAE"], "HPCA": PAPER["HPCA"], "Seasonal naive": dict(color="0.45", lw=1.3, ls=":")}
fig, axes = plt.subplots(1, 4, figsize=(12.5, 3.1))
for ax, (k, lab) in zip(axes, metrics):
    for m in ("HPCA", "HAE", "Seasonal naive"):
        g = R[R.method == m].groupby("horizon")[k]; mu, lo, hi = g.median(), g.quantile(0.25), g.quantile(0.75)
        st = {kk: v for kk, v in STY[m].items() if kk not in ("label", "marker", "ms")}
        ax.plot(mu.index, mu.values, marker="o", ms=3, label=m, **st)
        ax.fill_between(mu.index, lo, hi, color=st["color"], alpha=0.13, lw=0)
    ax.set_title(lab, fontsize=9.5); ax.set_xticks(range(1, h + 1)); ax.tick_params(labelsize=8)
    ax.set_xlabel("Forecast horizon (days)", fontsize=9); ax.grid(color="0.93", lw=0.6)
    ax.set_ylim(bottom=0)
axes[0].set_ylabel(f"Median over {n_or} weekly origins (IQR band)", fontsize=8.5)
h_, l_ = axes[0].get_legend_handles_labels()
fig.legend(h_, l_, loc="upper center", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 1.07))
fig.tight_layout(); fig.savefig(out / "metro_rolling_origin.pdf", bbox_inches="tight"); plt.close(fig)

for m in summ:
    print(f"{m:15s} " + "  ".join(f"{k}: mean {summ[m][k]['mean']:.4f} median {summ[m][k]['median']:.4f}" for k, _ in metrics))
print("origins where HAE < HPCA:", win, f"of {n_or}")
print(f"done in {time.time()-t0:.0f}s -> {out}")
