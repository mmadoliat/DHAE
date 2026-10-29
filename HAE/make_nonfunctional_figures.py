"""Diagnostics of the NONFUNCTIONAL component of the Metro hybrid data (HAE vs HPCA).

    python HAE/run_metro.py                       # writes HAE/results/metro/nonfunctional.npz
    python HAE/make_nonfunctional_figures.py      # [--dir HAE/results/metro]
    python HAE/make_nonfunctional_figures.py --seed-dirs HAE/results/metro_seeds/seed*   # + seed variability

With --seed-dirs (further run_metro.py outputs with other training seeds), the HAE entries are
summarized over all runs: per-variable R^2 = median over seeds with the seed range as a bar,
table entries = mean (SD) over seeds. HPCA is deterministic. The trajectory figure always shows
the run in --dir.

Outputs (same directory):
  metro_nonfunctional_r2.pdf          per-variable reconstruction R^2 = 1 - MSE_j / Var_j of the
                                      non-constant nonfunctional variables, grouped by block
  metro_nonfunctional_trajectories.pdf  selected variables in original units: observed values,
                                      reconstructions over the last 5 pre-forecast weeks and
                                      forecasts over the held-out week
  table_nonfunctional.tex             block decomposition of the nonfunctional MSE (contributions
                                      add up to the totals of Table 'traffic_hybrid') and recovery
                                      of the categorical variables from the reconstructions
  nonfunctional_summary.json          all numbers above
All errors are on the standardized scale used for training (pre-forecast mean / SD), so the
training-period variance of every non-constant variable is 1 and R^2_j = 1 - MSE_j.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from hybridae.metro import nonfunctional_blocks
from hybridae.plotting import PAPER

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default=str(ROOT / "HAE" / "results" / "metro"))
ap.add_argument("--seed-dirs", nargs="*", default=[])
a = ap.parse_args(); d = Path(a.dir)
z = np.load(d / "nonfunctional.npz", allow_pickle=True)
Z_seeds = [z] + [np.load(Path(sd) / "nonfunctional.npz", allow_pickle=True) for sd in a.seed_dirs]
S = len(Z_seeds)
names = [str(n) for n in z["names"]]; p = len(names)
ytr, yte, yraw = z["y_tr"], z["y_te"], z["y_raw"]; ntr = ytr.shape[0]
mean, scale = z["y_mean"], z["y_scale"]
orig = lambda v: np.asarray(v) * scale + mean
M = ("HPCA", "HAE")
rec = {m: z[f"{m}_recon"] for m in M}; fore = {m: z[f"{m}_fore"] for m in M}
blocks = nonfunctional_blocks(names)
const = [j for j in range(p) if np.ptp(yraw[:ntr, j]) == 0]

# ------------------------------------------------------------------ per-variable numbers
pv = lambda a_, b_: np.mean((np.asarray(a_, float) - np.asarray(b_, float)) ** 2, axis=0)
var_tr = ytr.var(0)
tab = pd.DataFrame(dict(variable=names, constant=[j in const for j in range(p)]))
for m in M:
    tab[f"recon_mse_{m}"] = pv(rec[m], ytr); tab[f"fore_mse_{m}"] = pv(fore[m], yte)
    tab[f"R2_{m}"] = np.where(var_tr > 0, 1 - tab[f"recon_mse_{m}"] / np.where(var_tr > 0, var_tr, 1), np.nan)
r2_seeds = np.array([np.where(var_tr > 0, 1 - pv(zz["HAE_recon"], ytr) / np.where(var_tr > 0, var_tr, 1), np.nan) for zz in Z_seeds])
tab["R2_HAE_median_seeds"], tab["R2_HAE_min_seeds"], tab["R2_HAE_max_seeds"] = (np.median(r2_seeds, 0), r2_seeds.min(0), r2_seeds.max(0))
R2H = "R2_HAE_median_seeds"
BL = [("weather_continuous", "Weather, continuous"), ("weather_indicator", "Weather, indicators"),
      ("calendar", "Calendar"), ("holiday", "Named holidays")]
blk = {j: b for b, _ in BL for j in blocks[b]}
tab["block"] = [blk[j] for j in range(p)]
tab.to_csv(d / "nonfunctional_per_variable.csv", index=False)

# ------------------------------------------------------------------ block decomposition
dec = {}
for b, lab in BL + [("weather", "Weather (all 29)")]:
    idx = blocks[b]
    dec[b] = {f"{k}_{m}": float(tab[f"{k}_mse_{m}"].to_numpy()[idx].sum() / p) for k in ("recon", "fore") for m in M}
    dec[b]["n"] = len(idx); dec[b]["n_constant"] = len([j for j in idx if j in const])
tot = {f"{k}_{m}": float(tab[f"{k}_mse_{m}"].sum() / p) for k in ("recon", "fore") for m in M}
# HAE over seeds: contributions per seed -> mean / SD
seed_dec = []
for zz in Z_seeds:
    vr, vf = pv(zz["HAE_recon"], ytr), pv(zz["HAE_fore"], yte)
    row = {f"{b}_{k}": float(v[blocks[b]].sum() / p) for b, _ in BL + [("weather", "")] for k, v in (("recon", vr), ("fore", vf))}
    row |= {"total_recon": float(vr.sum() / p), "total_fore": float(vf.sum() / p)}
    seed_dec.append(row)
seed_dec = pd.DataFrame(seed_dec)
ms = lambda col: (float(seed_dec[col].mean()), float(seed_dec[col].std(ddof=1)) if S > 1 else float("nan"))

# ------------------------------------------------------------------ categorical recovery
T = orig(ytr); R = {m: orig(rec[m]) for m in M}
wc = [j for j in blocks["weather_indicator"] if names[j].startswith("weather_")]
hc = blocks["holiday"]; ih = names.index("is_holiday")


def binary(j, m):
    pr, tr = R[m][:, j] > 0.5, T[:, j] > 0.5
    sens = (pr & tr).sum() / max(tr.sum(), 1); spec = (~pr & ~tr).sum() / max((~tr).sum(), 1)
    return dict(balanced_accuracy=float((sens + spec) / 2), sensitivity=float(sens), specificity=float(spec), n_pos=int(tr.sum()))


cat = {}
for m in M:
    hol = T[:, ih] > 0.5
    cat[m] = dict(weather_condition_accuracy=float((R[m][:, wc].argmax(1) == T[:, wc].argmax(1)).mean()),
                  rain_occurrence=binary(names.index("rain_any"), m), weekend=binary(names.index("is_weekend"), m),
                  holiday=binary(ih, m),
                  named_holiday_accuracy=float((R[m][hol][:, hc].argmax(1) == T[hol][:, hc].argmax(1)).mean()))
def cat_for(Rm):
    hol = T[:, ih] > 0.5
    def bal(j):
        pr, tr = Rm[:, j] > 0.5, T[:, j] > 0.5
        return float(((pr & tr).sum() / max(tr.sum(), 1) + (~pr & ~tr).sum() / max((~tr).sum(), 1)) / 2)
    return dict(weather_condition_accuracy=float((Rm[:, wc].argmax(1) == T[:, wc].argmax(1)).mean()),
                rain_occurrence=bal(names.index("rain_any")), holiday=bal(ih), weekend=bal(names.index("is_weekend")),
                named_holiday_accuracy=float((Rm[hol][:, hc].argmax(1) == T[hol][:, hc].argmax(1)).mean()))


seed_cat = pd.DataFrame([cat_for(orig(zz["HAE_recon"])) for zz in Z_seeds])
cat["weather_condition_majority_share"] = float(pd.Series(T[:, wc].argmax(1)).value_counts(normalize=True).iloc[0])
cat["n_days"] = int(ntr); cat["n_holiday_days"] = int((T[:, ih] > 0.5).sum())
summary = dict(blocks=dec, total=tot, categorical=cat, constant_variables=[names[j] for j in const], n_seeds=S,
               HAE_over_seeds=dict(blocks={c_: dict(zip(("mean", "sd"), ms(c_))) for c_ in seed_dec.columns},
                                   categorical={c_: dict(mean=float(seed_cat[c_].mean()), sd=float(seed_cat[c_].std(ddof=1)) if S > 1 else None,
                                                         min=float(seed_cat[c_].min())) for c_ in seed_cat.columns},
                                   median_R2={b: float(np.nanmedian(tab.loc[(tab.block == b) & ~tab.constant, R2H])) for b, _ in BL}),
               median_R2={b: {m: float(np.nanmedian(tab.loc[(tab.block == b) & ~tab.constant, f"R2_{m}"])) for m in M} for b, _ in BL})
(d / "nonfunctional_summary.json").write_text(json.dumps(summary, indent=2))

# ------------------------------------------------------------------ LaTeX table
red = lambda b_, h_: 100 * (b_ - h_) / b_ if b_ > 0 else float("nan")
L = [r"\begin{tabular}{lcccccc}", r"\toprule",
     r"& \multicolumn{3}{c}{Reconstruction} & \multicolumn{3}{c}{Forecasting} \\",
     r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}", r"Block (no.\ of variables) & HPCA & HAE & Red.\ (\%) & HPCA & HAE & Red.\ (\%) \\", r"\midrule"]
def hae_cell(col):
    mu, sd = ms(col)
    return (f"{mu:.4f} ({sd:.4f})" if S > 1 else f"{mu:.4f}"), mu


for b, lab in BL + [("total", f"Nonfunctional total")]:
    if b == "total":
        L.append(r"\midrule"); hp_r, hp_f, n_ = tot["recon_HPCA"], tot["fore_HPCA"], p
    else:
        hp_r, hp_f, n_ = dec[b]["recon_HPCA"], dec[b]["fore_HPCA"], dec[b]["n"]
    (cr, mr), (cf, mf) = hae_cell(f"{b}_recon"), hae_cell(f"{b}_fore")
    rd = lambda b_, h_: f"{red(b_, h_):.1f}" if b_ >= 0.01 else "--"      # omit for negligible contributions
    L.append(f"{lab} ({n_}) & {hp_r:.4f} & {cr} & {rd(hp_r, mr)} & {hp_f:.4f} & {cf} & {rd(hp_f, mf)} \\\\")
L += [r"\bottomrule", r"\end{tabular}", "", r"\medskip", "", r"\begin{tabular}{lcc}", r"\toprule",
      r"Recovered from the reconstructions (pre-forecast days) & HPCA & HAE \\", r"\midrule"]
pc = lambda x: f"{100 * x:.1f}"
def hc_(key):
    v = seed_cat[key]
    return f"{pc(v.mean())} ({pc(v.std(ddof=1))})" if S > 1 else pc(v.iloc[0])
L.append(rf"Modal weather condition, accuracy (\%) & {pc(cat['HPCA']['weather_condition_accuracy'])} & {hc_('weather_condition_accuracy')} \\")
for key, lab in [("rain_occurrence", "Rain occurrence"), ("holiday", "Holiday indicator"), ("weekend", "Weekend indicator")]:
    L.append(rf"{lab}, balanced accuracy (\%) & {pc(cat['HPCA'][key]['balanced_accuracy'])} & {hc_(key)} \\")
L.append(rf"Named holiday on the {cat['n_holiday_days']} holiday days, accuracy (\%) & {pc(cat['HPCA']['named_holiday_accuracy'])} & {hc_('named_holiday_accuracy')} \\")
L += [r"\bottomrule", r"\end{tabular}"]
(d / "table_nonfunctional.tex").write_text("\n".join(L) + "\n")

# ------------------------------------------------------------------ Figure: per-variable R^2
LABEL = {"temp": "Temperature", "rain_1h": "Rainfall", "snow_1h": "Snowfall", "clouds_all": "Cloud cover"}


def pretty(n):
    for k, v in LABEL.items():
        if n.startswith(k + "_") and n.split("_")[-1] in ("mean", "min", "max", "std"):
            return f"{v} ({n.split('_')[-1]})"
    return {"temp_range": "Temperature range", "clouds_range": "Cloud-cover range", "rain_any": "Rain occurrence",
            "snow_any": "Snow occurrence", "is_holiday": "Holiday indicator", "is_weekend": "Weekend indicator",
            "year_index": "Year index", "month_sin": "Month (sin)", "month_cos": "Month (cos)",
            "weekday_sin": "Weekday (sin)", "weekday_cos": "Weekday (cos)", "doy_sin": "Day of year (sin)",
            "doy_cos": "Day of year (cos)"}.get(n, n.replace("weather_", "Condition: ").replace("holiday_", ""))


groups = [[("weather_continuous", "Weather, continuous"), ("weather_indicator", "Weather, indicators")],
          [("calendar", "Calendar"), ("holiday", "Named holidays")]]
nrows = [sum(len([j for j in blocks[b] if j not in const]) for b, _ in g) + 1.75 * len(g) for g in groups]
fig, axes = plt.subplots(1, 2, figsize=(7.4, 0.168 * max(nrows) + 1.0), sharex=True)
for ax, g in zip(axes, groups):
    y = 0; yt, yl = [], []
    for gi_, (b, lab) in enumerate(g):
        idx = [j for j in blocks[b] if j not in const]
        y += 0.6 if gi_ else 0.0
        ax.text(0.01, y, f"{lab}   (median $R^2$: HPCA {summary['median_R2'][b]['HPCA']:.2f}, HAE {summary['HAE_over_seeds']['median_R2'][b]:.2f})",
                fontsize=6.6, fontweight="bold", va="center", ha="left", transform=ax.get_yaxis_transform())
        y += 1.15
        for j in idx:
            r_h, r_p = tab.loc[j, R2H], tab.loc[j, "R2_HPCA"]
            if S > 1:
                ax.plot([tab.loc[j, "R2_HAE_min_seeds"], tab.loc[j, "R2_HAE_max_seeds"]], [y, y], color=PAPER["HAE"]["color"],
                        lw=2.2, alpha=0.35, solid_capstyle="butt", zorder=2)
            ax.plot([r_p, r_h], [y, y], color="0.75", lw=1.2, zorder=1)
            ax.scatter(r_p, y, s=16, color=PAPER["HPCA"]["color"], marker="o", zorder=2, label="HPCA" if j == idx[0] and b == g[0][0] else None)
            ax.scatter(r_h, y, s=18, color=PAPER["HAE"]["color"], marker="D", zorder=3, label="HAE" if j == idx[0] and b == g[0][0] else None)
            yt.append(y); yl.append(pretty(names[j])); y += 1
    ax.set_yticks(yt); ax.set_yticklabels(yl, fontsize=6.8); ax.set_ylim(y - 0.4, -0.6)
    ax.set_xlim(-0.02, 1.02); ax.grid(axis="x", color="0.92", lw=0.6); ax.tick_params(axis="x", labelsize=7); ax.tick_params(axis="y", length=2, pad=2)
    ax.set_xlabel(r"Reconstruction $R^2_j = 1-\mathrm{MSE}_j/\mathrm{Var}_j$ (pre-forecast days)", fontsize=7.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
h, l = axes[0].get_legend_handles_labels()
if S > 1:
    h.append(plt.Line2D([], [], color=PAPER["HAE"]["color"], lw=2.2, alpha=0.35)); l = list(l) + [f"HAE range over {S} training seeds"]
    l = [x if x != "HAE" else f"HAE (median over {S} seeds)" for x in l]
fig.legend(h, l, loc="upper center", ncol=3, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, 1.02))
fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(d / "metro_nonfunctional_r2.pdf", bbox_inches="tight"); plt.close(fig)

# ------------------------------------------------------------------ Figure: trajectories (original units)
dates_tr = pd.to_datetime(z["dates_tr"]); dates_te = pd.to_datetime(z["dates_te"])
W = 35
sel = [("temp_mean", "Mean temperature (°C)", lambda v: v - 273.15), ("temp_range", "Temperature range (°C)", lambda v: v),
       ("clouds_all_mean", "Mean cloud cover (%)", lambda v: v), ("rain_any", "Rain occurrence (0/1)", lambda v: v),
       ("weekday_cos", "Weekday, cosine encoding", lambda v: v), ("is_weekend", "Weekend indicator (0/1)", lambda v: v)]
fig, axes = plt.subplots(2, 3, figsize=(7.4, 4.4), sharex=True)
Fo = {m: orig(fore[m]) for m in M}; To = orig(yte)
for ax, (v, lab, f) in zip(axes.flat, sel):
    j = names.index(v)
    t_in, t_out = dates_tr[-W:], dates_te
    ax.axvspan(t_out[0] - pd.Timedelta(hours=12), t_out[-1] + pd.Timedelta(hours=12), color="0.93", lw=0)
    ax.plot(t_in, f(T[-W:, j]), **{**PAPER["truth"], "lw": 1.0, "ms": 2.0})
    ax.plot(t_out, f(To[:, j]), **{**PAPER["truth"], "lw": 1.0, "ms": 2.0, "label": None})
    for m in M:
        st = {k: v_ for k, v_ in PAPER[m].items() if k != "label"}
        ax.plot(t_in, f(R[m][-W:, j]), **{**st, "lw": 1.0}, label=f"{m} reconstruction / forecast")
        ax.plot(t_out, f(Fo[m][:, j]), **{**st, "lw": 1.3})
    ax.set_title(lab, fontsize=8); ax.tick_params(labelsize=6.5)
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %d"))
for ax in axes[1]:
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
h, l = axes.flat[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=3, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, 1.03))
fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(d / "metro_nonfunctional_trajectories.pdf", bbox_inches="tight"); plt.close(fig)

print("constant variables:", summary["constant_variables"])
print("median R2 by block:", json.dumps(summary["median_R2"]))
print("block contributions (recon / fore):")
for b, _ in BL + [("weather", "")]:
    v = dec[b]; print(f"  {b:20s} HPCA {v['recon_HPCA']:.4f}/{v['fore_HPCA']:.4f}  HAE {v['recon_HAE']:.4f}/{v['fore_HAE']:.4f}")
print(f"  total                HPCA {tot['recon_HPCA']:.4f}/{tot['fore_HPCA']:.4f}  HAE {tot['recon_HAE']:.4f}/{tot['fore_HAE']:.4f}")
print(f"HAE over {S} seeds:", seed_dec[["total_recon", "total_fore", "weather_fore", "calendar_fore"]].agg(["mean", "std"]).round(4).to_dict())
print("HAE categorical over seeds:", seed_cat.agg(["mean", "std", "min"]).round(3).to_dict())
print("categorical:", json.dumps({m: {k: (v if not isinstance(v, dict) else round(v['balanced_accuracy'], 3)) for k, v in cat[m].items()} for m in M}))
print(f"-> {d}")
