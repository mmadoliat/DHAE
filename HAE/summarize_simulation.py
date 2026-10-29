"""Summarize HAE/results/simulation/paper_<scenario>.csv in the layout of the HAE paper.

    python HAE/summarize_simulation.py

Writes to HAE/results/simulation/:
  summary.csv                 mean, SD and n per scenario x K x B x method x metric
  table_recon.tex             Table 'sim_fae_fpca'  (reconstruction relMSE, mean (SD))
  table_forecast.tex          Table 'sim_fae_fpca2' (forecasting relMSE, mean (SD))
  combined_boxplots.pdf       Figure 'combined_boxplots' (log10 axis when the range spans >1.5 decades)
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
D = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "HAE" / "results" / "simulation"
files = sorted(D.glob("paper_*.csv"))
if not files:
    sys.exit(f"no paper_*.csv in {D}; run HAE/run_simulation.py first")
d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
d = d.drop_duplicates(["scenario", "rep", "K", "B"], keep="last")
Ks, Bs = sorted(d.K.unique()), sorted(d.B.unique())

long = d.melt(id_vars=["scenario", "rep", "K", "B"], value_vars=["fpca_recon", "fae_recon", "fpca_fore", "fae_fore"],
              var_name="mk", value_name="relMSE")
long["method"] = long.mk.str.split("_").str[0].str.upper(); long["metric"] = long.mk.str.split("_").str[1]
summ = (long.groupby(["scenario", "metric", "K", "B", "method"]).relMSE
        .agg(mean="mean", sd="std", median="median", n="count").reset_index())
summ.to_csv(D / "summary.csv", index=False)


def fmt(v):
    return f"{v:.3f}" if v >= 0.0095 else f"{v:.1e}".replace("e-0", "e-")


def table(metric, caption, label):
    lines = [r"\begin{table}[!t]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{ll" + "cc" * len(Bs) + "}", r"\toprule",
             r"\multirow{2}{*}{Scenario} & \multirow{2}{*}{$K$} & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{$B={B}$}}" for B in Bs) + r" \\",
             "".join(rf"\cmidrule(lr){{{3 + 2*i}-{4 + 2*i}}}" for i in range(len(Bs))),
             "& & " + " & ".join(["FPCA & FAE"] * len(Bs)) + r"\\", r"\midrule"]
    for j, sc in enumerate(["linear", "nonlinear"]):
        if sc not in d.scenario.unique():
            continue
        if j:
            lines.append(r"\midrule")
        for i, K in enumerate(Ks):
            cells = []
            for B in Bs:
                for m in ["FPCA", "FAE"]:
                    r = summ[(summ.scenario == sc) & (summ.metric == metric) & (summ.K == K) & (summ.B == B) & (summ.method == m)]
                    cells.append(f"{fmt(r['mean'].iloc[0])} ({fmt(r['sd'].iloc[0])})" if len(r) else "--")
            lead = rf"\multirow{{{len(Ks)}}}{{*}}{{{sc.capitalize()}}}" if i == 0 else ""
            lines.append(f"{lead} & ${K}$ & " + " & ".join(cells) + r" \\")
    n = int(summ.n.min())
    lines += [r"\bottomrule", r"\end{tabular}%", "}", rf"% R = {n} replicates per cell (minimum over cells)", r"\end{table}"]
    return "\n".join(lines) + "\n"


(D / "table_recon.tex").write_text(table("recon", "Reconstruction accuracy of FPCA and FAE (relMSE against the noise-free "
                                          "curves, mean and SD across replications).", "tab:sim_fae_fpca"))
(D / "table_forecast.tex").write_text(table("fore", "Out-of-sample forecasting accuracy of FPCA+VAR(1) and FAE+VAR(1) over the "
                                             "five held-out curves (relMSE against the noise-free curves, mean and SD).", "tab:sim_fae_fpca2"))

# ---------------- Figure 'combined_boxplots' (colours of the manuscript figure)
COL = {("FPCA", 3): "#F9D1A8", ("FPCA", 5): "#EE6A1E", ("FPCA", 7): "#A63A12",
       ("FAE", 3): "#B9E0DC", ("FAE", 5): "#2A9D8F", ("FAE", 7): "#0B5E55"}
fig, axes = plt.subplots(2, 2, figsize=(11, 7.2))
panels = [("recon", "linear", "(a) Reconstruction | Linear"), ("recon", "nonlinear", "(b) Reconstruction | Nonlinear"),
          ("fore", "linear", "(c) Forecast | Linear"), ("fore", "nonlinear", "(d) Forecast | Nonlinear")]
combos = [("FPCA", K) for K in Ks] + [("FAE", K) for K in Ks]
width = 0.8 / len(combos)
for ax, (metric, sc, title) in zip(axes.flat, panels):
    sub = long[(long.metric == metric) & (long.scenario == sc)]
    if sub.empty:
        ax.set_visible(False); continue
    logy = np.log10(sub.relMSE.max() / max(sub.relMSE.min(), 1e-12)) > 1.5
    for gi, B in enumerate(Bs):
        for ci, (m, K) in enumerate(combos):
            v = sub[(sub.B == B) & (sub.K == K) & (sub.method == m)].relMSE.to_numpy()
            if not len(v):
                continue
            pos = gi + (ci - (len(combos) - 1) / 2) * width
            bp = ax.boxplot(v, positions=[pos], widths=width * 0.85, patch_artist=True, showfliers=True,
                            flierprops=dict(marker="o", ms=2.5, mfc="white", mec="0.3", mew=0.6),
                            medianprops=dict(color="0.15", lw=1), whiskerprops=dict(color="0.25", lw=0.8),
                            capprops=dict(color="0.25", lw=0.8), boxprops=dict(edgecolor="0.25", lw=0.8))
            bp["boxes"][0].set_facecolor(COL[(m, K)])
        if gi:
            ax.axvline(gi - 0.5, color="0.8", lw=0.8, ls="--")
    if logy:
        ax.set_yscale("log")
    ax.set_xticks(range(len(Bs))); ax.set_xticklabels([str(B) for B in Bs]); ax.set_xlim(-0.5, len(Bs) - 0.5)
    ax.set_title(title, fontweight="bold", fontsize=11); ax.grid(axis="y", color="0.92", lw=0.6)
    ax.set_xlabel("B" if metric == "fore" else ""); ax.set_ylabel("Relative MSE" if sc == "linear" else "")
    for s in ax.spines.values():
        s.set_color("0.75")
handles = [plt.Rectangle((0, 0), 1, 1, fc=COL[c], ec="0.25", lw=0.8) for c in combos]
fig.legend(handles, [f"{m} (K={K})" for m, K in combos], loc="lower center", ncol=3, frameon=False, fontsize=10,
           bbox_to_anchor=(0.5, -0.06))
fig.tight_layout(); fig.savefig(D / "combined_boxplots.pdf", bbox_inches="tight"); plt.close(fig)

wide = summ.pivot_table(index=["scenario", "metric", "K", "B"], columns="method", values="mean").reset_index()
print(f"replicates per cell: {int(summ.n.min())}-{int(summ.n.max())}")
print(wide[(wide.K == 5)].round(4).to_string(index=False))
print(f"-> {D}")
