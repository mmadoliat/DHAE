"""Write the LaTeX tables and pgfplots figures of the VHAE draft from the result files.
    python VHAE/make_paper_tables.py --out <paper dir>      (writes tables/*.tex, figures/*.tex)"""
import argparse, glob
from pathlib import Path
import numpy as np, pandas as pd
R = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser(); ap.add_argument("--out", default=str(R / "VHAE" / "paper_assets")); a = ap.parse_args()
O = Path(a.out); (O / "tables").mkdir(parents=True, exist_ok=True); (O / "figures").mkdir(parents=True, exist_ok=True)
def ms(s, d=3):
    s = pd.Series(s).dropna()
    return f"{s.mean():.{d}f}" + (f" ({s.std():.{d}f})" if len(s) > 1 else "")

# ---------------- Metro ----------------
mm = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(R / "VHAE/results/metro/results*.csv"))) if "_fig" not in f], ignore_index=True).drop_duplicates(["method", "seed", "beta"])
mm["beta"] = mm["beta"].fillna(np.nan)
lines = []
for lab, sel in [("HPCA+VAR", mm.method == "HPCA"), ("HAE+VAR", mm.method == "HAE"),
                 (r"VHAE+VAR, $\beta=10^{-3}$", (mm.method == "VHAE") & np.isclose(mm.beta, 1e-3))] + \
                [(rf"VHAE+VAR, $\beta=10^{{{int(np.log10(b))}}}$\textsuperscript{{a}}", (mm.method == "VHAE") & np.isclose(mm.beta, b)) for b in (0.01, 0.1)]:
    g = mm[sel]
    if len(g) == 0: continue
    lines.append(f"{lab} & {len(g) if g.seed.notna().any() else '--'} & {ms(g.fore_func)} & {ms(g.fore_nonfunc)} & {ms(g.fore_weather)} & "
                 f"{ms(g.func_coverage90,2)} & {ms(g.func_interval_score90,2)} & {ms(g.func_crps)} & {ms(g.weather_crps)} \\\\")
(O / "tables/tab_metro.tex").write_text(r"""\begin{table}[!t]\centering
\caption{Metro traffic, seven-day forecasts. MSE of the functional component (standardised $\log(1+x)$ scale), of all 49
nonfunctional variables and of the 29 weather variables (standardised scale); coverage and interval score of pointwise
90\% intervals and CRPS for the curves, and CRPS for the weather variables. Mean (SD) over seeds; $n$ = number of seeds.}
\label{tab:metro}\small\resizebox{\textwidth}{!}{%
\begin{tabular}{lcccccccc}\toprule
 & & \multicolumn{3}{c}{Point forecast MSE} & \multicolumn{3}{c}{Curve, probabilistic} & Weather\\
\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-9}
Method & $n$ & Curve & Scalars & Weather & Cov.\ 90 & IS$_{0.1}$ & CRPS & CRPS\\\midrule
""" + "\n".join(lines) + r"""
\bottomrule\end{tabular}}
\par\smallskip\footnotesize\textsuperscript{a} Single seed (776); sensitivity analysis.
\end{table}
""")
# ---------------- fertility ----------------
ff = pd.read_csv(R / "VHAE/results/fertility/results.csv")
fl = []
for m, sub in ff.groupby(["method", ff.beta.fillna(-1)], sort=False):
    lab = m[0].replace("FPCA(K=", r"FPCA+VAR, $K=").replace(")", "$") if m[0].startswith("FPCA") else (
        "FAE+VAR" if m[0] == "FAE" else rf"VFAE+VAR, $\beta=10^{{{int(np.log10(m[1]))}}}$")
    ratio = ms(sub.post_sd / sub.latent_sd, 3) if "post_sd" in sub and sub.post_sd.notna().any() else "--"
    fl.append(f"{lab} & {ms(sub.recon*1e3,2)} & {ms(sub.fore,4)} & {ms(sub.coverage90,2)} & {ms(sub.interval_score90,4)} & {ms(sub.crps,4)} & {ratio} \\\\")
(O / "tables/tab_fertility.tex").write_text(r"""\begin{table}[!t]\centering
\caption{U.S. fertility, forecasts for 2019--2023 ($K=3$ unless stated; training-year centring). Reconstruction and
forecast relMSE; coverage and interval score of pointwise 90\% intervals; CRPS (rate per woman);
ratio of the mean posterior SD to the SD of the posterior means (VFAE only). Mean (SD) over seeds.}
\label{tab:fertility}\small
\begin{tabular}{lcccccc}\toprule
Method & Recon.\ relMSE ($\times10^{3}$) & Forecast relMSE & Cov.\ 90 & IS$_{0.1}$ & CRPS & $\bar\sigma/\mathrm{sd}(\mu)$\\\midrule
""" + "\n".join(fl) + r"""
\bottomrule\end{tabular}\end{table}
""")
# ---------------- simulation ----------------
sims = [f for f in sorted(glob.glob(str(R / "VHAE/results/simulation/*.csv")))]
if sims:
    sd = pd.concat([pd.read_csv(f) for f in sims], ignore_index=True); sl = []
    for sc, g in sd.groupby("scenario"):
        for m, lab in [("fpca", "FPCA+VAR"), ("fae", "FAE+VAR"), ("vfae", "VFAE+VAR")]:
            sl.append(f"{sc.capitalize() if m=='fpca' else ''} & {lab} & {ms(g[m+'_recon'])} & {ms(g[m+'_fore'])} & "
                      f"{ms(g[m+'_coverage90'],2)} & {ms(g[m+'_interval_score90'],3 if sc=='nonlinear' else 2)} & {ms(g[m+'_crps'],4 if sc=='nonlinear' else 3)} \\\\")
        sl.append(r"\midrule")
    n = sd.groupby("scenario").size().min()
    (O / "tables/tab_simulation.tex").write_text(r"""\begin{table}[!t]\centering
\caption{Simulation, $K=5$, $B=30$, horizon $L=5$, """ + f"{n}" + r""" replicates per scenario. Reconstruction and forecast relMSE
against the noise-free curves; coverage, interval score and CRPS of the predictive distributions against the observed
curves. Mean (SD) over replicates.}
\label{tab:sim}\small
\begin{tabular}{llccccc}\toprule
Scenario & Method & Recon.\ relMSE & Forecast relMSE & Cov.\ 90 & IS$_{0.1}$ & CRPS\\\midrule
""" + "\n".join(sl[:-1]) + r"""
\bottomrule\end{tabular}\end{table}
""")
# ---------------- figures (pgfplots, data inline) ----------------
def series(x, y): return " ".join(f"({a:.3g},{b:.3g})" for a, b in zip(x, y))
def panels(z, keys, grid, labels, xlabel, ylabel, lo, hi, scale=1.0, ncol=None):
    L = len(labels); ncol = ncol or L
    ymax = 1.05 * max(z["truth"].max(), hi.max()) * scale; ymin = min(0, lo.min() * scale)
    sty = {"truth": "black, thick", "HPCA": "HiddenDark, dashed", "FPCA": "HiddenDark, dashed", "HAE": "NonFunc, thick",
           "FAE": "Func, thick", "VHAE": "LatentDark, thick", "VFAE": "LatentDark, thick"}
    name = {"truth": "Observed", "HPCA": "HPCA+VAR", "FPCA": "FPCA+VAR", "HAE": "HAE+VAR", "FAE": "FAE+VAR", "VHAE": "VHAE+VAR", "VFAE": "VFAE+VAR"}
    s = [r"\begin{tikzpicture}\begin{groupplot}[group style={group size=" + f"{ncol} by {int(np.ceil(L/ncol))}" +
         r", horizontal sep=2pt, vertical sep=28pt, y descriptions at=edge left}, width=" + f"{1.0/ncol*1.18:.3f}" +
         r"\textwidth, height=4.2cm, ymin=" + f"{ymin:.3g}, ymax={ymax:.3g}" + r", scale only axis=false, tick label style={font=\tiny},"
         r" title style={font=\scriptsize, yshift=-4pt}, every axis plot/.append style={line width=0.8pt}," +
         (r" ylabel={" + ylabel + r"}, ylabel style={font=\scriptsize}," ) + r" legend style={font=\scriptsize, draw=none, legend columns=-1}]"]
    for i in range(L):
        opts = f"title={{{labels[i]}}}" + (", legend to name=sharedlegend" if i == 0 else "") + (r", axis y line*=left, y axis line style={dashed}" if i > 0 else "")
        s.append(rf"\nextgroupplot[{opts}]")
        s.append(rf"\addplot[name path=hi{i}, draw=none, forget plot] coordinates {{{series(grid, hi[i]*scale)}}};")
        s.append(rf"\addplot[name path=lo{i}, draw=none, forget plot] coordinates {{{series(grid, lo[i]*scale)}}};")
        s.append(rf"\addplot[Latent, fill opacity=0.22, draw=none{', forget plot' if i else ''}] fill between[of=hi{i} and lo{i}];" + (r"\addlegendentry{90\% band}" if i == 0 else ""))
        for k in keys:
            s.append(rf"\addplot[{sty[k]}{', forget plot' if i else ''}] coordinates {{{series(grid, z[k][i]*scale)}}};" + (rf"\addlegendentry{{{name[k]}}}" if i == 0 else ""))
    s.append(r"\end{groupplot}\node at ($(group c1r1.north)!0.5!(group c" + f"{ncol}" + r"r1.north)+(0,22pt)$) {\pgfplotslegendfromname{sharedlegend}};")
    s.append(r"\node[font=\scriptsize, anchor=north] at ([yshift=-2pt]current bounding box.south) {" + xlabel + "};\\end{tikzpicture}")
    return "\n".join(s)
z = np.load(R / "VHAE/results/metro/forecasts_fig.npz", allow_pickle=True)
labs = [pd.Timestamp(str(d)).strftime("%a %b %-d") for d in z["dates"]]
(O / "figures/fig_metro.tex").write_text(panels(z, ["truth", "HPCA", "HAE", "VHAE"], np.arange(24), labs, "Hour of day",
    r"Traffic volume ($\times 10^3$)", z["VHAE_lo"], z["VHAE_hi"], scale=1e-3, ncol=4))
zf = np.load(R / "VHAE/results/fertility/forecasts.npz", allow_pickle=True)
zf = {k: (zf[k][..., ::2] if k not in ("years",) else zf[k]) for k in zf.files}
(O / "figures/fig_fertility.tex").write_text(panels(zf, ["truth", "FPCA", "FAE", "VFAE"], zf["ages"], [str(y) for y in zf["years"]],
    "Age", "Fertility rate (per woman)", zf["VFAE_lo"], zf["VFAE_hi"]))
print("wrote", sorted(str(p.relative_to(O)) for p in O.rglob("*.tex")))
