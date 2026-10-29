"""Figures in the house style: compact panels, shared axes, consistent y-limits,
one shared legend, vector PDF."""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STYLE = {"truth": dict(color="black", lw=1.6, label="Observed"),
         "HPCA": dict(color="#7f7f7f", lw=1.3, ls="--", label="HPCA+VAR"),
         "HAE": dict(color="#F15A24", lw=1.5, label="HAE+VAR"),
         "VHAE": dict(color="#21B6CF", lw=1.5, label="VHAE+VAR"),
         "FPCA": dict(color="#7f7f7f", lw=1.3, ls="--", label="FPCA+VAR"),
         "FAE": dict(color="#159A6A", lw=1.5, label="FAE+VAR"),
         "VFAE": dict(color="#21B6CF", lw=1.5, label="VFAE+VAR")}


def plot_metro_forecasts(npz_path, methods, out_pdf, bands=None):
    z = np.load(npz_path, allow_pickle=True)
    truth, dates = z["truth"], z["dates"]; L = truth.shape[0]; hrs = np.arange(24)
    fig, axes = plt.subplots(1, L, figsize=(1.75 * L, 2.3), sharex=True, sharey=True)
    ymax = 1.05 * max(truth.max(), *[z[m].max() for m in methods])
    for i, ax in enumerate(axes):
        if bands is not None:
            ax.fill_between(hrs, bands["lo"][i], bands["hi"][i], color=STYLE[bands["method"]]["color"], alpha=0.18, lw=0,
                            label=f"90% band ({STYLE[bands['method']]['label']})" if i == 0 else None)
        ax.plot(hrs, truth[i], **{**STYLE["truth"], "label": STYLE["truth"]["label"] if i == 0 else None})
        for m in methods:
            ax.plot(hrs, z[m][i], **{**STYLE[m], "label": STYLE[m]["label"] if i == 0 else None})
        ax.set_title(str(dates[i]), fontsize=8); ax.set_ylim(0, ymax); ax.set_xticks([0, 12, 23])
        ax.tick_params(labelsize=7)
        if i > 0:
            ax.spines["left"].set_linestyle((0, (3, 3)))
    fig.supxlabel("Hour of day", fontsize=9); axes[0].set_ylabel("Traffic volume", fontsize=9)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=4, frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout(w_pad=0.3); fig.savefig(out_pdf, bbox_inches="tight"); plt.close(fig)


def plot_curve_forecasts(grid, truth, preds: dict, labels, out_pdf, xlabel, ylabel, bands=None):
    """Generic per-horizon panels (fertility, simulation)."""
    L = truth.shape[0]
    fig, axes = plt.subplots(1, L, figsize=(1.9 * L, 2.3), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    ymin = min(truth.min(), *[p.min() for p in preds.values()]); ymax = max(truth.max(), *[p.max() for p in preds.values()])
    if bands is not None:
        ymin, ymax = min(ymin, bands["lo"].min()), max(ymax, bands["hi"].max())
    pad = 0.05 * (ymax - ymin)
    for i, ax in enumerate(axes):
        if bands is not None:
            ax.fill_between(grid, bands["lo"][i], bands["hi"][i], color=STYLE[bands["method"]]["color"], alpha=0.18, lw=0,
                            label=f"90% band ({STYLE[bands['method']]['label']})" if i == 0 else None)
        ax.plot(grid, truth[i], **{**STYLE["truth"], "label": "Observed" if i == 0 else None})
        for m, p in preds.items():
            ax.plot(grid, p[i], **{**STYLE[m], "label": STYLE[m]["label"] if i == 0 else None})
        ax.set_title(str(labels[i]), fontsize=8); ax.set_ylim(ymin - pad, ymax + pad); ax.tick_params(labelsize=7)
        if i > 0:
            ax.spines["left"].set_linestyle((0, (3, 3)))
    fig.supxlabel(xlabel, fontsize=9); axes[0].set_ylabel(ylabel, fontsize=9)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=4, frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout(w_pad=0.3); fig.savefig(out_pdf, bbox_inches="tight"); plt.close(fig)


# ------------------------------------------------------------------ paper style
# Colours/line styles of the figures in the HAE manuscript (Figs. fertility_forecasts, hae_vs_hpca).
PAPER = {"truth": dict(color="#0B4A8B", lw=1.8, marker="o", ms=2.2, label="Observed"),
         "HPCA": dict(color="#E0A030", lw=1.8, ls="--", label="HPCA forecast"),
         "FPCA": dict(color="#E0A030", lw=1.8, ls="--", label="FPCA forecast"),
         "HAE": dict(color="#C8586C", lw=1.8, ls="-.", label="HAE forecast"),
         "FAE": dict(color="#C8586C", lw=1.8, ls="-.", label="FAE forecast")}


def _paper_grid(n, ncol=3, panel=(3.4, 2.15)):
    nrow = int(np.ceil(n / ncol))
    fig = plt.figure(figsize=(panel[0] * ncol, panel[1] * nrow))
    gs = fig.add_gridspec(nrow, 2 * ncol)
    axes = []
    for i in range(n):
        r, c = divmod(i, ncol)
        last = (r == nrow - 1) and (n % ncol == 1) and n > 1   # centre a single last panel
        sl = slice(ncol - 1, ncol + 1) if last else slice(2 * c, 2 * c + 2)
        axes.append(fig.add_subplot(gs[r, sl]))
    return fig, axes, nrow


def plot_paper_panels(grid, truth, preds: dict, titles, out_pdf, xlabel, ylabel, xticks=None, ncol=3, smooth=True):
    """Per-horizon panels in the manuscript layout: 3 columns, shared y-limits, legend in the
    first free cell (or under the panels). smooth=True draws the forecast lines through a
    200-point cubic interpolant of the grid values, as in the manuscript figures (observed
    values are drawn as a polyline with markers)."""
    from scipy.interpolate import CubicSpline
    gf = np.linspace(grid[0], grid[-1], 200)
    disp = (lambda v: (gf, CubicSpline(grid, v)(gf))) if smooth else (lambda v: (grid, v))
    L = truth.shape[0]
    fig, axes, nrow = _paper_grid(L, ncol)
    lo = min(truth.min(), *[p.min() for p in preds.values()]); hi = max(truth.max(), *[p.max() for p in preds.values()])
    pad = 0.05 * (hi - lo)
    for i, ax in enumerate(axes):
        ax.plot(grid, truth[i], **PAPER["truth"])
        for m, p in preds.items():
            ax.plot(*disp(p[i]), **PAPER[m])
        ax.set_title(str(titles[i]), fontsize=10); ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlim(grid[0], grid[-1])
        if xticks is not None:
            ax.set_xticks(xticks)
        ax.tick_params(labelsize=8.5)
        if i % ncol == 0 or (i == L - 1 and L % ncol == 1):
            ax.set_ylabel(ylabel, fontsize=10)
        else:
            ax.set_yticklabels([])
    handles, labels = axes[0].get_legend_handles_labels()
    if L % ncol:
        fig.legend(handles, labels, loc="center", bbox_to_anchor=(0.80, 0.5 / nrow), frameon=True, fontsize=9)
    else:
        fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False, fontsize=9)
    fig.supxlabel(xlabel, fontsize=10)
    fig.tight_layout(); fig.savefig(out_pdf, bbox_inches="tight"); plt.close(fig)
