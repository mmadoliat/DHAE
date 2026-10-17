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
