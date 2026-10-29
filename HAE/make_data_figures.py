"""Data figures of the HAE paper.

    python HAE/make_data_figures.py

Figure 'usafertility'    -> HAE/results/figures/usafertility.pdf
    U.S. age-specific fertility rates 1933-2023 (raw data), coloured by year.
    NB: the rates in data/USAfertility.csv are births per woman (the paper figure's axis
    label says 'per 1,000 women').
Figure 'metro_observed'  -> HAE/results/figures/weekday_weekend_traffic_curves.pdf
    The 1,214 complete daily traffic-volume curves, weekdays vs. weekends.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from hybridae.data import load_usa_fertility, load_hourly_traffic, build_metro_daily

out = ROOT / "HAE" / "results" / "figures"; out.mkdir(parents=True, exist_ok=True)

# ---- Figure 'usafertility'
ages, years, raw, _ = load_usa_fertility(ROOT / "data" / "USAfertility.csv", smooth_s=None)
fig, ax = plt.subplots(figsize=(9.5, 5.4))
cmap = plt.get_cmap("viridis"); norm = matplotlib.colors.Normalize(years.min(), years.max())
for y, r in zip(years, raw):
    ax.plot(ages, r, color=cmap(norm(y)), lw=1.2)
sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
fig.colorbar(sm, ax=ax, pad=0.02).set_label("Year", fontsize=11)
ax.set_xlabel("Age", fontsize=11); ax.set_ylabel("Fertility rate (births per woman)", fontsize=11)
ax.set_title(f"USA fertility curves by year, {years.min()}-{years.max()} (raw data)", fontsize=11)
fig.tight_layout(); fig.savefig(out / "usafertility.pdf", bbox_inches="tight"); plt.close(fig)

# ---- Figure 'metro_observed'
_, x_counts, _, _, dates = build_metro_daily(load_hourly_traffic(ROOT / "data" / "metro_interstate_traffic_volume.zip"))
wk = np.asarray(dates.day_of_week >= 5)
fig, ax = plt.subplots(figsize=(10, 5.4)); hrs = np.arange(24)
for lab, m, col in [("Weekday", ~wk, "#0B3B73"), ("Weekend", wk, "#E69F00")]:
    ax.plot(hrs, x_counts[m].T, color=col, lw=0.5, alpha=0.35)
    ax.plot([], [], color=col, lw=2.5, label=f"{lab} (n = {int(m.sum())})")
ax.set_xlim(0, 23); ax.set_xticks([0, 4, 8, 12, 16, 20, 23])
ax.set_xlabel("Hour of day", fontsize=12); ax.set_ylabel("Traffic volume", fontsize=12)
ax.set_title(f"Observed daily traffic-volume curves ({len(dates):,} complete days): weekday vs. weekend", fontsize=12)
ax.legend(fontsize=11, loc="upper right")
fig.tight_layout(); fig.savefig(out / "weekday_weekend_traffic_curves.pdf", bbox_inches="tight"); plt.close(fig)
print(f"fertility: {raw.shape[0]} years x {raw.shape[1]} ages, max rate {raw.max():.3f}; "
      f"metro: {len(dates)} days ({int(wk.sum())} weekend) -> {out}")
