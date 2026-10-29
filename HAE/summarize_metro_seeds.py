"""Seed variability of the Metro results (Table 'traffic_hybrid') and of the rolling-origin check.

    for s in 777 778 779 780 781 782 783 784 785; do
        python HAE/run_metro.py --seed $s --no-plot --out HAE/results/metro_seeds/seed$s; done
    for s in 776 777 778 779 780; do
        python HAE/run_metro_rolling.py --seed $s --out HAE/results/metro_rolling/seed$s; done
    python HAE/summarize_metro_seeds.py

Writes HAE/results/metro_seeds/summary.csv (and metro_rolling/summary_seeds.csv when present).
HPCA is deterministic; the 'paper' column is Table 'traffic_hybrid' of the manuscript
(original notebook run, seed 776, Windows/Python 3.9).
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "HAE" / "results"
PAPER = {"recon_func": 0.0664, "recon_nonfunc": 0.0735, "recon_hybrid": 0.1399, "fore_func": 0.2831,
         "fore_nonfunc": 0.3422, "fore_hybrid": 0.6253, "contrib_weather_recon": 0.0514, "contrib_weather_fore": 0.2679}
PAPER_HPCA = {"recon_func": 0.1920, "recon_nonfunc": 0.6085, "recon_hybrid": 0.8004, "fore_func": 0.4479,
              "fore_nonfunc": 0.4086, "fore_hybrid": 0.8565, "contrib_weather_recon": 0.2738, "contrib_weather_fore": 0.2784}
files = [R / "metro" / "results.json"] + sorted((R / "metro_seeds").glob("seed*/results.json"))
runs = [json.load(open(f)) for f in files if f.exists()]
if not runs:
    sys.exit("no Metro results found")
rows = []
for k in PAPER:
    v = np.array([r["HAE"][k] for r in runs]); hp = runs[0]["HPCA"][k]
    red = 100 * (hp - v) / hp
    rows.append(dict(metric=k, paper_HAE=PAPER[k], paper_HPCA=PAPER_HPCA[k], rerun_HPCA=hp,
                     HAE_mean=v.mean(), HAE_sd=v.std(ddof=1) if len(v) > 1 else np.nan, HAE_min=v.min(), HAE_max=v.max(),
                     reduction_mean_pct=red.mean(), reduction_min_pct=red.min(), reduction_max_pct=red.max(),
                     n_better_than_HPCA=int((v < hp).sum()), n_seeds=len(v)))
lags = [r["HAE"]["var"]["lag"] for r in runs]
t = pd.DataFrame(rows); (R / "metro_seeds").mkdir(parents=True, exist_ok=True)
t.to_csv(R / "metro_seeds" / "summary.csv", index=False)
print(f"Metro, HAE over {len(runs)} training seeds (seeds {[r['seed'] for r in runs]}); AIC VAR lags {lags}")
print(t.round(4).to_string(index=False))

roll = sorted((R / "metro_rolling").glob("seed*/summary.json"))
if roll:
    rr = []
    for f in roll:
        s = json.load(open(f))
        for m, d in s["by_method"].items():
            for k, v in d.items():
                rr.append(dict(seed=s["seed"], method=m, metric=k, mean=v["mean"], median=v["median"]))
        for k, v in s["HAE_better_than_HPCA_origins"].items():
            rr.append(dict(seed=s["seed"], method="HAE<HPCA origins", metric=k, mean=v, median=v))
    rr = pd.DataFrame(rr)
    rs = rr.groupby(["method", "metric"])[["mean", "median"]].agg(["mean", "min", "max"]).round(4)
    rs.to_csv(R / "metro_rolling" / "summary_seeds.csv")
    print(f"\nRolling origin ({len(roll)} seeds x {json.load(open(roll[0]))['n_origins']} weekly origins):")
    print(rs.to_string())
