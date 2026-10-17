"""Summarize simulation CSVs as mean (SD) tables in the layout of the HAE paper."""
import sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
d = pd.concat([pd.read_csv(f) for f in sorted((ROOT / "HAE" / "results" / "simulation").glob("*.csv"))])
for kind in ["recon", "fore"]:
    rows = []
    for (sc, mode, K), g in d.groupby(["scenario", "fae_mode", "K"]):
        row = {"scenario": sc, "mode": mode, "K": K}
        for B, gb in g.groupby("B"):
            for m in ["fpca", "fae"]:
                row[f"B{B}_{m.upper()}"] = f"{gb[f'{m}_{kind}'].mean():.3f} ({gb[f'{m}_{kind}'].std():.3f}) n={len(gb)}"
        rows.append(row)
    print(f"\n=== {kind} relMSE vs clean ==="); print(pd.DataFrame(rows).to_string(index=False))
