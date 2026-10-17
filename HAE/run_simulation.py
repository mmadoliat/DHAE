"""Simulation study of the HAE paper (Tables 'sim_fae_fpca', 'sim_fae_fpca2',
Figure 'combined_boxplots').

    python HAE/run_simulation.py --scenario nonlinear --K 5 --B 30 --reps 100
    python HAE/run_simulation.py --grid --reps 100          # full 2 x 3 x 5 grid

--fae-mode notebook  : amplitude--phase FAE exactly as in the notebook
                       (clean training target + latent-VAR penalty, dyn_lambda=8)
--fae-mode twostage  : FAE trained on the observed curves, no dynamics penalty
                       (the procedure described in the paper)
Each replicate is appended to HAE/results/simulation/<scenario>_K<K>_B<B>_<mode>.csv;
summarize with HAE/summarize_simulation.py.
"""
import argparse, sys, time
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from hybridae.simstudy import run_replicate

ap = argparse.ArgumentParser()
ap.add_argument("--scenario", default="nonlinear", choices=["linear", "nonlinear"])
ap.add_argument("--K", type=int, default=5); ap.add_argument("--B", type=int, default=30)
ap.add_argument("--reps", type=int, default=100); ap.add_argument("--start", type=int, default=1)
ap.add_argument("--fae-mode", default="notebook", choices=["notebook", "twostage"])
ap.add_argument("--grid", action="store_true")
ap.add_argument("--out", default=str(ROOT / "HAE" / "results" / "simulation"))
a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
grid = [(s, K, B) for s in ("linear", "nonlinear") for K in (3, 5, 7) for B in (10, 20, 30, 40, 50)] if a.grid else [(a.scenario, a.K, a.B)]
for sc, K, B in grid:
    f = out / f"{sc}_K{K}_B{B}_{a.fae_mode}.csv"; fixed = None
    for rep in range(a.start, a.start + a.reps):
        t0 = time.time()
        r, fixed = run_replicate(sc, rep, K, B, methods=("fpca", "fae"), fae_mode=a.fae_mode, fixed=fixed)
        pd.DataFrame([r]).to_csv(f, mode="a", header=not f.exists(), index=False)
        print(f"{sc} K={K} B={B} rep={rep}: FPCA recon {r['fpca_recon']:.3f} fore {r['fpca_fore']:.3f} | "
              f"FAE recon {r['fae_recon']:.3f} fore {r['fae_fore']:.3f} ({time.time()-t0:.0f}s)", flush=True)
