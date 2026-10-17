"""VHAE paper, simulation: FPCA+VAR vs two-stage FAE+VAR vs VFAE+VAR on the
linear and amplitude--phase benchmarks (point relMSE vs clean curves;
probabilistic scores vs observed curves).

    python VHAE/run_simulation.py --scenario nonlinear --K 5 --B 30 --reps 20 [--beta 1e-3]
"""
import argparse, sys, time
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
import hybridae.simstudy as ss
from hybridae.simstudy import run_replicate, make_data

ap = argparse.ArgumentParser()
ap.add_argument("--scenario", default="nonlinear", choices=["linear", "nonlinear"])
ap.add_argument("--K", type=int, default=5); ap.add_argument("--B", type=int, default=30)
ap.add_argument("--reps", type=int, default=20); ap.add_argument("--start", type=int, default=1)
ap.add_argument("--beta", type=float, default=1e-3)
ap.add_argument("--out", default=str(ROOT / "VHAE" / "results" / "simulation"))
a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
f = out / f"{a.scenario}_K{a.K}_B{a.B}_beta{a.beta:g}.csv"; fixed = None
for rep in range(a.start, a.start + a.reps):
    t0 = time.time()
    Xc, Xn, u, fixed = make_data(a.scenario, rep, fixed=fixed)
    r, fixed = run_replicate(a.scenario, rep, a.K, a.B, methods=("fpca", "fae", "vfae"), fae_mode="twostage",
                             beta=a.beta, fixed=fixed)
    r["beta"] = a.beta
    pd.DataFrame([r]).to_csv(f, mode="a", header=not f.exists(), index=False)
    print(f"{a.scenario} rep {rep}: fore FPCA {r['fpca_fore']:.3f} FAE {r['fae_fore']:.3f} VFAE {r['vfae_fore']:.3f} | "
          f"cov90 {r['fpca_coverage90']:.2f}/{r['fae_coverage90']:.2f}/{r['vfae_coverage90']:.2f} ({time.time()-t0:.0f}s)", flush=True)
