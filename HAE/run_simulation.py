"""Simulation study of the HAE paper (Section 3; Tables 'sim_fae_fpca', 'sim_fae_fpca2',
Figure 'combined_boxplots'), with the design as written in the paper:

* N = 500 curves on J = 100 grid points of [0, 1], last L = 5 held out (T = 495);
  data-generating latent state of dimension 5; linear benchmark mapped to 30 cubic
  B-spline coefficients; nonlinear amplitude--phase benchmark with sigma_h = 0.028 and
  sigma_eta = 0.001 (Appendix);
* fitted dimension K in {3, 5, 7}, basis dimension B in {10, 20, 30, 40, 50};
* FPCA: first K principal components of the B-dimensional B-spline representation of the
  training curves; FAE: M(I) = M(O) = B, encoder B-128-K, decoder K-128-B, tanh, Adam
  (lr 7e-4), batch 16, 1400 epochs, second-difference penalty lambda = 2e-4, trained on the
  observed curves; both followed by VAR(1) on the scores / latents (two-stage);
* relMSE against the noise-free curves; R = 100 replicates.

Settings the paper does not state are listed in hybridae.simstudy.PaperSimCfg.

    python HAE/run_simulation.py --reps 100 --workers 60           # full grid, both benchmarks
    python HAE/run_simulation.py --scenario nonlinear --K 5 --B 30 --reps 3
    python HAE/run_simulation.py --legacy-notebook --scenario nonlinear --K 3 --B 45 --reps 1

Results: HAE/results/simulation/paper_<scenario>.csv (one row per replicate x K x B; reruns
skip replicates already present). Summarize with HAE/summarize_simulation.py.
"""
import argparse, os, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
import pandas as pd

def _task(args):
    sc, rep, Ks, Bs = args
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import torch; torch.set_num_threads(1)
    from hybridae.simstudy import run_replicate_paper
    return run_replicate_paper(sc, rep, Ks=Ks, Bs=Bs)



if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", nargs="+", default=["linear", "nonlinear"], choices=["linear", "nonlinear"])
    ap.add_argument("--K", type=int, nargs="+", default=[3, 5, 7])
    ap.add_argument("--B", type=int, nargs="+", default=[10, 20, 30, 40, 50])
    ap.add_argument("--reps", type=int, default=100); ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--legacy-notebook", action="store_true",
                    help="amplitude--phase notebook procedure (clean target + latent-VAR penalty); see the audit")
    ap.add_argument("--out", default=str(ROOT / "HAE" / "results" / "simulation"))
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)


    if a.legacy_notebook:
        from hybridae.simstudy import run_replicate
        for sc in a.scenario:
            for K in a.K:
                for B in a.B:
                    f = out / f"legacy_{sc}_K{K}_B{B}.csv"; fixed = None
                    for rep in range(a.start, a.start + a.reps):
                        r, fixed = run_replicate(sc, rep, K, B, methods=("fpca", "fae"), fae_mode="notebook", fixed=fixed)
                        pd.DataFrame([r]).to_csv(f, mode="a", header=not f.exists(), index=False)
                        print(f"[legacy] {sc} K={K} B={B} rep={rep}: FPCA {r['fpca_recon']:.4f}/{r['fpca_fore']:.4f} "
                              f"FAE {r['fae_recon']:.4f}/{r['fae_fore']:.4f}", flush=True)
        sys.exit(0)

    tasks = []
    for sc in a.scenario:
        f = out / f"paper_{sc}.csv"
        done = set()
        if f.exists():
            d = pd.read_csv(f); done = set(d.groupby("rep").size().loc[lambda s: s >= len(a.K) * len(a.B)].index)
        tasks += [(sc, rep, tuple(a.K), tuple(a.B)) for rep in range(a.start, a.start + a.reps) if rep not in done]
    print(f"{len(tasks)} replicate tasks ({len(a.K) * len(a.B)} FPCA/FAE fits each), workers={a.workers}", flush=True)
    t0 = time.time()
    if a.workers > 1:
        import multiprocessing as mp
        pool = mp.get_context("spawn").Pool(a.workers, maxtasksperchild=4)
        it = pool.imap_unordered(_task, tasks)
    else:
        it = map(_task, tasks)
    for i, rows in enumerate(it, 1):
        d = pd.DataFrame(rows); f = out / f"paper_{d.scenario.iloc[0]}.csv"
        d.to_csv(f, mode="a", header=not f.exists(), index=False)
        m = d[(d.K == 5) & (d.B == 30)] if ((d.K == 5) & (d.B == 30)).any() else d.iloc[[0]]
        r = m.iloc[0]
        print(f"[{i}/{len(tasks)}] {r.scenario} rep={r.rep} (K={r.K},B={r.B}) FPCA {r.fpca_recon:.4f}/{r.fpca_fore:.4f} "
              f"FAE {r.fae_recon:.4f}/{r.fae_fore:.4f}  elapsed {time.time()-t0:.0f}s", flush=True)
    if a.workers > 1:
        pool.close(); pool.join()
