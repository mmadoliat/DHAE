# HAE paper — reproduction scripts

*Hybrid Autoencoders with Shared Temporal Latent Dynamics for Forecasting*
(Pourmoshir, Haghbin & Maadooliat). Manuscript: Overleaf/Dropbox `Mobina - FAE/HAE/main_HM.tex`
(version of 2026-10-27); `main_PHM.tex` adds the nonfunctional diagnostics of Section 4.2.

The scripts implement the procedures **as they are written in the manuscript**. Where the
manuscript is silent, the value of Mobina's original notebook is used and listed below.
Whether each paper number is reproduced is summarized in
[`../REPRODUCIBILITY_AUDIT.md`](../REPRODUCIBILITY_AUDIT.md).

## Script → paper mapping

| Paper item | Script | Output (`HAE/results/…`) | Reproduces the paper? |
|---|---|---|---|
| Sec. 3, Tables 2–3, Fig. 3 (simulation) | `run_simulation.py --workers N`, then `summarize_simulation.py` | `simulation/paper_{linear,nonlinear}.csv`, `table_recon.tex`, `table_forecast.tex`, `combined_boxplots.pdf` | **No** — see audit §1 |
| Sec. 4.1, Fig. 4 (fertility data) | `make_data_figures.py` | `figures/usafertility.pdf` | yes (axis-label unit differs, audit §2) |
| Sec. 4.1, relMSE values, Fig. 5 | `run_fertility.py` (`--legacy-notebook` for the original notebook) | `fertility/results_*.json`, `fertility_forecasts_*.pdf` | **No** — see audit §2 |
| Sec. 4.2, Table 4 (49 variables) | `src/hybridae/data.py::build_metro_daily` | — | yes (6 of the 49 are constant, audit §3) |
| Sec. 4.2, Fig. 6 (traffic curves) | `make_data_figures.py` | `figures/weekday_weekend_traffic_curves.pdf` | yes |
| Sec. 4.2, Table 5, Fig. 7, weather-only contributions | `run_metro.py` | `metro/results.json`, `forecasts.npz`, `hae_vs_hpca.pdf` | HPCA exactly; HAE approximately (audit §3) |
| Seed variability of Table 5 | `run_metro.py --seed s --no-plot --out metro_seeds/seed$s`, then `summarize_metro_seeds.py` | `metro_seeds/summary.csv` | — (not in the paper) |
| `main_PHM.tex`: nonfunctional diagnostics (Fig. 8–9, Table 6) | `make_nonfunctional_figures.py --seed-dirs metro_seeds/seed*` | `metro/metro_nonfunctional_r2.pdf`, `metro_nonfunctional_trajectories.pdf`, `table_nonfunctional.tex`, `nonfunctional_summary.json`, `nonfunctional_per_variable.csv` | — (new) |
| Rolling-origin check of the Metro forecasts | `run_metro_rolling.py --seed s --out metro_rolling/seed$s` | `metro_rolling/seed*/rolling.csv`, `summary.json`, `metro_rolling_origin.pdf` | — (not in the paper) |

`notebooks/` holds Mobina's original notebooks unchanged (with their saved outputs);
`notebooks/saved_results/` holds the spreadsheets that were in `Simulation/`, including the
`run8-*.xlsx` files whose means/SDs are the entries of Tables 2–3.

## Settings: manuscript vs. implementation

### Simulation (Section 3, Appendix) — `hybridae.simstudy.PaperSimCfg`

| | Manuscript | Implementation |
|---|---|---|
| Curves | N = 500, J = 100 points on [0, 1], last L = 5 held out (T = 495) | same |
| Latent state | dimension 5; linear: stable VAR(1), A rescaled; nonlinear: A h + g(h) + d_t + ε | same (`simulate.py`, A rescaled to spectral radius ≤ 0.85) |
| Linear map | to 30 B-spline coefficients | 5 → 20 → 30 linear layers (seed 123), cubic B-splines |
| Noise | nonlinear σ_h = 0.028, σ_η = 0.001 | same |
| | linear: **not stated** | σ_h = 0.08, σ_η = 0.5 (linear notebook) |
| Fitted K, B | K ∈ {3,5,7}, B ∈ {10,…,50}, M(I) = M(O) = B | same |
| FPCA | first K PCs of the B-dim. basis representation of the training curves | least-squares cubic B-spline fit with B functions, centred PCA (trapezoid weights) |
| FAE | B–128–K / K–128–B, tanh, Adam lr 7e-4, batch 16, 1400 epochs, λ = 2e-4 | same, with biases; trained on the observed curves (two-stage) |
| | **not stated** | weight decay 1e-5; N(0, 0.06²) init (seed 743); chronological 85/15 split with best-validation checkpoint; curves centred and divided by their SD before training (on the raw scale the FAE fails to train) |
| Latent dynamics | VAR(1) for both methods | least squares (ridge 1e-6) |
| Metric | relMSE vs. the noise-free curves (eq. relmse) | same; relMSE vs. the noisy curves also stored |
| Replicates | R = 100 | data seed 30000 + r, r = 1..100 |

`--legacy-notebook` runs the amplitude–phase notebook procedure instead (clean training target,
latent-VAR penalty `dyn_lambda = 8`, 45 basis functions).

### U.S. fertility (Section 4.1)

| | Manuscript | Implementation (`run_fertility.py`) |
|---|---|---|
| Data | 1933–2023, train 1933–2018, test 2019–2023 | same (`data/USAfertility.csv`, ages 13–54, rates per woman) |
| Pre-processing | **not stated** (the s = 50 spline-smoothing paragraph is commented out) | none; centred by the training-year mean; ages rescaled to [0, 1]. `--presmooth 50 --center all` = notebook |
| Basis | 25 cubic B-splines in and out, "selected by GCV" | 25; the GCV curve is reported (`gcv` in the JSON; GCV selects 16 on the raw curves) |
| Network | 25–100–3 / 3–100–25, ReLU hidden, linear latent/output; bias sentence deleted | same, with biases (`--no-bias` = notebook) |
| Training | MSE + λ = 0.01 second-difference penalty; Adam 3e-4, wd 1e-6, batch 16, 5000 epochs; N(0, 0.5²) init, seed 743; random 85/15 split | same |
| VAR | "selected by AIC, which yielded q = 1"; VAR(1) used | VAR(1); AIC/BIC/HQ orders reported (`var_orders`) |
| FPCA | "leading score sequence", K **not stated** | K = 3 (matched) and K = 5 (notebook) |
| Metric | relMSE over the held-out curves | relMSE of the centred curves (`fore_relMSE`) and of the original curves |

### Metro traffic (Section 4.2) — `hybridae.train.MetroCfg`

| | Manuscript | Implementation (`run_metro.py`) |
|---|---|---|
| Data | 1,214 complete days, last 7 (2018-09-24..30) held out; 24 hourly values + 49 variables (Table 4) | same |
| Pre-processing | log(1+x), separate standardization with pre-forecast statistics | same |
| Network | 16 fixed cubic B-spline features; 65–128–128–5 / 5–128–128–65, GELU + LayerNorm, linear outputs | same; plus LayerNorm on the 16 features and on the 65-dim input |
| Loss | eq. (hae_loss): MSE_F + γ MSE_N + λ P; "squared second-difference smoothness penalties" | γ = 0.7, λ = 2e-4 (coefficients) **and** 1.5e-3 × squared second differences of the latent states of consecutive days |
| Optimizer | "Adam", lr 6.5e-4, wd 2.5e-5, batch 16, ≤ 900 epochs, last 15 % chronological validation | **AdamW** (notebook, produced the paper numbers; `--optimizer adam` available); unshuffled batches; gradient clipping 5; best checkpoint after epoch 50, early stopping (patience 180); seed 776 |
| VAR | AIC, q = 7 | AIC with q ≤ 21 on standardized latent scores |
| HPCA | PCA of the concatenated standardized data, matched K and VAR | PCA (5 components), AIC-VAR (q ≤ 21) |
| Weather-only numbers | "contribution" to the nonfunctional MSE | `contrib_weather_{recon,fore}` = Σ_{29 weather vars} MSE_j / 49 |

`--functional-only` also fits the functional-only FAE/FPCA pair of the notebook (no longer in the
paper; note its 24→16 feature layer is learned, not fixed).

## Typical run times

Fertility ≈ 2 min; Metro ≈ 1–5 min per seed; rolling-origin ≈ 1–5 min per seed (1 CPU thread);
simulation ≈ 15–30 s per FAE fit, i.e. 2 × 100 × 15 fits ≈ 15–25 CPU-hours
(`--workers 56` on a 72-core server: ≈ 2 h).
