# HAE paper — reproduction scripts

*Hybrid Autoencoders with Shared Temporal Latent Dynamics for Forecasting*
(Pourmoshir, Haghbin & Maadooliat). Manuscript: Overleaf/Dropbox `Mobina - FAE/HAE/main_HM.tex`.

## Script → paper mapping

| Paper item | Script | Output | Status |
|---|---|---|---|
| Table 2–3, Fig. `combined_boxplots` (simulation) | `run_simulation.py --grid --reps 100`, then `summarize_simulation.py` | `results/simulation/*.csv` | **does not reproduce** – see audit §1 |
| Sec. 4.1, Fig. `fertility_forecasts` (U.S. fertility) | `run_fertility.py` | `results/fertility/` | notebook output reproduced exactly; paper numbers do not match it (audit §2) |
| Table `traffic_hybrid`, Fig. `hae_vs_hpca`, FAE vs FPCA, weather-only numbers (Metro) | `run_metro.py` | `results/metro/` | HPCA/FPCA exact; HAE/FAE close (audit §3) |
| Fig. `usafertility`, `weekday_weekend_traffic_curves` (data plots) | `notebooks/` (original) | – | not yet ported |

## Settings (as in the original notebooks)

* **Fertility**: smoothing spline (`s = 50`) per year; curves centred by the mean over **all**
  years (`--center train` gives the look-ahead-free version); ages rescaled to [0,1];
  25 cubic B-splines for input and output; encoder 25–100–3, decoder 3–100–25, ReLU, no biases,
  N(0, 0.5²) initialization; Adam (lr 3e-4, wd 1e-6), batch 16, λ = 0.01, 5000 epochs, seed 743;
  random 85/15 split of 1933–2018; ridge VAR(1) on the latents; FPCA with K = 5 (also K = 3).
* **Metro**: complete days (1,214), last 7 held out; log1p + standardization of the 24 hourly
  values and the 49 scalars using pre-forecast days only; 16 fixed B-spline features → LayerNorm
  → concat with y → LayerNorm → 65–128–128–5 (GELU + LayerNorm) → 5–128–128–65; AdamW
  (lr 6.5e-4, wd 2.5e-5), batch 16 unshuffled, gradient clip 5, loss
  MSE_F + 0.7·MSE_N + 2e-4·(coef. 2nd-difference) + 1.5e-3·(latent 2nd-difference across
  consecutive days), chronological 15 % validation, best checkpoint after epoch 50, patience 180;
  AIC-selected VAR(q ≤ 21) on standardized latent scores; seed 776.
* **Simulation**: see `src/hybridae/simulate.py` and `simstudy.py`.

`notebooks/` holds Mobina's original notebooks unchanged (with their saved outputs);
`notebooks/saved_results/` holds the spreadsheets that were in `Simulation/`.
