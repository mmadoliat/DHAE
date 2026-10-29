# Reproducibility audit of the HAE paper

Manuscript audited: `Mobina - FAE/HAE/main_HM.tex` (Dropbox version of 2026-10-27).
Audit date: 2026-10-28 (supersedes the audit of 2026-10-17, whose open items are carried over).
Code: this repository, `HAE/` scripts and `src/hybridae/`, with the procedures implemented **as the
manuscript describes them** (see [`HAE/README.md`](HAE/README.md) for the setting-by-setting mapping
and the values used where the manuscript is silent). All numbers below were produced by the
committed scripts; result files are under `HAE/results/`.

Runs: Linux server (Python 3.9.21, torch 2.8.0+cpu, numpy 2.0.2, statsmodels 0.14.6, scikit-learn
1.6.1), single-threaded processes; data figures on macOS (Python 3.12). The manuscript text and
numbers were **not** changed (authors' decision); `main_PHM.tex` only adds the nonfunctional
diagnostics of §3.4 below.

## Summary

| Paper item | Reproduced? | Main reason |
|---|---|---|
| Sec. 3, Tables 2–3, Fig. 3 (simulation) | **No — conclusion reversed** | with the committed generator and the stated settings FPCA beats FAE on the nonlinear benchmark by 1–2 orders of magnitude (§1) |
| Sec. 4.1, relMSE 0.33 (FPCA) vs 0.13 (FAE), Fig. 5 | **No** | no pre-processing variant gives these values; with the stated settings FAE ≈ FPCA (K = 3) and FAE < FPCA (K = 5) (§2) |
| Sec. 4.1, "AIC yielded q = 1", "basis selected by GCV" (25) | **No** | AIC selects 8 (BIC 2–3); GCV selects 16 (§2) |
| Table 5, HPCA columns; weather-only HPCA contributions 0.2738 / 0.2784 | **Yes, exactly** | deterministic |
| Table 5, HAE columns | approximately | seed/platform dependent; 10-seed means in §3.2 |
| Sec. 4.2, nonfunctional forecast reduction 16.26 % | **not robust** | 10-seed mean 0.6 % (range −21.0 % to +10.5 %) (§3.2) |
| Sec. 4.2, VAR(7) selected by AIC for HAE | partly | AIC selects 7 for 4 of 10 seeds, 13–14 for the others |
| Fig. 7 (`hae_vs_hpca`) | **inconsistent with Table 5** | its curves correspond to an earlier run (functional MSE HPCA 0.250, HAE 0.221) (§3.3) |
| Figs. 4 and 6 (data) | yes | Fig. 4's axis label says "per 1,000 women"; the rates are per woman |

## 1. Simulation study (Section 3)

`python HAE/run_simulation.py --reps 100 --workers 56` then `python HAE/summarize_simulation.py`
(results: `HAE/results/simulation/`, 2 × 100 data sets × 15 (K, B) cells, FPCA and FAE fitted to the
same data set in every cell).

Mean relMSE (vs. the noise-free curves), K = 5:

| | B | FPCA recon | FAE recon | FPCA fore | FAE fore | paper FPCA / FAE recon | paper FPCA / FAE fore |
|---|---|---|---|---|---|---|---|
| Linear | 10 | 0.601 | 0.608 | 1.030 | 1.038 | 0.061 / 0.070 | 0.148 / 0.171 |
| Linear | 30 | 0.149 | 0.163 | 1.029 | 1.034 | 0.045 / 0.051 | 0.119 / 0.132 |
| Nonlinear | 10 | 0.021 | 0.039 | 0.019 | 0.027 | 0.126 / 0.074 | 0.338 / 0.148 |
| Nonlinear | 30 | 1.2e-4 | 0.014 | 0.0030 | 0.030 | 0.118 / 0.049 | 0.323 / 0.086 |
| Nonlinear | 50 | 1.2e-4 | 0.012 | 0.0030 | 0.034 | 0.120 / 0.054 | 0.327 / 0.094 |

Full tables in `table_recon.tex` / `table_forecast.tex` (paper layout), all cells in `summary.csv`,
boxplots in `combined_boxplots.pdf` (log axis for the nonlinear panels).

1. **Nonlinear benchmark.** With σ_η = 0.001 the amplitude–phase curves are almost exactly
   represented by 5 principal components of a B ≥ 20 spline representation (reconstruction relMSE
   1.2e-4, K = 3: 2.1e-3). The FAE (B–128–K, tanh, 1400 epochs, as stated) reaches 0.007–0.026 for B ≥ 20. The
   conclusion "FAE outperforms FPCA across all choices of K and B, often by a substantial margin"
   is the opposite of what the committed design produces.
2. **Linear benchmark.** FPCA and FAE are close: FPCA is better for K = 3 and for B ≤ 30 at K = 5;
   FAE is slightly better for B ≥ 40 at K = 5 and for most B at K = 7 (largest gap K = 7, B = 50:
   0.217 vs 0.245). Forecast relMSE ≈ 1.03 for both: five-step VAR(1) forecasts are no better than
   the mean of the held-out curves, partly because the relMSE denominator is the variance of only
   five reference curves. σ_h and σ_η of the linear benchmark are not stated in the paper (the
   linear notebook's 0.08 and 0.5 are used; noise is then ≈ 73 % of the variance of the observed curves).
3. **Provenance of Tables 2–3** (unchanged from the 2026-10-17 audit): every cell equals the mean
   (SD) of `HAE/notebooks/saved_results/simulated_excels/run8-*.xlsx`; no committed code writes
   files with that schema, their metadata columns contradict the stated design (T = 100, noise
   0.1, …), and paired FPCA/FAE errors in the same replicate are essentially uncorrelated
   (r ≈ 0.06), which is implausible for methods fitted to the same data.
4. Unstated FAE details needed to train at all: the curves must be centred and scaled (on the raw
   scale the tanh FAE does not train, reconstruction relMSE > 1); validation split / checkpoint,
   weight decay and initialization are the notebook's (README).

**Recommendation.** Either describe and commit the generator and settings that produced the
`run8` files, or replace Tables 2–3 / Fig. 3 by the regenerated results and revise the
simulation conclusions; a nonlinear benchmark in which a linear subspace is genuinely
inefficient (larger phase variation, higher noise) would be needed to support the current claim.

## 2. U.S. fertility (Section 4.1)

`python HAE/run_fertility.py --seeds 743 744 745 746 747` (stated settings: raw rates, 25
B-splines, biases, VAR(1); centring by the training-year mean) and
`python HAE/run_fertility.py --legacy-notebook` (s = 50 pre-smoothing, all-year centring, no biases).

| Forecast relMSE 2019–2023 (centred curves) | stated settings | notebook | paper |
|---|---|---|---|
| FPCA K = 3 | 0.072 | 0.042 | — |
| FPCA K = 5 | 0.025 | 0.038 | 0.33 |
| FAE K = 3 | 0.063 (SD 0.004, seeds 743–747; range 0.058–0.069) | 0.0058 | 0.13 |

* None of the variants gives 0.33 / 0.13 (also not on the original scale: FPCA 0.016–0.049,
  FAE 0.039–0.047 with the stated settings and 0.0025 with the notebook pipeline).
* The large FAE advantage exists only with the s = 50 pre-smoothing, which turns every year into
  its cubic least-squares polynomial (2026-10-17 audit §5); the paper no longer describes any
  pre-processing (the smoothing/centring paragraph is commented out).
* VAR order: AIC (statsmodels, maxlags 8) selects 8 for the FAE latents and the FPCA scores
  (BIC 2–3), not 1. GCV for the number of cubic B-splines (pooled over training years) selects 16;
  GCV(25)/GCV(16) = 1.32.
* FPCA dimension in Sec. 4.1 is not stated (notebook K = 5; matched K = 3 also reported).
* Fig. 4: the y-axis label "per 1,000 women" is wrong — `USAfertility.csv` contains births per
  woman (maximum 0.27). `HAE/make_data_figures.py` labels it accordingly.

## 3. Metro traffic (Section 4.2)

### 3.1 Data and HPCA — exact
1,214 complete days, 1,207 pre-forecast, p = 49. HPCA reconstruction/forecast MSEs of Table 5 and
the weather-only *contributions* 0.2738 / 0.2784 (= Σ over the 29 weather variables of MSE_j / 49)
are reproduced to all printed digits. Six of the 49 variables are constant over the pre-forecast
days (`snow_1h_{mean,min,max,std}`, `snow_any`, `weather_Thunderstorm`); `rain_1h_min` is non-zero
on 9 of 1,207 days.

### 3.2 HAE — seed and platform dependence
`HAE/run_metro.py --seed s` for s = 776–785, summarized by `HAE/summarize_metro_seeds.py`
(`HAE/results/metro_seeds/summary.csv`):

| | paper | seed 776 rerun | mean (SD) over 10 seeds | range | seeds better than HPCA |
|---|---|---|---|---|---|
| recon functional | 0.0664 | 0.0890 | 0.0793 (0.0114) | 0.061–0.097 | 10/10 |
| recon nonfunctional | 0.0735 | 0.1108 | 0.0904 (0.0161) | 0.060–0.111 | 10/10 |
| fore functional | 0.2831 | 0.2327 | 0.2938 (0.0800) | 0.191–0.419 | 10/10 |
| fore nonfunctional | 0.3422 | 0.3657 | 0.4062 (0.0412) | 0.366–0.494 | 6/10 |
| weather contribution, fore | 0.2679 | 0.2793 | 0.3025 (0.0553) | 0.240–0.432 | 2/10 |

* The reconstruction and functional-forecast advantages over HPCA hold for every seed. The
  nonfunctional forecast reduction (paper 16.26 %) averages 0.6 % (range −21.0 % to +10.5 %);
  the weather variables are forecast better than by HPCA for 2 of 10 seeds.
* The same seed gives different results across platforms (seed 776: paper/Windows 0.2831,
  macOS 0.2963, Linux 0.2327 for the functional forecast MSE).
* AIC (q ≤ 21) selects VAR(7) for 4 seeds and VAR(13–14) for 6.
* `--optimizer adam` (the paper says Adam; the notebook used AdamW): seed 776 gives 0.0786 / 0.1008
  / 0.2978 / 0.3645 (recon F / recon N / fore F / fore N), VAR(7) — of the same order.

### 3.3 Figure 7 does not correspond to Table 5
The HPCA curves of `Plots/hae_vs_hpca.pdf` differ from the deterministic HPCA forecasts behind
Table 5 (mean absolute differences 47–561 vehicles/h per day; e.g. Sat Sep 29 at noon: 4,321 in
the figure vs. 3,478). Decoding the figure's vector paths and mapping them back to the
standardized log scale gives functional forecast MSEs of **0.2496 (HPCA) and 0.2214 (HAE)** —
the values of the earlier analysis that is still commented out in the manuscript ("0.250 to
0.221"), not 0.4479 / 0.2831. `HAE/results/metro/hae_vs_hpca.pdf` is the figure consistent with
Table 5's HPCA (HAE from the seed-776 rerun).

### 3.4 Nonfunctional diagnostics (new; `main_PHM.tex`, Figs. 8–9 and Table 6)
`HAE/make_nonfunctional_figures.py --seed-dirs HAE/results/metro_seeds/seed*`:
per-variable reconstruction R² (median over seeds: weather continuous 0.91 vs HPCA 0.68, weather
indicators 0.95 vs 0.05, calendar 0.89 vs 0.34, holidays 0.98 vs 0.01); block decomposition of
the nonfunctional MSE (reconstruction −63 % to −98 % in every block; forecasting −22 % for calendar,
+8–9 % for the weather blocks); categorical recovery (modal weather condition 99.2 % vs 76.3 %,
majority share 42.1 %; holiday balanced accuracy 99.9 % vs 50.0 %).

### 3.5 Rolling-origin check (not in the paper)
`HAE/run_metro_rolling.py --seed s` (s = 776–780): HAE and HPCA estimated once on the days before
2018-06-30, latent VAR re-estimated at 13 weekly origins (the last one is the paper's test week).
Medians over origins (HAE: range over seeds):

| | HAE | HPCA | seasonal naive | origins with HAE < HPCA |
|---|---|---|---|---|
| functional MSE | 0.190–0.305 | 0.200 | 0.194 | 2–5 of 13 |
| weather contribution | 0.274–0.327 | 0.242 | 0.505 | 1–5 of 13 |
| calendar + holiday contribution | 0.082–0.110 | 0.102 | 0.028 | 4–8 of 13 |

The forecasting advantage of Table 5 is specific to the single test week; over 13 origins neither
HAE nor HPCA beats a same-weekday-last-week forecast of the traffic curves.
Means over origins are dominated by rare variables (`rain_1h_min`, named holidays) whose standardized
values reach > 100; medians are therefore reported.

### 3.6 Details a reader cannot infer from the paper (unchanged)
AdamW rather than Adam; γ = 0.7 and λ = 2e-4; an additional penalty 1.5e-3 × squared second
differences of the latent states of consecutive days (a temporal regularizer inside the first
stage, not part of eq. (hae_loss)); LayerNorm on the 16 features and on the concatenated input;
best checkpoint after epoch 50 with patience 180; unshuffled batches; gradient clipping; VAR fitted
to standardized latent scores; HPCA uses VAR(12).

## 4. Suggested text corrections (not applied)
1. Sec. 3: state σ_h, σ_η of the linear benchmark; state the curve standardization, validation split
   and weight decay of the FAE; reconcile Tables 2–3 / Fig. 3 with committed code (§1).
2. Sec. 4.1: describe the pre-processing actually used; state the FPCA dimension; correct "AIC
   yielded q = 1" and "selected by GCV" or give the criterion that yields them; fix the Fig. 4 unit.
3. Sec. 4.2: "Adam" → "AdamW"; give γ, λ and the latent smoothness penalty; replace Fig. 7 by the
   version consistent with Table 5; qualify the nonfunctional forecasting gain (§3.2, §3.5); note
   the six constant variables in Table 4.
