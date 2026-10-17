# Reproducibility audit of the HAE paper (draft `main_HM.tex`, Dropbox version of 2026-10-14)

Audit date: 2026-10-17. Code audited: the notebooks in `HAE/notebooks/` (copied from
`mobinapourmoshir/Functional-Autoencoder`, commit `08061a8`; the private `mmadoliat/DHAE`
repository was not accessible and is assumed to hold the same files — **please confirm**).
Reruns: Linux, Python 3.12, torch 2.3.1 (CPU), statsmodels 0.15, scikit-learn 1.8. Nothing in
the manuscript was changed. Items are ordered by importance for submission.

## 1. Simulation study (Tables 2–3, Fig. `combined_boxplots`) — not reproducible from the repository

1. **Provenance of the reported numbers.** Every cell of Tables 2–3 equals the mean (SD) of
   the corresponding file `Simulation/simulated_excels/run8-<scenario>-b<B><B>-d<K>.xlsx`
   (100 rows each). No notebook in the repository writes files with this schema; the
   repository's runners write `sim_results_*.xlsx` with different columns.
2. **Inconsistent metadata in those files.** Their configuration columns read, e.g.,
   `T = 100`, `meas_noise_sd = 0.1`, `horizon = B`, `fae_n_rep = 100`,
   `fae_n_basis = K + 2`, none of which matches the design described in the paper
   (T = 495 + 5, noise 0.001, horizon 5, K latent, B basis functions).
3. **Implausible dependence structure.** Within a file, FPCA and FAE are supposedly fitted
   to the same simulated data set in each replicate, so their errors should be strongly
   positively correlated, and reconstruction and forecast errors should be correlated too.
   Observed correlations (nonlinear, K = 5, B = 30): FPCA vs FAE reconstruction r = 0.06;
   reconstruction vs forecast r ≈ 0.00; the same metric across the K = 3 and K = 5 files
   r ≈ 0.01–0.08 with different `data_seed` columns.
4. **The repository's own simulation code gives very different values.** With the
   amplitude–phase notebook code executed verbatim (data seed 30001), FPCA with K = 3 has
   reconstruction relMSE 0.0021 and forecast relMSE 0.0095 (paper: ≈ 0.12–0.13 and ≈ 0.34).
   Our port reproduces that notebook to machine precision. The notebook's own saved output
   for the same seed (`sim_results_nonlinear.xlsx`, row 1) shows 0.117 / 0.312, so the
   committed generator also differs from the one that produced its saved output. For the
   linear benchmark the committed runner's saved output has forecast relMSE ≈ 1–2
   (`sim_results_10runs_linear.xlsx`) versus 0.12–0.24 in Table 3.
5. **The FAE used in the amplitude–phase notebook differs from the paper's method.**
   (a) it is trained to reconstruct the *clean* curves from the noisy ones (an oracle not
   available to FPCA, which is fitted to the noisy curves); (b) every epoch includes a
   full-batch step on a latent VAR(1) residual penalty with weight `dyn_lambda = 8`, i.e. the
   latent dynamics are learned jointly with the autoencoder, whereas the paper states that
   the two stages are estimated separately; (c) architecture tanh, 128 hidden units, biases,
   K = 3, 45 basis functions; (d) the data-generating basis has 12 B-splines (paper: 30).
   `HAE/run_simulation.py --fae-mode twostage` runs the procedure as described in the paper.

**Recommendation:** confirm with Mobina how the `run8` files were produced; regenerate
Tables 2–3 with committed code (e.g. `python HAE/run_simulation.py --grid --reps 100
--fae-mode twostage`, ≈ 1–2 CPU-days) and update the simulation description accordingly.
A reduced check (3 replicates, K = 5, B = 30) is in `HAE/results/simulation/`.

## 2. U.S. fertility (Section 4.1)

* The notebook output is reproduced exactly: FPCA (K = 5) + VAR(1) forecast
  relRMSE 0.1949 (relMSE 0.0380); FAE (K = 3) + VAR(1) relRMSE 0.0761 (relMSE 0.0058),
  seed 743. Over seeds 743–747 FAE relMSE = 0.0072 (SD 0.0021).
* The paper reports relMSE 0.33 (FPCA) vs 0.13 (FAE). These values are not produced by
  the notebook on any scale we tried (relMSE, relRMSE, centered or original scale).
* FPCA uses K = 5 while FAE uses K = 3; with K = 3 FPCA gives relMSE 0.0416.
* Curves are centred with the mean over **all** years including 2019–2023 (look-ahead).
  `--center train` removes it.
* The paper says the VAR order was selected by AIC and equals 1; the code fixes VAR(1).
  AIC (statsmodels, maxlags 8) selects lag 6–8 for the FAE latents and 8 for FPCA scores.

## 3. Metro traffic (Table `traffic_hybrid`, Sec. 4.2)

* **Exact:** HPCA and FPCA (all reconstruction/forecast MSEs, VAR(12) for HPCA, weather-only
  HPCA numbers 0.4625 / 0.4704).
* **Close, not bit-identical** (float drift across torch builds; notebook epoch-1 losses match
  to all printed digits): HAE reconstruction 0.0650 / 0.0733 (paper 0.0664 / 0.0735),
  forecast 0.2948 / 0.3392 (paper 0.2831 / 0.3422); reductions vs HPCA 34.2 % / 17.0 % /
  26.0 % (paper 36.8 % / 16.3 % / 27.0 %). AIC chose VAR(14) instead of VAR(7).
* **Fragile claim:** functional-only FAE vs FPCA forecast on the standardized log scale:
  paper 0.1726 vs 0.2141 (−19.4 %); rerun 0.2360 vs 0.2141 (FAE worse by 10 %); on the
  count scale FAE is still better (−13.6 %). Multiple seeds are advisable before stating
  this comparison.
* **Undocumented details** that a reader cannot infer from the paper: AdamW (paper: Adam);
  nonfunctional loss weight 0.7; a latent second-difference penalty across consecutive
  (unshuffled) days, weight 1.5e-3, which is a temporal regulariser inside the first stage;
  LayerNorm on the B-spline features and on the concatenated input; VAR on standardized
  latent scores; HPCA uses VAR(12) while HAE uses VAR(7); the functional-only Metro "FAE"
  has a **learned** 24→16 feature layer, not the fixed basis layer of Section 2.1.

## 4. Smaller items
* Notation: encoder/decoder parameters are φ/ψ in Sec. 2.1 but φ/ϕ in Sec. 2.4; the
  appendix dynamics use z_{t-1}; σ_z vs σ_h; "Table 1" defines L as horizon while the
  appendix also uses it — fine, but q vs p for VAR order varies in comments.
* `\graphicspath{ {./HAE/Plots/} {.} }` plus explicit `HAE/Plots/...` paths — works on
  Overleaf only from the project root.

## 5. U.S. fertility smoothing (affects Section 4.1 and Fig. `fertility_forecasts`) — added 2026-10-17
`UnivariateSpline(ages, rates, s=50)` is applied to `USAfertility.csv`, whose rates are **per woman**
(max total sum of squares per year 0.69). With s = 50 ≥ the total sum of squares, every "smoothed"
curve is exactly the cubic least-squares polynomial in age (max difference 3e-15; the smoothed
91 × 42 matrix has rank 4; minimum value −0.037 at age 13). All fertility results (FPCA, FAE, and the
draft VHAE fertility section) are therefore computed on cubic polynomials, not on smoothed
fertility curves; this also explains why FPCA with K = 5 reconstructs the training curves exactly.
The value s = 50 presumably came from the Australian notebook (rates per 1,000); the per-woman
equivalent is s ≈ 5e-5. **Recommendation:** rerun the fertility application with a smoothing factor
appropriate to the data scale (e.g. GCV-chosen) before submission.
