# VHAE paper — scripts

Variational hybrid autoencoder (VHAE): the HAE encoder is replaced by a Gaussian encoder
q(h_t | x_t, y_t) = N(μ_t, diag σ_t²) with a standard-normal prior, trained with a β-weighted
ELBO (KL warm-up). The VAR(q) is fitted to the posterior means μ_t; forecasts are Monte-Carlo
draws combining (i) posterior uncertainty about the last q latent states, (ii) VAR innovations
and (iii) observation noise estimated from reconstruction residuals. The functional-only
special case is called VFAE. Manuscript: Dropbox `Mobina - FAE/VHAE/main.tex`.

| Paper item | Script | Output |
|---|---|---|
| Metro table + figure (HPCA / HAE / VHAE, point & probabilistic) | `run_metro.py --seeds 776 777 778` | `results/metro/` |
| Fertility table + figure (FPCA / FAE / VFAE, β sensitivity) | `run_fertility.py --seeds 743 744 745` | `results/fertility/` |
| Simulation table (linear and amplitude–phase, K = 5, B = 30) | `run_simulation.py --scenario {linear,nonlinear} --reps R` | `results/simulation/` |

Probabilistic scores: empirical coverage and mean width of pointwise 90 % intervals, the
interval score and the sample CRPS (Gneiting & Raftery, 2007), all on the scale used for
the point errors.

The previous `VAE/` folder (an MNIST PCA/AE/VAE tutorial) is preserved in `legacy/vae-mnist/`.
