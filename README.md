# DHAE — functional, hybrid and variational hybrid autoencoders for forecasting

Code, data and reproduction scripts for two papers by Mobina Pourmoshir, Hossein Haghbin and
Mehdi Maadooliat:

| Paper | Status | Folder |
|---|---|---|
| **HAE** — *Hybrid Autoencoders with Shared Temporal Latent Dynamics for Forecasting* | near submission | [`HAE/`](HAE/README.md) |
| **VHAE** — variational extension (probabilistic latent states and forecasts) | draft | [`VHAE/`](VHAE/README.md) |

Project website: <https://mmadoliat.github.io/DHAE> (source in [`docs/`](docs/)).

## What the methods do

Each time point carries a curve $X_t(\cdot)$ observed on a grid and, for hybrid data, a vector
$\mathbf y_t$ of scalar measurements. A fixed basis layer turns the curve into features
$\mathbf f_t$; a neural encoder maps $[\mathbf f_t;\mathbf y_t]$ to a low-dimensional latent state
$\mathbf h_t$; a VAR($q$) model forecasts $\mathbf h_{T+1},\dots,\mathbf h_{T+L}$; a shared decoder
returns B-spline coefficients (hence smooth curves) and the scalar part.

* **FAE / HAE**: deterministic encoder; point forecasts.
* **VFAE / VHAE**: Gaussian encoder $q(\mathbf h_t\mid \mathbf x_t,\mathbf y_t)$ trained with a
  $\beta$-weighted ELBO; Monte-Carlo predictive distributions for future curves and scalars.

## Repository layout

```
src/hybridae/        shared library (bases, models, training, VAR forecasting, baselines, metrics, data)
HAE/                 scripts reproducing the HAE paper + original notebooks (HAE/notebooks/)
VHAE/                scripts for the VHAE paper
data/                USAfertility.csv, metro_interstate_traffic_volume.zip (UCI, CC BY 4.0)
docs/                project website (GitHub Pages)
legacy/              exploratory notebooks and the former VAE/ MNIST tutorial
REPRODUCIBILITY_AUDIT.md   what reproduces, what does not, and why
```

## Installation

```bash
conda env create -f environment.yml      # creates/updates env "py39" and installs this package
conda activate py39
# or: pip install -r requirements.txt && pip install -e .
```

The scripts also run without installation (they add `src/` to the path).

## Quick start

```bash
python HAE/run_fertility.py                              # Sec. 4.1, ~2 min CPU
python HAE/run_metro.py                                  # Sec. 4.2, Table 5 / Fig. 7, 1-5 min CPU
python HAE/make_nonfunctional_figures.py                 # nonfunctional diagnostics (main_PHM.tex)
python HAE/make_data_figures.py                          # Figs. 4 and 6
python HAE/run_simulation.py --reps 100 --workers 56     # Sec. 3 (15-25 CPU-hours in total)
python HAE/summarize_simulation.py                       # Tables 2-3, Fig. 3
python VHAE/run_metro.py --seeds 776 777 778
python VHAE/run_fertility.py
```

Set `HYBRIDAE_MAX_EPOCHS=3` for a quick smoke test of any script. Every script writes its
numbers to `*/results/...` (JSON/CSV) together with the software versions used.
[`HAE/README.md`](HAE/README.md) maps every script to the paper item it produces and lists the
settings stated in the manuscript next to those used where the manuscript is silent.

## Reproducibility notes

The HAE scripts implement the procedures as the manuscript (`main_HM.tex`, 2026-10-27) describes
them. Linear baselines (FPCA, HPCA) reproduce exactly; autoencoder results depend on the seed and
on the platform (torch build), so Metro results are also reported over ten training seeds.
Several numbers in the manuscript are **not** reproduced by the committed code (simulation
Tables 2-3, fertility relMSEs, Fig. 7) — please read
[`REPRODUCIBILITY_AUDIT.md`](REPRODUCIBILITY_AUDIT.md) before citing them.

## License

TODO: choose a license (e.g. MIT for code, CC BY 4.0 for text). The Metro data are from the UCI
Machine Learning Repository (Hogue, 2019; CC BY 4.0).
