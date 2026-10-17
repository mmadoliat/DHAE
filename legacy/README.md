# Legacy material

Kept for provenance only; nothing here is needed to reproduce either paper.

* `fae-exploratory/` – early FAE notebooks (Australian fertility, El Niño, linear FAE vs PCA,
  Hyndman–Ullah FPCA, an early hybrid "TS data" notebook). They read data files that are **not**
  in this repository (`Australiafertility.csv`, `Dataset/ElNino_ERSST*.csv`).
* `vae-mnist/VAE MNIST.ipynb` – the contents of the former `VAE/` folder: a PCA vs AE vs VAE
  tutorial on MNIST (latent dimension 2). It was the starting point for the variational
  encoder (mean / log-variance heads, reparameterization, KL term) now implemented for
  functional and hybrid data in `src/hybridae/models.py` (`VFAEVanilla`, `VariationalHybridAE`).
