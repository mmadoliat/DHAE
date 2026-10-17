"""Network architectures.

FAEVanilla / VFAEVanilla : fixed basis projection -> MLP encoder -> latent ->
    MLP decoder -> basis coefficients -> fixed basis recovery
    (fertility and simulation experiments).
HybridAE                  : Metro HAE (port of ``HybridFAEAE``).
MetroFAE                  : Metro functional-only autoencoder (port of
    ``FunctionalOnlyFAEAE``; note its feature layer is *learned*).
VariationalHybridAE       : VHAE; same as HybridAE but the encoder returns a
    Gaussian posterior q(h|x,y) = N(mu, diag(sigma^2)).
"""
from __future__ import annotations
from typing import List, Optional
import numpy as np
import torch
import torch.nn as nn
from .bases import bspline_basis_scipy, trapezoid_weights_torch

_ACT = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU, "identity": nn.Identity}


# ------------------------------------------------------------------ FAE
class FAEVanilla(nn.Module):
    """Two-hidden-layer FAE: B^(I) features -> fc1 -> act -> fc2 = h -> fc3 -> act -> fc4 = b."""
    variational = False

    def __init__(self, n_basis_project, n_rep, n_basis_revert, hidden: int = 100,
                 activation: str = "relu", bias: bool = False,
                 init_weight_sd: Optional[float] = None):
        super().__init__()
        self.fc1 = nn.Linear(n_basis_project, hidden, bias=bias)
        self.fc2 = nn.Linear(hidden, n_rep, bias=bias)
        self.fc3 = nn.Linear(n_rep, hidden, bias=bias)
        self.fc4 = nn.Linear(hidden, n_basis_revert, bias=bias)
        self.activation = _ACT[activation]()
        if init_weight_sd is not None:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0.0, std=init_weight_sd)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    @staticmethod
    def project(x, tpts, Bp):
        return (x * trapezoid_weights_torch(tpts).to(x.dtype)) @ Bp

    @staticmethod
    def revert(coef, Br):
        return coef @ Br.T

    def encode(self, feature):
        return self.fc2(self.activation(self.fc1(feature)))

    def decode_from_rep(self, rep, Br):
        coef = self.fc4(self.activation(self.fc3(rep)))
        return self.revert(coef, Br), coef

    def forward(self, x, tpts, Bp, Br):
        feature = self.project(x, tpts, Bp)
        rep = self.encode(feature)
        x_hat, coef = self.decode_from_rep(rep, Br)
        return x_hat, rep, feature, coef


class VFAEVanilla(FAEVanilla):
    """Variational FAE: fc1 -> act -> (fc_mu, fc_logvar); h = mu + sigma*eps (train) or mu (eval)."""
    variational = True

    def __init__(self, n_basis_project, n_rep, n_basis_revert, hidden: int = 100,
                 activation: str = "relu", bias: bool = False,
                 init_weight_sd: Optional[float] = None, logvar_init: float = -4.0):
        super().__init__(n_basis_project, n_rep, n_basis_revert, hidden, activation, bias, init_weight_sd)
        self.fc_logvar = nn.Linear(hidden, n_rep, bias=True)
        nn.init.normal_(self.fc_logvar.weight, 0.0, 1e-3)
        nn.init.constant_(self.fc_logvar.bias, logvar_init)
        self.last_mu = self.last_logvar = None

    def posterior(self, feature):
        t = self.activation(self.fc1(feature))
        return self.fc2(t), self.fc_logvar(t).clamp(-12.0, 6.0)

    def encode(self, feature):
        mu, logvar = self.posterior(feature)
        self.last_mu, self.last_logvar = mu, logvar
        if self.training:
            return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        return mu


# ------------------------------------------------------------------ Metro
def tuned_mlp(widths: List[int], activation=nn.GELU) -> nn.Sequential:
    """Linear layers with LayerNorm + activation between them (none after the last)."""
    layers: List[nn.Module] = []
    for i in range(len(widths) - 1):
        layers.append(nn.Linear(widths[i], widths[i + 1]))
        if i != len(widths) - 2:
            layers += [nn.LayerNorm(widths[i + 1]), activation()]
    return nn.Sequential(*layers)


class HybridAE(nn.Module):
    """HAE: fixed B-spline features of x, LayerNorm, concatenation with y, LayerNorm,
    shared MLP encoder -> h (K), shared MLP decoder -> [b (n_basis), y_hat (p)]."""
    variational = False

    def __init__(self, n_grid, n_nonfunctional, latent_dim, functional_feature_dim,
                 hidden_width, n_basis, basis: torch.Tensor, n_metadata: int = 0):
        super().__init__()
        self.n_basis, self.n_nonfunctional, self.n_metadata = n_basis, n_nonfunctional, n_metadata
        self.register_buffer("feature_basis", bspline_basis_scipy(np.arange(n_grid), functional_feature_dim))
        w = torch.ones(n_grid); w[0] = w[-1] = 0.5
        self.register_buffer("integration_weights", w / (n_grid - 1))
        d_in = functional_feature_dim + n_nonfunctional + n_metadata
        self.functional_feature_norm = nn.LayerNorm(functional_feature_dim)
        self.shared_input_norm = nn.LayerNorm(d_in)
        self._build_encoder(d_in, hidden_width, latent_dim)
        self.shared_decoder_to_coef_and_nonfunc = tuned_mlp(
            [latent_dim, hidden_width, hidden_width, n_basis + n_nonfunctional])
        self.register_buffer("basis", basis)

    def _build_encoder(self, d_in, hw, K):
        self.shared_latent_encoder = tuned_mlp([d_in, hw, hw, K])

    def _shared_input(self, x_func, y_nonfunc, m_meta=None):
        f = self.functional_feature_norm((x_func * self.integration_weights) @ self.feature_basis.T)
        parts = [f, y_nonfunc] + ([m_meta] if (m_meta is not None and self.n_metadata) else [])
        return self.shared_input_norm(torch.cat(parts, dim=1))

    def encode(self, x_func, y_nonfunc, m_meta=None):
        return self.shared_latent_encoder(self._shared_input(x_func, y_nonfunc, m_meta))

    def decode_latent(self, h):
        out = self.shared_decoder_to_coef_and_nonfunc(h)
        coef, y_hat = out[:, :self.n_basis], out[:, self.n_basis:]
        return coef @ self.basis, coef, y_hat

    def forward(self, x_func, y_nonfunc, m_meta=None):
        h = self.encode(x_func, y_nonfunc, m_meta)
        x_hat, coef, y_hat = self.decode_latent(h)
        return x_hat, y_hat, h, coef


class VariationalHybridAE(HybridAE):
    """VHAE: encoder trunk [d_in -> hw -> hw] (LayerNorm+GELU) with two linear heads
    giving mu_t and log sigma_t^2; decoder identical to HAE."""
    variational = True

    def __init__(self, *args, logvar_init: float = -4.0, **kw):
        self._logvar_init = logvar_init
        super().__init__(*args, **kw)
        self.last_mu = self.last_logvar = None

    def _build_encoder(self, d_in, hw, K):
        self.encoder_trunk = nn.Sequential(nn.Linear(d_in, hw), nn.LayerNorm(hw), nn.GELU(),
                                           nn.Linear(hw, hw), nn.LayerNorm(hw), nn.GELU())
        self.fc_mu = nn.Linear(hw, K)
        self.fc_logvar = nn.Linear(hw, K)
        nn.init.normal_(self.fc_logvar.weight, 0.0, 1e-3)
        nn.init.constant_(self.fc_logvar.bias, self._logvar_init)

    def posterior(self, x_func, y_nonfunc, m_meta=None):
        t = self.encoder_trunk(self._shared_input(x_func, y_nonfunc, m_meta))
        return self.fc_mu(t), self.fc_logvar(t).clamp(-12.0, 6.0)

    def encode(self, x_func, y_nonfunc, m_meta=None):
        mu, logvar = self.posterior(x_func, y_nonfunc, m_meta)
        self.last_mu, self.last_logvar = mu, logvar
        if self.training:
            return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        return mu


class MetroFAE(nn.Module):
    """Functional-only Metro autoencoder (port). NB: the 24->16 feature layer is learned."""
    variational = False

    def __init__(self, n_grid, latent_dim, functional_feature_dim, hidden_width, n_basis, basis):
        super().__init__()
        self.functional_feature_layer = tuned_mlp([n_grid, functional_feature_dim])
        self.functional_feature_norm = nn.LayerNorm(functional_feature_dim)
        self.latent_encoder = tuned_mlp([functional_feature_dim, hidden_width, hidden_width, latent_dim])
        self.decoder_to_coef = tuned_mlp([latent_dim, hidden_width, hidden_width, n_basis])
        self.register_buffer("basis", basis)

    def encode(self, x):
        return self.latent_encoder(self.functional_feature_norm(self.functional_feature_layer(x)))

    def decode_latent(self, h):
        coef = self.decoder_to_coef(h)
        return coef @ self.basis, coef

    def forward(self, x):
        h = self.encode(x)
        x_hat, coef = self.decode_latent(h)
        return x_hat, h, coef


def kl_standard_normal(mu, logvar):
    """Per-observation KL(N(mu, diag e^logvar) || N(0, I)), summed over latent dims."""
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
