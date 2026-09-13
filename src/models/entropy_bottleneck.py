"""Learned entropy bottleneck.

Implements the standard Balle et al. (2018) factorized entropy model that
the project's report cites (Eq. 17: R = -sum log2 p(z_hat)). During
training, quantization is approximated by adding uniform noise U(-0.5, 0.5)
so gradients can flow (the classic straight-through alternative -- round()
with a stop-gradient -- is used at eval/inference time so the reported
bitrate matches what rANS will actually encode).

We also expose a *per-leaf* rate multiplier derived from the quadtree leaf's
complexity score. This is the "variable bit allocation" half of the
novelty: two leaves with identical latent statistics but different
complexity scores are allowed different effective quantization step sizes,
so the model can legitimately spend more bits on complex leaves and fewer
on flat ones instead of only relying on the tessellation itself.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FactorizedEntropyModel(nn.Module):
    """A small MLP-based CDF model per latent channel, following the
    factorized prior of Balle et al. 2018 (simplified, single-layer variant
    -- sufficient for a project-scale implementation; swap in
    `compressai.entropy_models.EntropyBottleneck` for a production-grade,
    battle-tested version, see docs/ARCHITECTURE.md)."""

    def __init__(self, channels: int, init_scale: float = 10.0, filters=(3, 3, 3)):
        super().__init__()
        self.channels = channels
        self._filters = (1,) + tuple(filters) + (1,)
        self._matrices = nn.ParameterList()
        self._biases = nn.ParameterList()
        self._factors = nn.ParameterList()

        scale = init_scale ** (1.0 / (len(self._filters) - 1))
        for i in range(len(self._filters) - 1):
            in_f, out_f = self._filters[i], self._filters[i + 1]
            mat = torch.eye(out_f, in_f).unsqueeze(0).repeat(channels, 1, 1)
            mat = mat + torch.randn_like(mat) * 0.01
            self._matrices.append(nn.Parameter(mat * scale))
            self._biases.append(nn.Parameter(torch.zeros(channels, out_f, 1)))
            if i < len(self._filters) - 2:
                self._factors.append(nn.Parameter(torch.zeros(channels, out_f, 1)))

    def _logits_cumulative(self, x: torch.Tensor) -> torch.Tensor:
        # x: (channels, 1, N)
        for i in range(len(self._matrices)):
            mat = torch.nn.functional.softplus(self._matrices[i])
            x = torch.matmul(mat, x) + self._biases[i]
            if i < len(self._factors):
                x = x + torch.tanh(self._factors[i]) * torch.tanh(x)
        return x

    def likelihood(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, C, L) quantized (or noisy) latent. Returns per-element
        probability mass under the learned model (used for the rate term)."""
        b, c, l = z.shape
        z = z.permute(1, 0, 2).reshape(c, 1, b * l)
        lower = self._logits_cumulative(z - 0.5)
        upper = self._logits_cumulative(z + 0.5)
        sign = -torch.sign(lower + upper).detach()
        p = torch.abs(torch.sigmoid(sign * upper) - torch.sigmoid(sign * lower))
        p = p.reshape(c, b, l).permute(1, 0, 2)
        return torch.clamp(p, min=1e-9)


class EntropyBottleneck(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.model = FactorizedEntropyModel(channels)

    def forward(self, z: torch.Tensor, complexity_weight: torch.Tensor | None = None):
        """
        z: (B, C, L) continuous latent, where L indexes quadtree LEAVES (one
            latent token per leaf) -- not the batch dimension. In this
            project's usage B is always 1 (one image = one attention
            sequence of leaves, see models/autoencoder.py), and L varies
            per image because the tessellation is content-adaptive.
        complexity_weight: optional (L,) tensor in (0, 1], one value per
            LEAF/token (not per batch element), from the quadtree's
            complexity score. Rescales the quantization noise (train) /
            step size (eval) per leaf so complex leaves keep finer
            granularity and flat leaves get coarser (cheaper) quantization.
            This is the variable-rate mechanism referenced in
            docs/ARCHITECTURE.md Section 4.2.

        Returns: z_hat (quantized/noisy latent), bits (scalar estimated
        total bits for this batch), likelihoods (for logging).
        """
        if complexity_weight is None:
            step = torch.ones(z.shape[-1], device=z.device)
        else:
            # map complexity in [0,1] -> step size in [0.5, 1.5]
            # low complexity (flat) -> larger step -> coarser -> fewer bits
            assert complexity_weight.shape[0] == z.shape[-1], (
                f"complexity_weight must have one entry per leaf/token (L={z.shape[-1]}), "
                f"got shape {tuple(complexity_weight.shape)}"
            )
            step = 1.5 - complexity_weight.clamp(0, 1)

        step_ = step.view(1, 1, -1)  # broadcast over (B, C, L) on the L axis

        if self.training:
            noise = (torch.rand_like(z) - 0.5) * step_
            z_hat = z + noise
        else:
            z_hat = torch.round(z / step_) * step_

        likelihoods = self.model.likelihood(z_hat / step_.clamp(min=1e-6))
        bits = -torch.log2(likelihoods).sum()
        return z_hat, bits, likelihoods
