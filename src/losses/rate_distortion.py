"""Rate-distortion training objective, matching the report's Eq. 16/19:
    L = R + lambda * D

Distortion combines MSE (pixel fidelity, matches Eq. 18) with an optional
MS-SSIM term (perceptual/structural fidelity) since the project's own
evaluation plan weighs both PSNR and SSIM -- optimizing for MSE alone tends
to under-serve SSIM.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def mse_distortion(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(recon, target)


def rate_distortion_loss(
    recon_leaves: torch.Tensor,       # (N, h, w, C) decoder output, differentiable
    target_leaves: torch.Tensor,      # (N, h, w, C) ground-truth leaf pixels
    bits: torch.Tensor,               # scalar, from EntropyBottleneck
    num_pixels: int,
    lam: float = 0.01,
    ssim_weight: float = 0.0,
    distortion_scale: float = 255.0 ** 2,
):
    """Returns (total_loss, logs_dict). `lam` trades rate vs distortion --
    sweep this (e.g. 0.001, 0.005, 0.01, 0.05, 0.1) to trace the
    rate-distortion curve requested in the report's Section VIII (Table II
    / Fig. rate-distortion curves).

    IMPORTANT scale note: images in this codebase are stored as [0, 1]
    floats, so raw MSE here is tiny (~1e-2 to 1e-3). Standard rate-distortion
    literature (Balle et al. 2018 and everything the Phase I report cites
    that reuses their lambda convention) computes MSE on 0-255-scale pixels,
    which is why `lam` values like 0.001-0.1 are meaningful there. We
    replicate that convention internally via `distortion_scale = 255**2` so
    the same lambda values behave the same way here -- without this, `lam`
    is effectively ~65000x too weak, the rate term dominates the gradient,
    and training collapses to near-zero bitrate with distortion barely
    improving (symptom: bpp shrinks every epoch, distortion stays flat).
    """
    recon_flat = recon_leaves.permute(0, 3, 1, 2)
    target_flat = target_leaves.permute(0, 3, 1, 2) if target_leaves.dim() == 4 else target_leaves

    distortion = mse_distortion(recon_flat, target_flat)
    scaled_distortion = distortion * distortion_scale

    if ssim_weight > 0:
        from utils.metrics import ssim_torch
        ssim_val = ssim_torch(recon_flat, target_flat)
        scaled_distortion = (1 - ssim_weight) * scaled_distortion + ssim_weight * (1 - ssim_val)

    bpp = bits / num_pixels
    loss = lam * scaled_distortion + bpp

    return loss, {
        "loss": float(loss.detach()),
        "distortion": float(distortion.detach()),          # raw [0,1]-scale MSE, for eyeballing
        "scaled_distortion": float(scaled_distortion.detach()),  # what lam actually multiplies
        "bpp": float(bpp.detach()),
        "bits": float(bits.detach()),
    }