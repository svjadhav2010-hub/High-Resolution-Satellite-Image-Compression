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
):
    """Returns (total_loss, logs_dict). `lam` trades rate vs distortion --
    sweep this (e.g. 0.001, 0.005, 0.01, 0.05, 0.1) to trace the
    rate-distortion curve requested in the report's Section VIII (Table II
    / Fig. rate-distortion curves)."""
    recon_flat = recon_leaves.permute(0, 3, 1, 2)
    target_flat = target_leaves.permute(0, 3, 1, 2) if target_leaves.dim() == 4 else target_leaves

    distortion = mse_distortion(recon_flat, target_flat)

    if ssim_weight > 0:
        from utils.metrics import ssim_torch
        ssim_val = ssim_torch(recon_flat, target_flat)
        distortion = (1 - ssim_weight) * distortion + ssim_weight * (1 - ssim_val)

    bpp = bits / num_pixels
    loss = lam * distortion + bpp  # bits are the "natural" units of R; scale distortion instead

    return loss, {
        "loss": float(loss.detach()),
        "distortion": float(distortion.detach()),
        "bpp": float(bpp.detach()),
        "bits": float(bits.detach()),
    }
