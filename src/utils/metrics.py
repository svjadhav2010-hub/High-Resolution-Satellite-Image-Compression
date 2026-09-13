"""Metrics matching exactly what the report's Section VII defines:
compression ratio (Eq. 21), bitrate/bpp (Eq. 22), MSE/PSNR (Eq. 23/24), and
SSIM (Wang et al. 2004, as cited in the report's Section VII-D)."""

from __future__ import annotations

import numpy as np

try:
    from skimage.metrics import peak_signal_noise_ratio as _sk_psnr
    from skimage.metrics import structural_similarity as _sk_ssim
    _HAS_SKIMAGE = True
except ImportError:  # pragma: no cover
    _HAS_SKIMAGE = False


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))


def psnr(a: np.ndarray, b: np.ndarray, data_range: float = 1.0) -> float:
    if _HAS_SKIMAGE:
        return float(_sk_psnr(a, b, data_range=data_range))
    m = mse(a, b)
    if m == 0:
        return float("inf")
    return 10 * np.log10((data_range ** 2) / m)


def ssim(a: np.ndarray, b: np.ndarray, data_range: float = 1.0) -> float:
    if _HAS_SKIMAGE:
        channel_axis = -1 if a.ndim == 3 else None
        return float(_sk_ssim(a, b, data_range=data_range, channel_axis=channel_axis))
    raise RuntimeError("scikit-image is required for SSIM; pip install scikit-image")


def compression_ratio(original_bytes: int, compressed_bytes: int) -> float:
    if compressed_bytes == 0:
        return float("inf")
    return original_bytes / compressed_bytes


def bits_per_pixel(compressed_bytes: int, height: int, width: int) -> float:
    return (8.0 * compressed_bytes) / (height * width)


def ssim_torch(recon, target):
    """Lightweight differentiable SSIM approximation for use inside the
    training loss (single-scale, fixed Gaussian window). For final reported
    numbers use the skimage `ssim()` above on denormalized uint8 images, not
    this training-time approximation."""
    import torch
    import torch.nn.functional as F

    def gaussian_window(size=11, sigma=1.5, channels=3, device="cpu"):
        coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = (g / g.sum()).unsqueeze(0)
        window_2d = g.t() @ g
        return window_2d.expand(channels, 1, size, size).contiguous()

    c = recon.shape[1]
    window = gaussian_window(channels=c, device=recon.device)
    mu1 = F.conv2d(recon, window, padding=5, groups=c)
    mu2 = F.conv2d(target, window, padding=5, groups=c)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2

    sigma1_sq = F.conv2d(recon * recon, window, padding=5, groups=c) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=5, groups=c) - mu2_sq
    sigma12 = F.conv2d(recon * target, window, padding=5, groups=c) - mu1_mu2

    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return ssim_map.mean()
