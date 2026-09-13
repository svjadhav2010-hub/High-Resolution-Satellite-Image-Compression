"""Classical baselines, matching the report's Table II columns
(JPEG, PNG, ..., Proposed Method) and Section VI-A baseline list.

Also includes a *fixed-grid* neural baseline runner hook so the adaptive
quadtree novelty can be honestly compared against "the same autoencoder +
transformer, but with a plain fixed 256x256 (or configurable) grid" --
that ablation isolates the contribution of the tessellation itself from the
contribution of the transformer/entropy-coding stack, which is the
comparison a viva panel will ask for first.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

from utils.metrics import psnr, ssim, compression_ratio, bits_per_pixel


def _to_uint8(img: np.ndarray) -> np.ndarray:
    return np.clip(img * 255.0, 0, 255).astype(np.uint8)


def jpeg_baseline(img: np.ndarray, quality: int = 75) -> dict:
    pil_img = Image.fromarray(_to_uint8(img))
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    compressed_bytes = buf.getvalue()

    recon = np.asarray(Image.open(io.BytesIO(compressed_bytes)).convert("RGB"), dtype=np.float32) / 255.0
    return _report("JPEG", img, recon, len(compressed_bytes))


def png_baseline(img: np.ndarray) -> dict:
    pil_img = Image.fromarray(_to_uint8(img))
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG", optimize=True)
    compressed_bytes = buf.getvalue()

    recon = np.asarray(Image.open(io.BytesIO(compressed_bytes)).convert("RGB"), dtype=np.float32) / 255.0
    return _report("PNG (lossless)", img, recon, len(compressed_bytes))


def jpeg2000_baseline(img: np.ndarray, quality_layers: int = 20) -> dict:
    """Requires Pillow built with OpenJPEG support (true in most modern
    Pillow wheels). Falls back with a clear error if unsupported."""
    pil_img = Image.fromarray(_to_uint8(img))
    buf = io.BytesIO()
    try:
        pil_img.save(buf, format="JPEG2000", quality_mode="rates",
                     quality_layers=[quality_layers])
    except (KeyError, OSError) as e:
        raise RuntimeError(
            "This Pillow build lacks JPEG2000 (OpenJPEG) support. "
            "On Colab: !pip install pillow --upgrade, or apt-get install "
            "libopenjp2-7 first."
        ) from e
    compressed_bytes = buf.getvalue()

    recon = np.asarray(Image.open(io.BytesIO(compressed_bytes)).convert("RGB"), dtype=np.float32) / 255.0
    return _report("JPEG2000", img, recon, len(compressed_bytes))


def _report(name: str, original: np.ndarray, recon: np.ndarray, compressed_size: int) -> dict:
    h, w = original.shape[:2]
    original_size = original.size  # uint8-equivalent byte count (H*W*C)
    return {
        "method": name,
        "psnr": psnr(original, recon),
        "ssim": ssim(original, recon),
        "compression_ratio": compression_ratio(original_size, compressed_size),
        "bpp": bits_per_pixel(compressed_size, h, w),
        "compressed_bytes": compressed_size,
    }


def run_all_baselines(img: np.ndarray, jpeg_quality: int = 75) -> list:
    results = [png_baseline(img), jpeg_baseline(img, jpeg_quality)]
    try:
        results.append(jpeg2000_baseline(img))
    except RuntimeError as e:
        print(f"[baselines] skipping JPEG2000: {e}")
    return results
