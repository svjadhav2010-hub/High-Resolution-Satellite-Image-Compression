"""Run this FIRST on Colab/Kaggle, right after `pip install -r requirements.txt`,
before spending any real GPU time. Confirms the whole model wires together
correctly on synthetic data (no dataset download needed).

    python scripts/smoke_test.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

from data.datasets import SyntheticSatelliteDataset
from models.autoencoder import AdaptiveSatelliteCodec
from losses.rate_distortion import rate_distortion_loss


def main():
    print("=== Smoke test: AdaptiveSatelliteCodec ===")
    torch.manual_seed(0)

    ds = SyntheticSatelliteDataset(n=1, size=128)
    image_np = ds[0]
    print(f"[1/5] synthetic image: shape={image_np.shape}, dtype={image_np.dtype}, "
          f"range=({image_np.min():.3f}, {image_np.max():.3f})")

    model = AdaptiveSatelliteCodec(
        max_patch_size=128, min_patch_size=32, split_threshold=0.05,
        embed_dim=64, depth=2, num_heads=4, latent_dim=32,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[2/5] model built: {n_params:,} parameters")

    out = model(image_np)
    print(f"[3/5] forward pass ok: {len(out['leaves'])} adaptive leaves, "
          f"recon_patches shape={tuple(out['recon_patches'].shape)}, "
          f"bits={float(out['bits']):.1f}")

    target_leaves = torch.from_numpy(
        model.patcher.extract_patches(image_np, out["leaves"])
    ).float()
    loss, logs = rate_distortion_loss(
        out["recon_patches"], target_leaves, out["bits"], out["num_pixels"], lam=0.01,
    )
    assert torch.isfinite(loss), "loss is not finite!"
    print(f"[4/5] loss computed: {logs}")

    loss.backward()
    n_with_grad = sum(1 for p in model.parameters() if p.grad is not None)
    n_total = sum(1 for _ in model.parameters())
    print(f"[5/5] backward pass ok: {n_with_grad}/{n_total} parameters received gradients")

    recon_img = model.reconstruct_image(out)
    assert recon_img.shape[:2] == image_np.shape[:2], "reconstructed image shape mismatch!"

    print("\nAll smoke-test checks passed. Safe to proceed to real training "
          "(see src/data/download.py for fetching UC Merced / EuroSAT, then "
          "`python -m src.train --data_root <path> ...`).")


if __name__ == "__main__":
    main()
