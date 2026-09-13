"""Training loop matching the report's Fig. 2 methodology: dataset prep ->
forward pass -> rate-distortion loss -> backprop/update -> validation ->
checkpoint selection -> (separate) final testing.

Run from the repo root, e.g. (on Colab, after `pip install -r
requirements.txt` and fetching a dataset per src/data/download.py):

    python -m src.train --data_root /content/eurosat/2750 \
        --max_patch_size 64 --min_patch_size 16 --epochs 30 --lam 0.01

Batch size is effectively 1 image per forward pass (variable leaf count per
image -- see models/autoencoder.py docstring), with `--accum_steps`
gradient-accumulation batches for a stable update.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.datasets import SatelliteImageDataset, SyntheticSatelliteDataset
from models.autoencoder import AdaptiveSatelliteCodec
from losses.rate_distortion import rate_distortion_loss
from utils.metrics import psnr as psnr_np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default=None,
                    help="Folder of images; omit to train on synthetic data (smoke test).")
    p.add_argument("--max_patch_size", type=int, default=256)
    p.add_argument("--min_patch_size", type=int, default=32)
    p.add_argument("--split_threshold", type=float, default=0.12)
    p.add_argument("--embed_dim", type=int, default=256)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--num_heads", type=int, default=8)
    p.add_argument("--latent_dim", type=int, default=128)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lam", type=float, default=0.01,
                    help="Rate-distortion trade-off; sweep for multiple operating points.")
    p.add_argument("--ssim_weight", type=float, default=0.0)
    p.add_argument("--accum_steps", type=int, default=8)
    p.add_argument("--limit", type=int, default=None, help="Cap dataset size (debugging).")
    p.add_argument("--log_every", type=int, default=20,
                    help="Print a running-average log line every N images (0 to disable).")
    p.add_argument("--out_dir", type=str, default="checkpoints")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def build_dataset(args):
    if args.data_root:
        return SatelliteImageDataset(args.data_root, max_patch_size=args.max_patch_size,
                                      limit=args.limit)
    print("[train] --data_root not set: training on synthetic data (see src/data/download.py "
          "for how to fetch UC Merced / EuroSAT for real runs).")
    return SyntheticSatelliteDataset(n=args.limit or 64, size=args.max_patch_size * 2)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)

    dataset = build_dataset(args)
    # batch_size=1 at the DataLoader level: each item is a full image (variable
    # leaf count handled inside the model), collate_fn=identity avoids torch's
    # default collate trying (and failing) to stack variable-size outputs.
    loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=lambda x: x[0])

    model = AdaptiveSatelliteCodec(
        image_channels=3,
        max_patch_size=args.max_patch_size,
        min_patch_size=args.min_patch_size,
        split_threshold=args.split_threshold,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        latent_dim=args.latent_dim,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    history = []

    print(f"[train] dataset size: {len(dataset)} images "
          f"({'no --limit set: this can be slow on the full EuroSAT set (~27k images); '
             'pass --limit 100 or so for a first sanity run' if args.limit is None else 'limited'})")

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        running = {"loss": 0.0, "distortion": 0.0, "bpp": 0.0}
        n_seen = 0
        import time
        epoch_start = time.time()

        for step, image_np in enumerate(loader):
            out = model(image_np)
            target_leaves = torch.from_numpy(
                model.patcher.extract_patches(image_np, out["leaves"])
            ).float().to(device)

            loss, logs = rate_distortion_loss(
                out["recon_patches"], target_leaves, out["bits"], out["num_pixels"],
                lam=args.lam, ssim_weight=args.ssim_weight,
            )
            (loss / args.accum_steps).backward()

            if (step + 1) % args.accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                optimizer.zero_grad()

            for k in running:
                running[k] += logs[k]
            n_seen += 1

            if args.log_every and (step + 1) % args.log_every == 0:
                elapsed = time.time() - epoch_start
                rate = n_seen / elapsed
                remaining = len(dataset) - n_seen
                eta_min = (remaining / rate) / 60 if rate > 0 else float("nan")
                print(f"  [epoch {epoch}] {n_seen}/{len(dataset)} images "
                      f"({rate:.1f} img/s, ~{eta_min:.1f} min left this epoch) "
                      f"loss={running['loss']/n_seen:.4f}")

        avg = {k: v / max(n_seen, 1) for k, v in running.items()}
        history.append({"epoch": epoch, **avg})
        print(f"[epoch {epoch}] loss={avg['loss']:.4f} distortion={avg['distortion']:.5f} "
              f"bpp={avg['bpp']:.4f}")

        ckpt_path = os.path.join(args.out_dir, f"model_epoch{epoch}.pt")
        torch.save({"model_state": model.state_dict(), "args": vars(args)}, ckpt_path)

    with open(os.path.join(args.out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
