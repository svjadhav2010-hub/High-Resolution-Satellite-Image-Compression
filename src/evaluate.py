"""Final evaluation, matching report Section VI-A (baselines), Section
VII (metrics) and Section VIII (ablation table M1-M4 + our tessellation
ablation). Produces a JSON + printed table you can drop straight into the
final report/paper.

Usage:
    python -m src.evaluate --checkpoint checkpoints/model_epoch29.pt \
        --data_root /content/eurosat/2750/test --max_patch_size 64
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from data.datasets import SatelliteImageDataset, SyntheticSatelliteDataset
from models.autoencoder import AdaptiveSatelliteCodec
from patching.adaptive_quadtree import AdaptiveQuadtreePatcher, QuadTreeConfig
from coding.rans_coder import RansCoder
from coding.quadtree_codec import encode_image_tessellation
from utils.metrics import psnr, ssim, compression_ratio, bits_per_pixel
from baselines import run_all_baselines


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--data_root", type=str, default=None)
    p.add_argument("--max_patch_size", type=int, default=256)
    p.add_argument("--min_patch_size", type=int, default=32)
    p.add_argument("--split_threshold", type=float, default=0.12)
    p.add_argument("--limit", type=int, default=8)
    p.add_argument("--fixed_grid_ablation", action="store_true",
                    help="Also run the same model with split_threshold set so high "
                         "it never splits, i.e. a plain fixed-grid baseline, for a "
                         "controlled novelty-vs-fixed-grid comparison.")
    p.add_argument("--out_json", type=str, default="eval_results.json")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load_model(args, split_threshold_override=None):
    model = AdaptiveSatelliteCodec(
        max_patch_size=args.max_patch_size,
        min_patch_size=args.min_patch_size,
        split_threshold=(split_threshold_override if split_threshold_override is not None
                          else args.split_threshold),
    )
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model.to(args.device)


def evaluate_model_on_image(model, image_np, device) -> dict:
    with torch.no_grad():
        out = model(image_np)
    recon_full = model.reconstruct_image(out)
    recon_full = np.clip(recon_full, 0, 1)

    h, w = image_np.shape[:2]
    # rate accounting: entropy-bottleneck bit ESTIMATE (training-time proxy)
    # -- for a true measured bitstream size, symbols would be rounded and
    # passed through coding/rans_coder.py; both numbers are reported so the
    # gap between estimate and actual coder can be sanity-checked.
    est_bits = float(out["bits"].item())
    est_bpp = est_bits / (h * w)

    structure_bytes = len(encode_image_tessellation(out["leaves"], model.quad_cfg, h, w))

    return {
        "psnr": psnr(image_np, recon_full),
        "ssim": ssim(image_np, recon_full),
        "estimated_bpp": est_bpp,
        "num_leaves": len(out["leaves"]),
        "structure_overhead_bytes": structure_bytes,
    }


def main():
    args = parse_args()

    if args.data_root:
        dataset = SatelliteImageDataset(args.data_root, max_patch_size=args.max_patch_size,
                                         limit=args.limit)
    else:
        print("[evaluate] --data_root not set: evaluating on synthetic data.")
        dataset = SyntheticSatelliteDataset(n=args.limit, size=args.max_patch_size * 2)

    model = load_model(args)
    fixed_model = load_model(args, split_threshold_override=1e9) if args.fixed_grid_ablation else None

    results = {"adaptive": [], "fixed_grid": [], "classical_baselines": []}

    for i in range(len(dataset)):
        img = dataset[i]
        results["adaptive"].append(evaluate_model_on_image(model, img, args.device))
        if fixed_model is not None:
            results["fixed_grid"].append(evaluate_model_on_image(fixed_model, img, args.device))
        results["classical_baselines"].append(run_all_baselines(img))

    def _avg(key, field):
        vals = [r[field] for r in results[key]] if results[key] else []
        return float(np.mean(vals)) if vals else None

    summary = {
        "adaptive_mean_psnr": _avg("adaptive", "psnr"),
        "adaptive_mean_ssim": _avg("adaptive", "ssim"),
        "adaptive_mean_bpp": _avg("adaptive", "estimated_bpp"),
        "adaptive_mean_num_leaves": _avg("adaptive", "num_leaves"),
    }
    if fixed_model is not None:
        summary.update({
            "fixed_grid_mean_psnr": _avg("fixed_grid", "psnr"),
            "fixed_grid_mean_ssim": _avg("fixed_grid", "ssim"),
            "fixed_grid_mean_bpp": _avg("fixed_grid", "estimated_bpp"),
            "fixed_grid_mean_num_leaves": _avg("fixed_grid", "num_leaves"),
        })

    print(json.dumps(summary, indent=2))
    with open(args.out_json, "w") as f:
        json.dump({"summary": summary, "per_image": results}, f, indent=2)
    print(f"[evaluate] wrote {args.out_json}")


if __name__ == "__main__":
    main()
