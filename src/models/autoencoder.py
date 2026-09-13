"""End-to-end model: image -> adaptive quadtree leaves -> transformer
encoder -> entropy bottleneck (variable-rate) -> transformer decoder ->
reassembled image.

This module operates on ONE image at a time at the neural-network level
(a "batch" of leaves, all belonging to a single image, forms one attention
sequence) because the number of leaves varies per image under the adaptive
tessellation -- that is the whole point of the novelty. Training therefore
uses batch size 1 image with gradient accumulation, OR images are padded to
a common leaf count (see src/train.py for both options; gradient
accumulation is the default since it needs no padding logic and satellite
scenes vary a lot in local complexity).
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch
import torch.nn as nn

from patching.adaptive_quadtree import AdaptiveQuadtreePatcher, QuadTreeConfig, QuadNode
from .encoder import TransformerEncoder
from .decoder import TransformerDecoder
from .entropy_bottleneck import EntropyBottleneck


class AdaptiveSatelliteCodec(nn.Module):
    def __init__(
        self,
        image_channels: int = 3,
        max_patch_size: int = 256,
        min_patch_size: int = 32,
        split_threshold: float = 0.12,
        embed_dim: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        latent_dim: int = 128,
    ):
        super().__init__()
        self.quad_cfg = QuadTreeConfig(
            max_patch_size=max_patch_size,
            min_patch_size=min_patch_size,
            split_threshold=split_threshold,
        )
        self.patcher = AdaptiveQuadtreePatcher(self.quad_cfg)

        possible_sizes = tuple(
            min_patch_size * (2 ** i)
            for i in range(int(np.log2(max_patch_size // min_patch_size)) + 1)
        )

        self.encoder = TransformerEncoder(
            min_patch_size=min_patch_size, in_channels=image_channels,
            embed_dim=embed_dim, depth=depth, num_heads=num_heads,
            latent_dim=latent_dim, possible_leaf_sizes=possible_sizes,
        )
        self.entropy_bottleneck = EntropyBottleneck(latent_dim)
        self.decoder = TransformerDecoder(
            min_patch_size=min_patch_size, out_channels=image_channels,
            embed_dim=embed_dim, depth=depth, num_heads=num_heads,
            latent_dim=latent_dim, possible_leaf_sizes=possible_sizes,
        )
        self.grid_stride = min_patch_size  # coarse grid unit for positional index

    # ------------------------------------------------------------------
    def _leaves_to_tensors(self, image_np: np.ndarray, leaves: List[QuadNode], device):
        patches_np = self.patcher.extract_patches(image_np, leaves)     # (N, h, w, C)
        patches = torch.from_numpy(patches_np).float().permute(0, 3, 1, 2).to(device)
        sizes = torch.tensor([l.size for l in leaves], dtype=torch.long, device=device)
        complexities = torch.tensor([l.complexity for l in leaves], dtype=torch.float32,
                                     device=device)
        max_w = image_np.shape[1]
        cols = max_w // self.grid_stride
        grid_pos = torch.tensor(
            [(l.y // self.grid_stride) * cols + (l.x // self.grid_stride) for l in leaves],
            dtype=torch.long, device=device,
        )
        return patches, sizes, complexities, grid_pos

    def forward(self, image_np: np.ndarray):
        """
        image_np: HxWxC float32 numpy array in [0, 1]. H, W must be
        multiples of max_patch_size.

        Returns a dict with the reconstructed image, estimated bits, the
        quadtree leaves (for inspection/visualization), and the number of
        pixels (for bits-per-pixel reporting).
        """
        device = next(self.parameters()).device
        leaves = self.patcher.build(image_np)

        patches, sizes, complexities, grid_pos = self._leaves_to_tensors(image_np, leaves, device)

        z = self.encoder(patches, sizes, grid_pos)                       # (N, latent_dim)
        z_seq = z.t().unsqueeze(0)                                       # (1, latent_dim, N)
        z_hat_seq, bits, likelihoods = self.entropy_bottleneck(z_seq, complexity_weight=complexities)
        z_hat = z_hat_seq.squeeze(0).t()                                 # (N, latent_dim)

        recon_patches = self.decoder(z_hat, sizes, grid_pos)             # (N, C, h, w)
        recon_patches_np_ready = recon_patches.permute(0, 2, 3, 1)       # (N, h, w, C)

        h, w = image_np.shape[:2]
        num_pixels = h * w

        return {
            "leaves": leaves,
            "recon_patches": recon_patches_np_ready,   # torch tensor, differentiable
            "bits": bits,
            "num_pixels": num_pixels,
            "z": z,
            "z_hat": z_hat,
        }

    def reconstruct_image(self, out: dict) -> np.ndarray:
        """Convenience: stitch the decoder's leaf patches back into a full
        image using the adaptive patcher's feathered reassembly. Detaches
        from autograd -- use for eval/visualization, not for the loss."""
        recon_np = out["recon_patches"].detach().cpu().numpy()
        leaves = out["leaves"]
        h = max(l.y + l.size for l in leaves)
        w = max(l.x + l.size for l in leaves)
        c = recon_np.shape[-1]
        return self.patcher.reassemble(leaves, recon_np, (h, w, c))
