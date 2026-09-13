"""Encoder: patch embedding -> scale/position embedding -> transformer ->
downsample to latent.

Consumes the output of `AdaptiveQuadtreePatcher.extract_patches`: every leaf
has already been resampled to a common (min_patch_size x min_patch_size)
tensor, so a standard patch-embedding Conv2d works unchanged. What differs
from a fixed-grid pipeline is the *scale embedding* (src/models/
transformer_blocks.py::ScaleEmbedding), added per-token so the transformer
knows which tokens represent large, homogeneous regions of the image versus
small, complex ones.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .transformer_blocks import TransformerBlock, SinusoidalPositionalEmbedding, ScaleEmbedding


class PatchEmbed(nn.Module):
    def __init__(self, patch_size: int, in_channels: int, embed_dim: int):
        super().__init__()
        # patch_size here is min_patch_size (all leaves are resampled to it),
        # embedded with a single conv that covers the whole leaf -> 1 token.
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size,
                               stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N_leaves, C, H, W) -> (N_leaves, embed_dim)
        return self.proj(x).flatten(1)


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        min_patch_size: int = 32,
        in_channels: int = 3,
        embed_dim: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        latent_dim: int = 128,
        possible_leaf_sizes: tuple = (32, 64, 128, 256),
        max_grid_positions: int = 1024,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim

        self.patch_embed = PatchEmbed(min_patch_size, in_channels, embed_dim)
        self.scale_embed = ScaleEmbedding(embed_dim, possible_leaf_sizes)
        self.pos_embed = SinusoidalPositionalEmbedding(embed_dim, max_grid_positions)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout=dropout) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        self.to_latent = nn.Linear(embed_dim, latent_dim)

    def forward(self, leaf_patches: torch.Tensor, leaf_sizes: torch.Tensor,
                grid_positions: torch.Tensor) -> torch.Tensor:
        """
        leaf_patches: (N, C, H, W) all leaves for one image (or a batch
            flattened across images -- see autoencoder.py for how batches
            with a variable number of leaves per image are handled).
        leaf_sizes: (N,) long tensor, native pixel size of each leaf.
        grid_positions: (N,) long tensor, a flattened row-major index of the
            leaf's top-left corner on a coarse grid (used for sinusoidal
            positional embedding so the transformer knows *where* in the
            image each token sits, independent of its size).

        Returns latent tokens of shape (N, latent_dim), one per leaf.
        """
        tokens = self.patch_embed(leaf_patches)               # (N, embed_dim)
        tokens = tokens + self.scale_embed(leaf_sizes)         # + scale info
        tokens = tokens + self.pos_embed(grid_positions)       # + position

        x = tokens.unsqueeze(0)  # treat the whole image's leaves as one sequence: (1, N, D)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x).squeeze(0)  # (N, embed_dim)

        z = self.to_latent(x)  # (N, latent_dim)
        return z
