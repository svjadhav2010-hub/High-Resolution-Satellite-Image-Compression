"""Decoder: latent tokens -> transformer refinement -> per-leaf pixel patch
at the common min_patch_size resolution (the adaptive patcher's
`reassemble()` then upsamples each leaf back to its native size and stitches
the full image, with seam feathering)."""

from __future__ import annotations

import torch
import torch.nn as nn

from .transformer_blocks import TransformerBlock, SinusoidalPositionalEmbedding, ScaleEmbedding


class PatchUnembed(nn.Module):
    def __init__(self, patch_size: int, out_channels: int, embed_dim: int):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.ConvTranspose2d(embed_dim, out_channels, kernel_size=patch_size,
                                        stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, embed_dim) -> (N, embed_dim, 1, 1) -> (N, C, patch, patch)
        x = x.unsqueeze(-1).unsqueeze(-1)
        return self.proj(x)


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        min_patch_size: int = 32,
        out_channels: int = 3,
        embed_dim: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        latent_dim: int = 128,
        possible_leaf_sizes: tuple = (32, 64, 128, 256),
        max_grid_positions: int = 1024,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.from_latent = nn.Linear(latent_dim, embed_dim)
        self.scale_embed = ScaleEmbedding(embed_dim, possible_leaf_sizes)
        self.pos_embed = SinusoidalPositionalEmbedding(embed_dim, max_grid_positions)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout=dropout) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.patch_unembed = PatchUnembed(min_patch_size, out_channels, embed_dim)

    def forward(self, z_hat: torch.Tensor, leaf_sizes: torch.Tensor,
                grid_positions: torch.Tensor) -> torch.Tensor:
        x = self.from_latent(z_hat)
        x = x + self.scale_embed(leaf_sizes)
        x = x + self.pos_embed(grid_positions)

        x = x.unsqueeze(0)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x).squeeze(0)

        patches = self.patch_unembed(x)  # (N, C, min_patch, min_patch)
        return torch.sigmoid(patches)    # pixel values in [0, 1]
