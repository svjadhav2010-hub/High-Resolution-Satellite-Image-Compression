"""Reusable transformer components.

Kept deliberately small/standard (pre-LN transformer encoder blocks with
multi-head self-attention + GELU MLP) so the *novelty* stays where it
belongs -- the adaptive tessellation and variable-rate allocation -- rather
than in a bespoke attention variant that would be hard to defend/justify in
a viva. This mirrors the architecture described in the project's own report
(Section 3.2.1: "Transformer encoder uses multi-head self-attention").
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, attn_dropout: float = 0.0,
                 proj_dropout: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # 3, B, heads, N, head_dim
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(b, n, c)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden = int(dim * hidden_ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: float = 4.0,
                 dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, num_heads, dropout, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio, dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class SinusoidalPositionalEmbedding(nn.Module):
    """Standard fixed sinusoidal embedding, used for the patch's (x, y)
    location within the image grid."""

    def __init__(self, dim: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        return self.pe[indices]


class ScaleEmbedding(nn.Module):
    """Learned embedding keyed on the *native quadtree leaf size* (32, 64,
    128, 256, ...). This is what lets a single transformer encoder make
    sense of patches that came from very different tessellation depths --
    it tells the model "this token represents a 128x128 region of the
    original image", which matters both for reconstruction scale and for
    the model to learn to allocate more latent capacity to finer (smaller,
    more complex) leaves. Required because of the adaptive patcher; a
    fixed-grid pipeline would not need this module at all."""

    def __init__(self, dim: int, possible_sizes: tuple = (32, 64, 128, 256, 512)):
        super().__init__()
        self.size_to_idx = {s: i for i, s in enumerate(possible_sizes)}
        self.embed = nn.Embedding(len(possible_sizes), dim)

    def forward(self, sizes: torch.Tensor) -> torch.Tensor:
        # sizes: LongTensor of raw pixel sizes (e.g. 32, 128, ...)
        idx = torch.tensor(
            [self.size_to_idx[int(s)] for s in sizes.tolist()],
            device=sizes.device, dtype=torch.long,
        )
        return self.embed(idx)
