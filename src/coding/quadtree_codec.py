"""Encodes/decodes the quadtree TESSELLATION STRUCTURE itself (which nodes
were split, down to which leaves) as a compact bitstream.

This is necessary metadata overhead the fixed-grid baseline does not have,
and its cost must be counted honestly in the total bitrate (see
docs/ARCHITECTURE.md Section 4.3 and the ablation table in
src/evaluate.py) -- otherwise the novelty's bitrate savings would be
overstated.

Encoding scheme: a pre-order traversal of the quadtree, emitting one bit
per *internal* node ("1" = split into 4 children, "0" = leaf). This is the
standard, provably near-optimal way to encode a quadtree shape and costs at
most 1 bit per node in the tree (internal + leaf nodes), which is tiny
next to the pixel bitstream -- e.g. a 256x256 image tessellated down to a
32x32 minimum has at most 1 + 4 + 16 + 64 = 85 nodes, i.e. <= 85 bits
(~11 bytes) of structural overhead.
"""

from __future__ import annotations

from typing import List, Tuple

from patching.adaptive_quadtree import QuadNode, QuadTreeConfig


class BitWriter:
    def __init__(self):
        self._bits: List[int] = []

    def write_bit(self, bit: int):
        self._bits.append(1 if bit else 0)

    def to_bytes(self) -> bytes:
        pad = (-len(self._bits)) % 8
        bits = self._bits + [0] * pad
        out = bytearray()
        for i in range(0, len(bits), 8):
            byte = 0
            for b in bits[i:i + 8]:
                byte = (byte << 1) | b
            out.append(byte)
        return bytes([pad]) + bytes(out)  # first byte = padding amount


class BitReader:
    def __init__(self, data: bytes):
        pad = data[0]
        payload = data[1:]
        bits = []
        for byte in payload:
            for i in range(7, -1, -1):
                bits.append((byte >> i) & 1)
        self._bits = bits[: len(bits) - pad] if pad else bits
        self._pos = 0

    def read_bit(self) -> int:
        b = self._bits[self._pos]
        self._pos += 1
        return b


def encode_image_tessellation(leaves: List[QuadNode], cfg: QuadTreeConfig,
                               image_h: int, image_w: int) -> bytes:
    writer = BitWriter()
    leaf_lookup = {(l.x, l.y, l.size) for l in leaves}

    def visit(x, y, size):
        if (x, y, size) in leaf_lookup:
            writer.write_bit(0)
        else:
            writer.write_bit(1)
            half = size // 2
            visit(x, y, half)
            visit(x + half, y, half)
            visit(x, y + half, half)
            visit(x + half, y + half, half)

    for gy in range(0, image_h, cfg.max_patch_size):
        for gx in range(0, image_w, cfg.max_patch_size):
            visit(gx, gy, cfg.max_patch_size)

    return writer.to_bytes()


def decode_image_tessellation(data: bytes, cfg: QuadTreeConfig,
                               image_h: int, image_w: int) -> List[Tuple[int, int, int]]:
    reader = BitReader(data)
    leaves: List[Tuple[int, int, int]] = []

    def visit(x, y, size):
        bit = reader.read_bit()
        if bit == 0:
            leaves.append((x, y, size))
        else:
            half = size // 2
            visit(x, y, half)
            visit(x + half, y, half)
            visit(x, y + half, half)
            visit(x + half, y + half, half)

    for gy in range(0, image_h, cfg.max_patch_size):
        for gx in range(0, image_w, cfg.max_patch_size):
            visit(gx, gy, cfg.max_patch_size)

    return leaves
