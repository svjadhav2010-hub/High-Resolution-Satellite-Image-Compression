import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from patching.adaptive_quadtree import AdaptiveQuadtreePatcher, QuadTreeConfig
from coding.quadtree_codec import encode_image_tessellation, decode_image_tessellation


def make_test_image(seed=0):
    rng = np.random.default_rng(seed)
    img = np.full((256, 256, 3), 0.2, dtype=np.float32)
    img[100:180, 40:160] = rng.random((80, 120, 3)).astype(np.float32)
    return img


def test_flat_image_stays_coarse():
    img = np.full((256, 256, 3), 0.5, dtype=np.float32)
    cfg = QuadTreeConfig(max_patch_size=256, min_patch_size=32, split_threshold=0.05)
    patcher = AdaptiveQuadtreePatcher(cfg)
    leaves = patcher.build(img)
    assert len(leaves) == 1
    assert leaves[0].size == 256


def test_complex_region_splits_to_min_size():
    img = make_test_image()
    cfg = QuadTreeConfig(max_patch_size=256, min_patch_size=32, split_threshold=0.02)
    patcher = AdaptiveQuadtreePatcher(cfg)
    leaves = patcher.build(img)
    sizes = {l.size for l in leaves}
    assert 32 in sizes, "complex region should have split down to the minimum patch size"
    assert 256 not in sizes or len(leaves) > 1


def test_leaves_tile_the_image_exactly():
    img = make_test_image()
    cfg = QuadTreeConfig(max_patch_size=256, min_patch_size=32, split_threshold=0.03)
    patcher = AdaptiveQuadtreePatcher(cfg)
    leaves = patcher.build(img)
    total_area = sum(l.size * l.size for l in leaves)
    assert total_area == img.shape[0] * img.shape[1]


def test_extract_and_reassemble_shapes():
    img = make_test_image()
    cfg = QuadTreeConfig(max_patch_size=256, min_patch_size=32, split_threshold=0.03)
    patcher = AdaptiveQuadtreePatcher(cfg)
    leaves = patcher.build(img)
    patches = patcher.extract_patches(img, leaves)
    assert patches.shape == (len(leaves), 32, 32, 3)

    recon = patcher.reassemble(leaves, patches, img.shape)
    assert recon.shape == img.shape


def test_quadtree_structure_roundtrip():
    img = make_test_image()
    cfg = QuadTreeConfig(max_patch_size=256, min_patch_size=32, split_threshold=0.02)
    patcher = AdaptiveQuadtreePatcher(cfg)
    leaves = patcher.build(img)

    encoded = encode_image_tessellation(leaves, cfg, 256, 256)
    decoded = decode_image_tessellation(encoded, cfg, 256, 256)

    orig = sorted((l.x, l.y, l.size) for l in leaves)
    dec = sorted(decoded)
    assert orig == dec


def test_config_rejects_non_power_of_two_ratio():
    with pytest.raises(AssertionError):
        QuadTreeConfig(max_patch_size=256, min_patch_size=48)
