"""Folder-based dataset loader.

Expects a directory of images (any layout scikit-image/Pillow can open --
this matches how UC Merced Land Use Dataset and EuroSAT are typically
distributed: class-named subfolders of .tif/.jpg/.png files). Because the
adaptive quadtree patcher requires image dimensions to be a multiple of
`max_patch_size`, images are center-cropped/padded on load.

Download note: this sandbox has no access to the dataset hosts (UC Merced /
EuroSAT are not on the allowed domain list here), so `download.py` documents
the exact steps to fetch them yourself (Colab/Kaggle both have one-line
downloads for both). Point `SatelliteImageDataset` at the resulting local
folder.
"""

from __future__ import annotations

import glob
import os
from typing import List, Optional

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


def _pad_to_multiple(img: np.ndarray, multiple: int) -> np.ndarray:
    h, w = img.shape[:2]
    new_h = ((h + multiple - 1) // multiple) * multiple
    new_w = ((w + multiple - 1) // multiple) * multiple
    if new_h == h and new_w == w:
        return img
    pad_h, pad_w = new_h - h, new_w - w
    return np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")


class SatelliteImageDataset(Dataset):
    def __init__(self, root: str, max_patch_size: int = 256,
                 extensions: tuple = (".tif", ".tiff", ".jpg", ".jpeg", ".png"),
                 limit: Optional[int] = None):
        self.root = root
        self.max_patch_size = max_patch_size
        self.paths: List[str] = []
        for ext in extensions:
            self.paths.extend(glob.glob(os.path.join(root, "**", f"*{ext}"), recursive=True))
        self.paths.sort()
        if limit:
            self.paths = self.paths[:limit]
        if not self.paths:
            raise FileNotFoundError(
                f"No images with extensions {extensions} found under {root}. "
                "See src/data/download.py for how to fetch UC Merced / EuroSAT."
            )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int) -> np.ndarray:
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = _pad_to_multiple(arr, self.max_patch_size)
        return arr  # HxWx3 float32 in [0, 1]


class SyntheticSatelliteDataset(Dataset):
    """Dependency-free synthetic dataset for CI/tests and smoke tests when
    no real dataset is available. Generates scenes with a mix of flat
    regions and textured/edge regions so the adaptive patcher has something
    meaningful to react to."""

    def __init__(self, n: int = 8, size: int = 256, seed: int = 0):
        self.n = n
        self.size = size
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.n

    def __getitem__(self, idx: int) -> np.ndarray:
        rng = np.random.default_rng(idx)
        s = self.size
        img = np.full((s, s, 3), rng.uniform(0.1, 0.4), dtype=np.float32)

        # a couple of "urban" high-frequency blocks
        for _ in range(rng.integers(1, 4)):
            bs = int(rng.choice([32, 64, 96]))
            y = int(rng.integers(0, s - bs))
            x = int(rng.integers(0, s - bs))
            img[y:y + bs, x:x + bs] = rng.random((bs, bs, 3)).astype(np.float32)

        # a sharp linear "road"
        w = 4
        y0 = int(rng.integers(0, s - w))
        img[y0:y0 + w, :, :] = 0.9
        return np.clip(img, 0, 1)
