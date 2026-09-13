# Test Plan

## 1. Unit tests (run locally, no GPU, no dataset — `pytest tests/`)

| Test file | Covers |
|---|---|
| `tests/test_quadtree.py` | Flat regions stay coarse; complex regions split to the minimum size; leaves exactly tile the image (no gaps/overlaps); extract/reassemble shapes are correct; quadtree structure bitstream round-trips exactly; invalid configs are rejected. |
| `tests/test_coding.py` | rANS encode/decode round-trips exactly on random Gaussian-distributed symbols; compression actually reduces size vs. raw storage when the real `constriction` backend is available. |

Run: `python -m pytest tests/ -v`. All 8 tests pass with zero external dataset or GPU
dependency (verified during development — see chat history / CI log).

## 2. Integration smoke test (needs `torch`, no dataset — `scripts/smoke_test.py`)

Builds a tiny `AdaptiveSatelliteCodec`, runs one forward pass on a synthetic image,
checks: (a) no shape errors across variable leaf counts, (b) loss is finite, (c) a
backward pass produces gradients on every trainable parameter, (d) `reconstruct_image`
returns an array matching the input's shape. This is the first thing to run on Colab
after `pip install -r requirements.txt`, before spending GPU hours on real training.

## 3. Ablation matrix (Report Table I, extended with the tessellation axis)

| Config | Autoencoder | Transformer | Entropy bottleneck | rANS | Tessellation |
|---|---|---|---|---|---|
| M1 | Yes | No | No | No | Fixed grid |
| M2 | Yes | Yes | No | No | Fixed grid |
| M3 | Yes | Yes | Yes | No | Fixed grid |
| M4 (Phase I baseline) | Yes | Yes | Yes | Yes | Fixed grid |
| **M5 (this project's novelty)** | Yes | Yes | Yes | Yes | **Adaptive quadtree** |

M1–M4 isolate the contribution of each Phase-I-described component (transformer, entropy
model, real entropy coding). M4 vs M5 isolates the contribution of *this project's*
novelty on top of the Phase I architecture — the single most important comparison for
the viva. Run via:

```bash
python -m src.evaluate --checkpoint checkpoints/model.pt --fixed_grid_ablation \
    --data_root <test_set_folder>
```

which reports adaptive vs. fixed-grid PSNR/SSIM/bpp/leaf-count on the same images.

## 4. Rate-distortion sweep

Train multiple checkpoints varying `--lam` (e.g. 0.001, 0.005, 0.01, 0.05, 0.1) and plot
bpp (x-axis) vs. PSNR (y-axis) for both adaptive and fixed-grid configs on the same test
set — this produces the rate-distortion curve the Phase I report calls for (§VI-B,
§X) and is the clearest single plot for showing the novelty's benefit (or lack of one —
see docs/ETHICS.md §5 on reporting honestly either way).

## 5. Baseline comparison

`src/baselines.py::run_all_baselines` computes JPEG/PNG/JPEG2000 PSNR/SSIM/compression
ratio/bpp on the exact same test images used for the neural model, filling Report
Table II. JPEG quality and JPEG2000 quality-layers should be swept to bracket the
neural model's operating bitrate, not compared at one arbitrary quality setting.

## 6. Regression checklist before each report/paper update

- [ ] `pytest tests/` passes.
- [ ] `scripts/smoke_test.py` passes on the current model config.
- [ ] Latest checkpoint's `--fixed_grid_ablation` numbers are refreshed if the
      architecture or training recipe changed.
- [ ] Any new numbers quoted in the report/paper trace back to a file in
      `checkpoints/*/eval_results.json` (no numbers typed in from memory).
