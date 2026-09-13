# High-Resolution Satellite Image Compression — Content-Adaptive Neural Codec

Final year B.Tech project (K. K. Wagh Institute of Engineering Education & Research,
Dept. of Computer Engineering). Builds on the Phase I seminar report and research paper
draft (autoencoder + transformer + entropy coding for satellite imagery) with one
concrete novelty: **content-adaptive quadtree patch tessellation with variable bit
allocation**, replacing the fixed patch grid used in every reviewed prior work.

## Why this novelty

Every method in the Phase I literature survey (Zhu 2021, Guerrisi 2022/2023,
Zhang 2023, Wen & Li 2025, ...) tessellates images with a fixed grid. Satellite scenes
are extremely heterogeneous (uniform ocean/cropland next to dense urban/coastal detail),
so a fixed grid spends identical bits everywhere. This project instead recursively
splits the image only where local complexity is high, down to a minimum patch size, and
additionally lets the entropy model quantize complex regions more finely than flat ones.
Full rationale: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Repo layout

```
docs/                  SRS, architecture rationale, ethics, test plan (SDLC artifacts)
src/
  patching/            adaptive_quadtree.py -- THE NOVELTY
  models/              transformer encoder/decoder, entropy bottleneck, full autoencoder
  coding/              rANS entropy coder + quadtree-structure codec (real bitstreams)
  losses/              rate-distortion training objective
  data/                dataset loader + download instructions (UC Merced / EuroSAT)
  utils/               PSNR / SSIM / compression-ratio / bpp metrics
  baselines.py         JPEG / PNG / JPEG2000 comparison baselines
  train.py             training loop
  evaluate.py          evaluation + fixed-grid ablation + baseline comparison
tests/                 unit tests (no GPU / dataset required)
scripts/smoke_test.py  first thing to run on Colab -- validates the pipeline in seconds
notebooks/             ready-to-run Colab notebook
```

## Quickstart

**Locally (this repo, no GPU needed) — verify the novelty logic and coding correctness:**
```bash
pip install -r requirements.txt
pytest tests/ -v
```
8/8 tests pass without any dataset or GPU: the adaptive tessellation (flat regions stay
coarse, complex regions split to the minimum size, leaves exactly tile the image with no
gaps/overlap, the tree shape round-trips through its bitstream encoding exactly) and the
rANS entropy coder round-trip are both verified this way.

**On Colab/Kaggle (free-tier GPU) — train and evaluate the full model:**
Open `notebooks/colab_train.ipynb`, or run by hand:
```bash
pip install -r requirements.txt
python scripts/smoke_test.py        # sanity check, ~10s, no dataset needed
# fetch EuroSAT — see src/data/download.py for the exact commands
python -m src.train --data_root <path/to/eurosat/2750> \
    --max_patch_size 64 --min_patch_size 16 --epochs 10 --lam 0.01
python -m src.evaluate --checkpoint checkpoints/model_epoch9.pt \
    --data_root <path> --fixed_grid_ablation
```

## Compute plan

Training targets **free-tier Colab/Kaggle GPU hours** specifically: EuroSAT (small,
64×64 images) is the primary dataset, model width/depth are config knobs so capacity can
be traded for training time, and gradient accumulation avoids needing large batches.
UC Merced (256×256, matches the Phase I report exactly) is for a smaller-scale
generalization check once the EuroSAT pipeline is validated.

## Status

- [x] Novelty module (adaptive quadtree + variable rate) implemented and unit-tested.
- [x] Full model (transformer encoder/decoder, entropy bottleneck, rANS coding) implemented.
- [x] Training loop, evaluation + ablation harness, classical baselines implemented.
- [x] SDLC docs (SRS, architecture, ethics, test plan) written.
- [ ] Real training run on Colab (needs GPU — this dev sandbox has none available).
- [ ] Rate-distortion sweep across λ values, final numbers for the report/paper.
- [ ] Downstream-task (classification/segmentation) fidelity evaluation.

See `docs/SRS.md` §7 for acceptance criteria and `docs/TEST_PLAN.md` for the full
verification plan including the ablation matrix that isolates this project's novelty
contribution from the Phase I architecture it builds on.
