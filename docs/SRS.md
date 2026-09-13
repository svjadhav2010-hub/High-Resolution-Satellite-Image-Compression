# Software Requirements Specification (SRS)
## High-Resolution Satellite Image Compression — Content-Adaptive Neural Codec

**Team:** Swayam Jadhav, Rakesh More, Om Chumbhale, Pranav Chitalkar
**Guide:** Prof. N. S. Gaware — K. K. Wagh Institute of Engineering Education & Research
**Document status:** Phase II (implementation) — builds on the Phase I seminar report and paper draft.

---

## 1. Purpose

Phase I established the problem (satellite data volume vs. storage/bandwidth constraints),
surveyed 12+ prior works, and specified an autoencoder + transformer compression
architecture with a fixed-grid patch pipeline. This document specifies the Phase II
system: a full, testable, runnable implementation, plus one architectural novelty —
**content-adaptive quadtree patch tessellation with variable bit allocation** — that
directly addresses a limitation shared by every fixed-grid baseline in the Phase I
literature survey.

## 2. Scope

In scope:
- End-to-end neural codec: adaptive patcher → transformer encoder → learned entropy
  bottleneck (variable rate) → transformer decoder → seam-aware reassembly.
- rANS entropy coding of latent symbols (real bitstream, not just an estimate) and a
  compact codec for the quadtree structure itself.
- Training pipeline with a rate-distortion objective and checkpointing.
- Evaluation pipeline reproducing the report's metrics (PSNR, SSIM, compression ratio,
  bitrate, rate-distortion curves) plus an ablation isolating the novelty's contribution.
- Classical baselines (JPEG, PNG, JPEG2000) for the comparison table.

Out of scope (explicitly, per Phase I report Section 2.3.2):
- SAR / thermal imagery.
- On-board satellite hardware deployment (architecture is designed with it in mind —
  see docs/ARCHITECTURE.md §5 — but flight-qualification is future work).
- Multispectral (>3 band) input in the first working version; the data pipeline is
  written so adding channels is a config change, not a rewrite (see `image_channels`
  in `AdaptiveSatelliteCodec`).

## 3. Functional Requirements

| ID | Requirement |
|----|-------------|
| FR1 | The system SHALL decompose an input image into variable-size patches based on local content complexity, bounded by configurable max/min patch sizes. |
| FR2 | The system SHALL encode each patch into a latent representation using a transformer-based encoder that is aware of each patch's native size and image-grid position. |
| FR3 | The system SHALL quantize latent representations and estimate/compute their coding rate via a learned entropy model, with an optional per-patch rate multiplier derived from that patch's complexity score. |
| FR4 | The system SHALL losslessly encode quantized latents into a binary bitstream using rANS, and losslessly decode them back. |
| FR5 | The system SHALL losslessly encode/decode the quadtree tessellation structure itself, accounted for in total bitrate. |
| FR6 | The system SHALL reconstruct a full image from decoded patches, including seam blending across tessellation boundaries. |
| FR7 | The system SHALL report PSNR, SSIM, compression ratio, and bits-per-pixel for both the proposed method and classical baselines (JPEG/PNG/JPEG2000) on the same test images. |
| FR8 | The system SHALL support an ablation mode that disables adaptivity (forces a fixed grid) so the novelty's contribution can be isolated. |
| FR9 | The system SHALL be trainable end-to-end via a rate-distortion loss with a configurable rate/distortion trade-off (λ), supporting multiple operating points. |

## 4. Non-Functional Requirements

| ID | Requirement |
|----|-------------|
| NFR1 | **Reproducibility** — all randomness seeded; config saved alongside every checkpoint. |
| NFR2 | **Testability** — core logic (tessellation, structural codec, entropy coding) covered by unit tests that run without a GPU. |
| NFR3 | **Portability** — training/inference runs on free-tier Colab/Kaggle GPUs (single mid-range GPU, ~12–16 GB VRAM, session time limits); no requirement on multi-GPU. |
| NFR4 | **Modularity** — patcher, encoder, entropy model, decoder, and coding stage are independently swappable/testable modules (see docs/ARCHITECTURE.md). |
| NFR5 | **Honesty of measurement** — bitrate figures must include quadtree structural overhead; ablations must compare against a fixed-grid version of the *same* trained architecture, not just cite external numbers. |
| NFR6 | **Extensibility** — adding a spectral band or a new baseline codec must not require touching the training loop. |

## 5. Constraints

- Compute budget: free-tier Colab/Kaggle GPU with limited session hours → drives the
  choice of EuroSAT (small 64×64 images) as the primary training set, with UC Merced
  (256×256) used for a smaller-scale generalization check, and modest model sizes
  (`embed_dim`, `depth` are config knobs specifically so the team can trade capacity for
  training time).
- No access to onboard satellite hardware for real deployment testing (per Phase I
  Assumptions/Scope) — computational-cost reporting stands in for this (docs/TEST_PLAN.md).

## 6. Stakeholders & Users

- **Project team / evaluators**: need a working, testable, explainable system for the
  final viva — every design choice here traces back to a specific gap named in the
  team's own Phase I literature survey (see docs/ARCHITECTURE.md §1).
- **Downstream users (simulated)**: GIS analysts / agricultural monitoring pipelines
  that consume reconstructed imagery — motivates keeping SSIM and downstream-task
  fidelity in the loss and eval, not just PSNR.

## 7. Acceptance Criteria

1. `pytest tests/` passes (tessellation, structural codec, entropy coding — no GPU needed).
2. `python -m src.train` runs to completion on synthetic data with no dataset present
   (CI/smoke-test path) and on a real dataset on Colab.
3. `python -m src.evaluate --fixed_grid_ablation` produces a table showing adaptive vs.
   fixed-grid PSNR/SSIM/bpp on the same images.
4. Compression ratio / PSNR / SSIM / bpp are reported for the proposed method and for
   JPEG/PNG/JPEG2000 on the same test images (fills Report Table II).
