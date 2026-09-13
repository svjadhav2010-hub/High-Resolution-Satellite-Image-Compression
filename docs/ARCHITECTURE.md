# Architecture & Design Rationale

## 1. Why this novelty, and why it's defensible

The Phase I literature survey (12 papers) shows every reviewed system — TIC (Zhu 2021),
the Φ-Sat-2 CAE (Guerrisi 2022/2023), HCCNet (Zhang 2023), NAS-designed autoencoders
(Wen & Li 2025), SS-JPEG-AI (Liu 2024), etc. — operates on a **fixed spatial grid** of
patches (or the whole image via convolutions, which is the continuous-grid limit of the
same idea). None of the twelve exploit the fact that satellite scenes are extremely
heterogeneous in local information content: a scene can be 80% ocean/desert/uniform
cropland and 20% urban/coastal/field-boundary detail, yet a fixed grid spends identical
representational budget everywhere.

**Content-Adaptive Quadtree Tessellation (CAQT)**, implemented in
`src/patching/adaptive_quadtree.py`, replaces the fixed grid with a recursive
split-if-complex decomposition, and `src/models/entropy_bottleneck.py` additionally lets
each leaf's *quantization step* scale with its complexity score (finer for complex
leaves, coarser for flat ones) — so the novelty acts at both the *tokenization* level
and the *rate allocation* level.

This was also literally the team's own starting idea ("build efficient patches before
compression for a more efficient compression process") — CAQT is a concrete, measurable,
implementable realization of that idea, with an honest accounting of its own overhead
(the quadtree structure bitstream, `src/coding/quadtree_codec.py`) so the reported gains
can't be an artifact of hiding costs.

## 2. Pipeline overview

```
Image (H x W x C)
   │
   ▼
[Adaptive Quadtree Patcher]  <- NOVELTY: recursive split based on local complexity
   │  variable-size leaves (32..256 px), each tagged with (x, y, size, complexity)
   ▼
[Resample every leaf to min_patch_size]   (so one encoder handles all leaf sizes)
   │
   ▼
[Transformer Encoder]
   patch embed (conv) + scale embedding (leaf size) + positional embedding (grid xy)
   -> N transformer blocks (multi-head self-attention + MLP) -> latent tokens
   │
   ▼
[Entropy Bottleneck]  <- NOVELTY (rate half): per-leaf quantization step scaled by
   factorized probability model (Balle et al. 2018)     that leaf's complexity score
   │
   ▼
[rANS Entropy Coder]  ->  binary bitstream  (+ quadtree structure bitstream, ~1 bit/node)
   │                                              (real, lossless, measured — not just estimated)
   ▼  (decoder side, mirror image)
[Transformer Decoder] -> per-leaf reconstructed patches (min_patch_size²)
   │
   ▼
[Adaptive Patcher: reassemble()]  -> upsample each leaf to native size, feathered blend
   │
   ▼
Reconstructed image (H x W x C)
```

See `src/models/autoencoder.py::AdaptiveSatelliteCodec.forward` for the code that wires
this together for a single image.

## 3. Component-to-report mapping

| Report artifact | Implementation |
|---|---|
| Eq. 2 (Z = E(I)) | `models/encoder.py::TransformerEncoder` |
| Eq. 3 (quantization) | `models/entropy_bottleneck.py::EntropyBottleneck.forward` |
| Eq. 4 (reconstruction) | `models/decoder.py::TransformerDecoder` + `patching/adaptive_quadtree.py::reassemble` |
| Eq. 9 (scaled dot-product attention) | `models/transformer_blocks.py::MultiHeadSelfAttention` |
| Eq. 11/12 (rANS encode/decode) | `coding/rans_coder.py::RansCoder` |
| Eq. 16–19 (rate-distortion loss) | `losses/rate_distortion.py::rate_distortion_loss` |
| Fig. 2 training methodology | `train.py` |
| Table I (ablation M1–M4) | `evaluate.py --fixed_grid_ablation` is the tessellation half; the AE/transformer/entropy/rANS ablation ladder (M1–M4) is a config sweep over `AdaptiveSatelliteCodec` args + whether `RansCoder` is invoked — documented in `docs/TEST_PLAN.md` §3. |
| Table II (compression comparison) | `baselines.py::run_all_baselines` + `evaluate.py` |

## 4. Design decisions worth defending in a viva

### 4.1 Why quadtree (not superpixels / saliency maps / learned segmentation)?
Quadtrees are: (a) trivial to encode losslessly and cheaply (§4.3 below), (b) trivial to
invert deterministically (no learned/ambiguous boundary), (c) hardware-friendly — a
recursive Laplacian-variance check is orders of magnitude cheaper than a learned
segmentation network, which matters given the eventual onboard-deployment goal
(Phase I report §2.2.1, goal 4). Superpixels/learned segmentation could plausibly do
better but would need arbitrary-shape patch handling, which the transformer/rANS stages
aren't built for, and would themselves need to be transmitted or re-derived
deterministically at the decoder — a much bigger scope increase for a project timeline.

### 4.2 Why also vary the quantization step, not just patch size?
Patch-size adaptivity alone (bigger patches carry proportionally more pixels but were
*originally* going to be embedded as a single token each) already saves tokens on flat
regions. But two leaves of the same size can still differ in complexity (e.g. a 32×32
leaf that only exists because its 64×64 parent narrowly failed the split test, vs. one
that's maximally complex). The `complexity_weight` argument to `EntropyBottleneck`
gives a second, finer-grained lever: same token budget, different bit budget.

### 4.3 Overhead accounting
For a 256×256 image tessellated down to a 32×32 minimum, the tree has at most
1 + 4 + 16 + 64 = 85 nodes → ≤ 85 bits (≈11 bytes) of structural overhead, encoded via
`coding/quadtree_codec.py`'s pre-order bit scheme (verified round-trip-exact in
`tests/test_quadtree.py::test_quadtree_structure_roundtrip`). This is why the
tessellation's savings (tokens/bits reallocated away from flat regions) can plausibly
outweigh its own bookkeeping cost — this is exactly the number to report in the final
paper's rate-distortion discussion, not hand-waved.

### 4.4 Why factorized entropy model instead of `compressai` directly
`FactorizedEntropyModel` reimplements the Balle et al. (2018) factorized prior
(simplified) so the codebase has no hard dependency on `compressai` (which pulls in a
specific pinned torch version that can conflict with Colab's preinstalled torch). Once
training is running, swapping in `compressai.entropy_models.EntropyBottleneck` is a
one-line change if a more battle-tested rate estimator is wanted — the interface
(`forward(z) -> z_hat, bits, likelihoods`) is compatible.

## 5. Onboard-deployment posture (why this design, even though FYP scope stops before flight code)

- The complexity scorer is O(N), parameter-free, and runs before any neural network —
  cheap enough to be a plausible first stage on constrained hardware (echoing Guerrisi
  2022/2023 and Giuffrida 2023's onboard CAE precedent from the lit review).
- All neural components are configurable in width/depth (`embed_dim`, `depth`,
  `num_heads`) specifically so the same codebase can be scaled down for a hypothetical
  onboard variant without an architecture rewrite.
- This is explicitly **not** claimed as flight-qualified; docs/SRS.md §2 lists it as out
  of scope, matching the Phase I report's own scope boundary (§2.3.2).
