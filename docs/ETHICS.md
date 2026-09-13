# Ethics & Responsible Use

This maps to Program Outcome PO8 (Ethics) and PO6/PO7 (society/environment) from the
Phase I report's PO/PSO mapping, made concrete for the working implementation.

## 1. Data provenance and licensing

- **UC Merced Land Use Dataset** and **EuroSAT** are both public, open, research-use
  datasets (see `src/data/download.py` for sources). No proprietary, classified, or
  export-controlled imagery is used.
- The codebase does not scrape or process live/real-time satellite feeds, personal
  property imagery at identifying resolution, or any data requiring privacy review.
- Anyone extending this project to a different (e.g. commercial or higher-resolution)
  dataset is responsible for checking that dataset's license and any government
  export-control restrictions on high-resolution overhead imagery before redistributing
  results or trained weights.

## 2. Dual-use and misuse considerations

Satellite image compression is a general-purpose data-engineering technique; it doesn't
itself create surveillance capability, but it does make imagery cheaper to store,
transmit, and archive at scale — which is also true of every classical codec (JPEG2000,
CCSDS) already in wide civil use. This project:

- Targets civilian applications named in the Phase I report — GIS mapping, agricultural
  monitoring, environmental/disaster tracking (Report §1.1) — and the evaluation plan
  is scoped to those tasks.
- Does not target, and this repository will not be extended by the team to target,
  identification of individuals, vehicles, or license plates in imagery, or any
  military targeting use case.
- Compression is lossy; a documented limitation (docs/ARCHITECTURE.md, and the report's
  own §XI "Limitations") is that reconstruction error must be characterized *before*
  compressed imagery is used for any safety-relevant downstream decision (e.g.
  disaster-response resource allocation) — PSNR/SSIM alone do not guarantee a specific
  downstream task stays accurate, which is exactly why the eval plan includes
  downstream-task fidelity checks, not just pixel metrics.

## 3. Environmental footprint of the research itself

- Model sizes and dataset choice (EuroSAT primary, small resolution) were deliberately
  picked to fit free-tier Colab/Kaggle GPU budgets (see docs/SRS.md §5), keeping the
  project's own training compute/energy footprint modest — worth stating explicitly
  given the project's stated goal of *supporting* environmental monitoring.

## 4. Academic integrity

- All twelve+ related works from the Phase I literature survey are cited in code
  comments and docs wherever a specific design choice is drawn from them
  (`docs/ARCHITECTURE.md` §4), rather than presenting standard techniques (transformer
  attention, factorized entropy models, rANS) as if newly invented here. Only the
  content-adaptive quadtree tessellation + variable bit allocation combination is
  claimed as this project's contribution.
- Any pretrained weights, if used in later iterations (e.g. for a learned complexity
  scorer, docs/ARCHITECTURE.md §4.1), must be from openly licensed sources, and that
  source must be recorded in this file and in the final report.

## 5. Honest reporting

- Ablation and comparison protocols (docs/TEST_PLAN.md) are written to make it possible
  to *disprove* the novelty's value on a given dataset (via `--fixed_grid_ablation`)
  rather than only show configurations where it wins. If the fixed-grid ablation ever
  outperforms the adaptive version on a metric, that result should be reported, not
  filtered out of the final paper.
