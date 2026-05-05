# Architecture

A high-level map of the codebase, in the order you would naturally read
it. Each box is a folder under [`src/uplift_bench/`](https://github.com/yablochnikovds/uplift-bench/tree/main/src/uplift_bench).

```
                  ┌─────────────────────────┐
                  │  configs/  (Hydra YAML) │
                  └────────────┬────────────┘
                               │ composed at runtime
                               ▼
┌──────────┐     ┌──────────────────────────────┐     ┌────────────────┐
│  data/   │ ──▶ │       pipelines/train.py     │ ──▶ │   tracking/    │
│ loaders  │     │   (orchestrator: 6 stages)   │     │  MLflow logger │
└────┬─────┘     └─┬────────────┬────────────┬──┘     └────────────────┘
     │             │            │            │
     ▼             ▼            ▼            ▼
┌────────┐    ┌─────────┐  ┌─────────┐  ┌────────────┐
│ splits │    │ models/ │  │metrics/ │  │ robustness/│
└────────┘    └─────────┘  └─────────┘  └────────────┘
                  │             │             │
                  └─────────────┴─────────────┘
                                │
                                ▼
                           ┌────────┐
                           │  viz/  │
                           └────────┘
```

## Six-stage pipeline (one run)

`pipelines/train.run_one` is the heart of the project. It reads
top-to-bottom like a script:

| stage | function | what it does |
|---|---|---|
| 1 | `_prepare_data` | load loader → validate schema → stratified split |
| 2 | `_fit_uplift_model` | factory builds the meta-learner; `model.fit` |
| 3 | `_compute_metrics` | Qini / AUUC / uplift@k + bootstrap CI |
| 4 | `_compute_artifacts` | save 6 plots (qini curve, distribution, deciles, calibration, etc.) + CSVs + config dump |
| 5 | `_compute_robustness` | permutation importance + propensity overlap (when enabled in cfg) |
| 6 | `_log_run` | one MLflow run with params, metrics, artifacts, dataset SHA |

Each stage is a small dataclass-returning helper, so they're independently
testable from notebooks. `run_one` itself is ~30 lines.

## Module responsibilities

### `data/`

* `base.py` — abstract `DatasetLoader` (download → read → validate).
* `validation.py` — `DatasetSchema` (pydantic) + `UpliftDataset`
  (validated frozen container with `X`, `t`, `y` views).
* `splits.py` — stratified train/val/test on the joint (T, Y).
* `factory.py` — `make_loader(name, ...)`.
* `download.py` — HTTP downloader with sha-verified caching.
* `hillstrom.py`, `criteo.py`, `retailhero.py`, `megafon.py` — one loader
  per dataset.

### `models/`

* `base.py` — `UpliftModel` ABC. Two abstract methods: `fit`,
  `predict_uplift`. Stores hyper-parameters on `self.params` for
  uniform MLflow logging.
* `_base_learners.py` — factory turning `("catboost", "regression")` →
  fresh sklearn-compatible estimator. Handles cross-library kwarg
  aliasing (`iterations` ⇄ `n_estimators`).
* `factory.py` — short-name → class registry.
* One file per meta-learner: `s_learner.py`, `t_learner.py`,
  `x_learner.py`, `r_learner.py`, `dr_learner.py`,
  `class_transformation.py`, `causal_forest.py`. Each ~50–140 lines.

### `metrics/`

* `qini.py` — normalised Qini (raw / perfect-curve area).
* `auuc.py` — AUUC, perfect-ranking-normalised.
* `uplift_at_k.py` — top-k uplift.
* `decile.py` — per-decile uplift table.
* `bootstrap.py` — percentile + BCa CIs, paired-bootstrap test.
* `_common.py` — shared helpers (stable sort, shape coercion).

### `robustness/`

* `permutation.py` — feature shuffle vs Qini drop.
* `feature_drop.py` — refit-with-feature-removed.
* `learning_curve.py` — Qini vs train fraction.
* `overlap.py` — propensity ESS, fraction-in-clip-tail.

### `viz/`

* `qini_curve.py`, `uplift_distribution.py`, `comparison_plots.py` —
  basic per-run plots.
* `diagnostic_plots.py` — calibration, decile bar, propensity histogram,
  learning curve, permutation importance, model×dataset heatmap, Qini
  curves overlay, bootstrap distribution.

### `tracking/`

* `mlflow_logger.py` — thin wrapper around `mlflow.start_run` with a
  no-op variant when tracking is disabled.

### `pipelines/`

* `train.py` — `run_one(...)` (the 6-stage orchestrator).
* `benchmark.py` — Hydra `@main` entry point. Pure thin wrapper.
* `report.py` — aggregate `TrainResult` records into a tidy DataFrame +
  Markdown.

### `config/schemas.py`

Dataclasses that mirror the Hydra YAML hierarchy. Used both for
type-safe access in pipelines and for IDE autocomplete.

### `utils/`

* `logging.py` — structlog setup.
* `reproducibility.py` — `seed_everything()` + `SeedBundle`.
* `io.py` — atomic-write parquet, sha256, JSON dump.

## Where new things go

| You want to add… | Edit / create |
|---|---|
| a new meta-learner | new file in `models/`, register in `factory.MODEL_REGISTRY`, add YAML in `configs/model/`, add an entry to the integration smoke test |
| a new dataset | new file in `data/`, register in `factory.DATASET_REGISTRY`, YAML in `configs/dataset/`, sample fixture in `data/sample/<name>/` |
| a new metric | new file in `metrics/`, plumb into `pipelines/train._compute_metrics` |
| a new robustness check | new file in `robustness/`, gate in `pipelines/train._compute_robustness` behind a new `robustness_cfg` flag |
| a new diagnostic plot | function in `viz/diagnostic_plots.py`, call site in `pipelines/train._compute_artifacts` |
