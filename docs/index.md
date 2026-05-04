# uplift-bench

Reproducible benchmark of seven uplift-modeling approaches on four public
datasets, with bootstrap confidence intervals, robustness analysis, and a
full MLflow pipeline.

## What's inside

* **7 meta-learners** — S, T, X, R, DR (doubly robust), class
  transformation, causal forest.
* **3 base learners** — CatBoost (default), LightGBM, LogisticRegression.
* **4 datasets** — Hillstrom (MineThatData), Criteo Uplift v2, X5
  RetailHero, MegaFon.
* **Metrics** — Qini, AUUC, uplift@k, per-decile uplift, with BCa
  bootstrap CIs and a paired-bootstrap significance test.
* **Robustness** — permutation importance for *uplift* (not outcome),
  drop-feature stability, learning curves, propensity overlap diagnostics.
* **Tracking** — every run logged to MLflow with parameters, metrics
  (with CIs), and artifacts (Qini curves, configs, dataset hashes).
* **Reproducibility** — Hydra structured configs + seeded RNG everywhere.

## Quickstart

```bash
uv sync --extra bench --extra dev
uv run uplift-bench download all
uv run uplift-bench benchmark +experiment=quick_smoke
```

See [Reproducing](reproducing.md) for the full benchmark recipe.

## Status

This is a research / engineering benchmark — not a production library.
The focus is on producing comparable, reproducible numbers across
methods, not on the absolute fastest implementation of any one method.
