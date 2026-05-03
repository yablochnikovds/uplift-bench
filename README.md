# uplift-bench

[![CI](https://github.com/yablochnikovds/uplift-bench/actions/workflows/ci.yml/badge.svg)](https://github.com/yablochnikovds/uplift-bench/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/yablochnikovds/uplift-bench/branch/main/graph/badge.svg)](https://codecov.io/gh/yablochnikovds/uplift-bench)
[![python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](pyproject.toml)
[![ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![mypy: strict](https://img.shields.io/badge/mypy-strict-blue.svg)](https://mypy.readthedocs.io/en/stable/)
[![license: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Reproducible benchmark of seven uplift modeling approaches on four public
datasets, with bootstrap confidence intervals, robustness analysis and a
full MLflow pipeline.

> Status: under active construction. Stage tracker lives in [`results/`](results/).

## What's inside

- **7 meta-learners** — S, T, X, R, DR (doubly robust), class transformation, causal forest
- **3 base learners** — CatBoost (default), LightGBM, LogisticRegression
- **4 datasets** — Hillstrom (MineThatData), Criteo Uplift v2, X5 RetailHero, MegaFon
- **Metrics** — Qini, AUUC, uplift@k, per-decile uplift, with BCa bootstrap CIs and a paired bootstrap significance test
- **Robustness** — permutation feature importance for *uplift* (not outcome), drop-feature stability, learning curves, propensity overlap diagnostics
- **Tracking** — every run logged to MLflow with parameters, metrics (with CIs), artifacts (Qini curves, configs, dataset hashes)
- **Reproducibility** — Hydra structured configs + seeded RNG everywhere; same seed → identical metrics

## Quickstart

```bash
# 1. install (uv recommended, https://docs.astral.sh/uv/)
uv sync --extra bench --extra dev

# 2. download what we can grab automatically
uv run uplift-bench download all

# 3. smoke run on Hillstrom (~30 seconds)
uv run uplift-bench benchmark --config-name quick_smoke
```

Full benchmark and reproducibility instructions live in the
[docs](https://yablochnikovds.github.io/uplift-bench).

## Layout

```
src/uplift_bench/   library
configs/            Hydra configs (dataset / model / base_learner / experiment)
tests/              unit + integration tests + synthetic fixtures
results/            committed CSV/MD with the latest benchmark numbers
docs/               MkDocs site (methodology, datasets, results, API)
```

## License

MIT — see [LICENSE](LICENSE).
