# Reproducing the benchmark

## Local install

```bash
git clone https://github.com/yablochnikovds/uplift-bench
cd uplift-bench
uv sync --extra bench --extra dev
```

`uv` (https://docs.astral.sh/uv/) is required. CatBoost / LightGBM /
econml are heavy; the first sync takes a couple of minutes.

On macOS, install libomp for LightGBM:
```bash
brew install libomp
```

## Smoke run (~30 seconds)

```bash
uv run uplift-bench benchmark +experiment=quick_smoke dataset.data_dir=data/sample
```

Should produce a `qini` value in the structured log and a few PNG / CSV
artifacts under `outputs/`.

## Full benchmark on Hillstrom (~minutes)

```bash
uv run uplift-bench download hillstrom
uv run python scripts/run_full_benchmark.py \
  --datasets hillstrom \
  --models s_learner t_learner x_learner r_learner dr_learner class_transformation causal_forest \
  --seeds 42 43 44 \
  --base-learner catboost \
  --n-boot 500 \
  --bootstrap-method bca \
  --enable-overlap
```

Writes `results/benchmark_results.csv` and `.md`. Causal forest takes
the longest (15+ min per seed at `n_estimators=200` — drop to 100 if
you're impatient).

## Full benchmark on Criteo subsample (~30 min on M-chip)

```bash
uv run uplift-bench download criteo
uv run python scripts/run_full_benchmark.py \
  --datasets criteo \
  --models s_learner t_learner x_learner r_learner dr_learner class_transformation causal_forest \
  --seeds 42 \
  --criteo-subsample 1000000 \
  --n-boot 300 \
  --base-learner catboost
```

For the *full* 13.9M-row Criteo, drop `--criteo-subsample` and budget
several hours of CPU. Most of the time is spent in CatBoost; switch to
`--base-learner lightgbm` for ~2× speedup at small Qini cost.

## RetailHero / MegaFon

Both require manual data download — see [Datasets](datasets.md). Then:

```bash
uv run python scripts/run_full_benchmark.py \
  --datasets retailhero megafon \
  --models s_learner t_learner x_learner dr_learner causal_forest \
  --seeds 42
```

## MLflow UI

```bash
docker compose up -d mlflow
# open http://localhost:5000
```

Or natively:

```bash
uv run mlflow ui --backend-store-uri file://./mlruns
```

## Reproducibility guarantees

* Same `--seeds` produce identical metrics down to floating-point.
* Dataset SHA-256 is logged with every MLflow run, so you can confirm
  you used the same bytes.
* `uv.lock` pins every package to an exact version.

## Reproducing on limited hardware

Default settings target 16 GB RAM and an 8-core CPU. For tighter
constraints:

* Add `--criteo-subsample 100_000` for Criteo.
* Drop `--enable-permutation` (it refits feature-by-feature).
* Use `--base-learner logreg` for the lightest possible run.
