# Results

The latest benchmark numbers committed to the repo.

The full table lives at
[`results/benchmark_results.md`](https://github.com/yablochnikovds/uplift-bench/blob/main/results/benchmark_results.md)
and the underlying CSV at
[`results/benchmark_results.csv`](https://github.com/yablochnikovds/uplift-bench/blob/main/results/benchmark_results.csv).

## How to read the table

* `qini` — point estimate on the held-out test fold.
* `qini_ci_lower` / `qini_ci_upper` — 95% BCa bootstrap interval
  (1000 resamples by default).
* `auuc_normalized` — AUUC divided by perfect-ranking AUUC, in roughly
  $[-1, 1]$.
* `uplift_at_k` — realised uplift in the top-k targeted population.
* `overlap_ess_ratio` — effective sample size after IPW reweighting,
  divided by N. Closer to 1 = better treatment/control overlap.

Per-dataset breakdowns live in `results/<dataset>_results.md`.

## Reproducing

See [Reproducing](reproducing.md). Same `--seeds` give bit-identical
numbers; the dataset SHA is logged to MLflow so you can be sure you
trained on the same bytes.
