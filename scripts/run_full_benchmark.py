"""Run the full benchmark matrix and write the results table.

Programmatic equivalent of `uv run uplift-bench benchmark -m ...`. We use
the Python API rather than shelling out to Hydra multirun because:
  * Tighter control over which (model, dataset, seed) combos to attempt
    when only some datasets are available locally.
  * Single aggregator producing the final CSV / Markdown without scraping
    Hydra output dirs.

Usage:
    uv run python scripts/run_full_benchmark.py
    uv run python scripts/run_full_benchmark.py --datasets hillstrom criteo \\
        --models s_learner dr_learner --seeds 42 43 \\
        --base-learner catboost --criteo-subsample 1000000

The benchmark CSV is written to results/benchmark_results.csv and the
Markdown summary to results/benchmark_results.md.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

from uplift_bench.pipelines.report import aggregate_results, write_markdown_table
from uplift_bench.pipelines.train import TrainResult, run_one
from uplift_bench.utils.logging import configure, get_logger

log = get_logger(__name__)

DEFAULT_MODELS = (
    "s_learner",
    "t_learner",
    "x_learner",
    "r_learner",
    "dr_learner",
    "class_transformation",
    "causal_forest",
)
DEFAULT_DATASETS = ("hillstrom",)  # extend on the CLI: --datasets hillstrom criteo
DEFAULT_SEEDS = (42,)


def _build_dataset_cfg(name: str, data_dir: str, criteo_subsample: int | None) -> dict:
    base = {"name": name, "data_dir": data_dir, "loader_params": {}}
    if name == "hillstrom":
        base["loader_params"] = {"treatment_arm": "Womens E-Mail", "outcome": "visit"}
    elif name == "criteo":
        base["loader_params"] = {
            "outcome": "visit",
            "subsample": criteo_subsample,
            "subsample_seed": 42,
        }
    return base


def _build_model_cfg(name: str) -> dict:
    if name in {"r_learner", "dr_learner"}:
        return {"name": name, "extra_params": {"n_splits": 5}}
    if name == "causal_forest":
        return {
            "name": name,
            "extra_params": {"n_estimators": 200, "min_samples_leaf": 30},
        }
    return {"name": name, "extra_params": {}}


def _build_base_learner_cfg(base_learner: str, fast: bool) -> dict:
    if base_learner == "catboost":
        params = {"iterations": 200 if fast else 500, "depth": 6, "learning_rate": 0.05}
    elif base_learner == "lightgbm":
        params = {"n_estimators": 200 if fast else 500, "num_leaves": 63}
    elif base_learner == "logreg":
        params = {"max_iter": 1000}
    else:
        raise SystemExit(f"unknown base_learner {base_learner!r}")
    return {"name": base_learner, "params": params}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--base-learner", default="catboost", choices=["catboost", "lightgbm", "logreg"]
    )
    parser.add_argument(
        "--data-dir",
        default="data/raw",
        help="passed through to each loader; use data/sample for smoke runs",
    )
    parser.add_argument(
        "--criteo-subsample",
        type=int,
        default=None,
        help="subsample Criteo to N rows (omit for full ~13.9M rows)",
    )
    parser.add_argument("--output-dir", default="outputs/benchmark")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--bootstrap-method", default="bca", choices=["bca", "percentile"])
    parser.add_argument("--enable-permutation", action="store_true")
    parser.add_argument("--enable-overlap", action="store_true")
    parser.add_argument(
        "--no-tracking", action="store_true", help="disable MLflow logging (CI / smoke runs)"
    )
    parser.add_argument(
        "--fast", action="store_true", help="reduce base learner iterations for quick iteration"
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="log and skip rather than fail the whole matrix",
    )
    args = parser.parse_args()

    configure(level="INFO")
    output_root = Path(args.output_dir).resolve()
    results_dir = Path(args.results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    matrix: list[tuple[str, str, int]] = [
        (ds, m, s) for ds in args.datasets for m in args.models for s in args.seeds
    ]
    log.info(
        "benchmark_matrix",
        n_combos=len(matrix),
        datasets=args.datasets,
        models=args.models,
        seeds=args.seeds,
    )

    results: list[TrainResult] = []
    failures: list[dict] = []

    for ds, model, seed in matrix:
        run_dir = output_root / f"{model}_{ds}_seed{seed}"
        log.info("matrix_step_start", dataset=ds, model=model, seed=seed)
        try:
            result = run_one(
                dataset_cfg=_build_dataset_cfg(ds, args.data_dir, args.criteo_subsample),
                model_cfg=_build_model_cfg(model),
                base_learner_cfg=_build_base_learner_cfg(args.base_learner, args.fast),
                split_cfg={"train_frac": 0.7, "val_frac": 0.15, "seed": seed},
                bootstrap_cfg={
                    "n_boot": args.n_boot,
                    "method": args.bootstrap_method,
                    "alpha": 0.05,
                    "n_jobs": 1,
                    "seed": seed,
                },
                robustness_cfg={
                    "enable_permutation": args.enable_permutation,
                    "permutation_n_repeats": 5,
                    "enable_overlap": args.enable_overlap,
                },
                tracking_cfg={
                    "enabled": not args.no_tracking,
                    "experiment_name": f"benchmark.{ds}",
                    "tracking_uri": "file://./mlruns",
                },
                seed=seed,
                output_dir=run_dir,
            )
            results.append(result)
            log.info("matrix_step_done", dataset=ds, model=model, seed=seed, qini=result.qini)
        except Exception as exc:
            tb = traceback.format_exc()
            failures.append(
                {"dataset": ds, "model": model, "seed": seed, "error": str(exc), "traceback": tb}
            )
            log.error("matrix_step_failed", dataset=ds, model=model, seed=seed, error=str(exc))
            if not args.continue_on_error:
                raise

    if not results:
        log.error("no_results_to_aggregate")
        return 1

    df = aggregate_results(results)
    csv_path = results_dir / "benchmark_results.csv"
    md_path = results_dir / "benchmark_results.md"
    df.to_csv(csv_path, index=False)
    write_markdown_table(df, md_path)
    log.info(
        "benchmark_done",
        results_csv=str(csv_path),
        md=str(md_path),
        n_runs=len(results),
        n_failures=len(failures),
    )

    if failures:
        # Failures captured to disk so we can inspect them even when CI
        # later truncates stdout. The matrix continues, the report shows
        # what made it through.
        from uplift_bench.utils.io import dump_json  # noqa: PLC0415

        dump_json(failures, results_dir / "benchmark_failures.json")

    return 0


if __name__ == "__main__":
    sys.exit(main())
