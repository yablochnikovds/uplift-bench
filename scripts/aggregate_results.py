"""Aggregate per-seed runs into a single per-(dataset, model) summary.

Reads `results/<dataset>_results.csv` files (one per dataset run) and
writes:
    results/benchmark_summary.csv     — per (dataset, model) means / stds
    results/benchmark_summary.md      — markdown for the README
    results/figures/qini_<dataset>.png — bar chart with CIs
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from uplift_bench.utils.io import ensure_dir
from uplift_bench.viz.comparison_plots import plot_qini_comparison
from uplift_bench.viz.diagnostic_plots import plot_qini_heatmap


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument(
        "--inputs", nargs="+", help="explicit list of CSVs; default = results/*_results.csv"
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if args.inputs:
        csvs = [Path(p) for p in args.inputs]
    else:
        csvs = sorted(results_dir.glob("*_results.csv"))
    if not csvs:
        raise SystemExit(f"no per-dataset CSVs found under {results_dir}/")

    frames = [pd.read_csv(p) for p in csvs]
    df = pd.concat(frames, ignore_index=True)

    summary = (
        df.groupby(["dataset", "model"])
        .agg(
            qini_mean=("qini", "mean"),
            qini_std=("qini", "std"),
            qini_ci_lower=("qini_ci_lower", "mean"),
            qini_ci_upper=("qini_ci_upper", "mean"),
            auuc=("auuc_normalized", "mean"),
            n_seeds=("seed", "nunique"),
        )
        .reset_index()
        .sort_values(["dataset", "qini_mean"], ascending=[True, False])
    )
    summary.to_csv(results_dir / "benchmark_summary.csv", index=False)

    md_path = results_dir / "benchmark_summary.md"
    md_path.write_text(summary.round(4).to_markdown(index=False))

    figures_dir = ensure_dir(results_dir / "figures")
    for ds, sub in summary.groupby("dataset"):
        per_ds = sub.rename(
            columns={
                "qini_mean": "qini",
            }
        )[["model", "qini", "qini_ci_lower", "qini_ci_upper"]]
        plot_qini_comparison(
            per_ds,
            title=f"Qini by model — {ds}",
            save_path=figures_dir / f"qini_{ds}.png",
        )

    # Cross-dataset heatmap (only meaningful with ≥ 2 datasets).
    if summary["dataset"].nunique() >= 2:
        plot_qini_heatmap(
            summary,
            value_col="qini_mean",
            title="Qini by model x dataset",
            save_path=figures_dir / "heatmap_qini.png",
        )
    print(f"wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
