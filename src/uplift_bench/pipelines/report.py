"""Aggregate per-run TrainResult records into the bench-wide CSV/MD report.

Used by `scripts/run_full_benchmark.sh` after a full multirun completes.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from uplift_bench.pipelines.train import TrainResult


def aggregate_results(results: list[TrainResult]) -> pd.DataFrame:
    """Flatten a list of TrainResults into a tidy DataFrame.

    Each row is one (model, dataset, seed). Columns: model, dataset, seed,
    qini, qini_ci_lower, qini_ci_upper, auuc, then any extra metrics.
    """
    rows: list[dict[str, float | str | int]] = []
    for r in results:
        base: dict[str, float | str | int] = {
            "model": r.model_name,
            "dataset": r.dataset_name,
            "seed": r.seed,
            "qini": r.qini,
            "qini_ci_lower": r.qini_ci_lower,
            "qini_ci_upper": r.qini_ci_upper,
            "auuc": r.auuc,
        }
        base.update(r.metrics)
        rows.append(base)
    return pd.DataFrame(rows)


def write_markdown_table(
    df: pd.DataFrame,
    path: Path,
    *,
    sort_by: str = "qini",
) -> None:
    """Pretty-print the results table as Markdown for the README."""
    cols = [
        c
        for c in [
            "dataset",
            "model",
            "qini",
            "qini_ci_lower",
            "qini_ci_upper",
            "auuc_normalized",
            "auuc",
            "uplift_at_10",
            "uplift_at_20",
            "uplift_at_30",
            "overlap_ess_ratio",
        ]
        if c in df.columns
    ]
    pretty = df[cols].copy()
    if sort_by in pretty.columns:
        pretty = pretty.sort_values(["dataset", sort_by], ascending=[True, False])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(pretty.to_markdown(index=False, floatfmt=".4f"))
