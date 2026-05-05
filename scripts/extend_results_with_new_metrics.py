"""Extend existing per-dataset CSVs with cumulative_gain_auc + policy_value_b*.

Avoids re-running the whole benchmark — just refits each (dataset, model,
seed) once and computes the new metrics on the same test split. Existing
columns (qini, auuc, uplift_at_*) are preserved.

Use this *only* when adding new metrics to an already-committed result
set. Default flow for fresh runs is `scripts/run_full_benchmark.py`,
which writes the new columns natively.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from uplift_bench.data.factory import make_loader
from uplift_bench.data.splits import make_splits
from uplift_bench.metrics.cumulative_gain import cumulative_gain_curve
from uplift_bench.metrics.policy_value import policy_value_curve
from uplift_bench.models.factory import make_model
from uplift_bench.utils.logging import configure, get_logger
from uplift_bench.utils.reproducibility import seed_everything

# Sibling helper module — shared with `build_comparison_plots.py`.
sys.path.insert(0, str(Path(__file__).parent))
from _bench_helpers import fast_model_kwargs, loader_params

log = get_logger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-dataset-dir", default="results/per_dataset")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--datasets", nargs="+", default=["hillstrom", "synthetic"])
    parser.add_argument("--criteo-subsample", type=int, default=1_000_000)
    args = parser.parse_args()

    configure(level="INFO")
    per_dir = Path(args.per_dataset_dir)

    for ds in args.datasets:
        csv = per_dir / f"{ds}.csv"
        if not csv.exists():
            log.warning("missing_csv_skip", path=str(csv))
            continue
        df = pd.read_csv(csv)

        new_cols: dict[tuple[str, int], dict[str, float]] = {}
        for (model, seed_raw), _ in df.groupby(["model", "seed"]):
            seed = int(seed_raw)
            seed_everything(seed)
            loader = make_loader(
                ds,
                data_dir=args.data_dir,
                **loader_params(ds, seed=seed, criteo_subsample=args.criteo_subsample),
            )
            dataset = loader.load()
            splits = make_splits(dataset, train_frac=0.7, val_frac=0.15, seed=seed)
            X = dataset.X
            t = dataset.t
            y = dataset.y
            X_train = X.iloc[splits.train]
            t_train = t[splits.train]
            y_train = y[splits.train]
            X_test = X.iloc[splits.test]
            t_test = t[splits.test]
            y_test = y[splits.test]

            model_obj = make_model(model, **fast_model_kwargs(model, seed=seed))
            log.info("refitting", dataset=ds, model=model, seed=seed)
            model_obj.fit(X_train, t_train, y_train)
            preds = model_obj.predict_uplift(X_test)

            cg = cumulative_gain_curve(preds, t_test, y_test)
            pv = policy_value_curve(
                preds,
                t_test,
                y_test,
                budgets=[0.0, 0.1, 0.2, 0.3, 0.5, 1.0],
            )
            extras = {"cumulative_gain_auc": float(cg.auc)}
            for b, v in zip(pv.budgets, pv.policy_values, strict=True):
                extras[f"policy_value_b{int(b * 100):02d}"] = float(v)
            new_cols[(model, seed)] = extras

        # Stitch back: add new columns onto df aligned on (model, seed).
        # Build a wide DataFrame once and join, instead of df.apply per column.
        new_df = pd.DataFrame.from_dict(new_cols, orient="index")
        new_df.index = pd.MultiIndex.from_tuples(new_df.index, names=["model", "seed"])
        df = df.merge(
            new_df.reset_index(),
            on=["model", "seed"],
            how="left",
            validate="m:1",
        )
        df.to_csv(csv, index=False)
        df.round(4).to_markdown(per_dir / f"{ds}.md", index=False)
        log.info("extended_csv", path=str(csv), n_new_cols=len(extras))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
