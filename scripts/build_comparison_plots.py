"""Build presentation-ready comparison plots: every model on one canvas.

For each dataset, produces:
    results/figures/comparison_qini_curves_<dataset>.png
    results/figures/comparison_deciles_<dataset>.png
    results/figures/comparison_calibration_<dataset>.png

Reads pre-computed per-run artefacts from `outputs/v0.2.0/` (deciles.csv +
the trained model is not re-loaded; we re-run predict_uplift on the
test fold to rebuild Qini and calibration arrays). Uses the first seed
of each (dataset, model) pair so the comparison is apples-to-apples.

Run after `scripts/run_full_benchmark.py` and
`scripts/aggregate_results.py`.
"""

from __future__ import annotations

import argparse
import functools
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from uplift_bench._presets import fast_model_kwargs, loader_params
from uplift_bench.data.factory import make_loader
from uplift_bench.data.splits import make_splits
from uplift_bench.metrics._common import make_bucket_indices
from uplift_bench.metrics.cumulative_gain import cumulative_gain_curve
from uplift_bench.metrics.policy_value import policy_value_curve
from uplift_bench.metrics.qini import qini_curve
from uplift_bench.models.factory import make_model
from uplift_bench.utils.logging import configure, get_logger
from uplift_bench.utils.reproducibility import seed_everything
from uplift_bench.viz.qini_curve import plot_qini_curves as plot_qini_curves_canonical

log = get_logger(__name__)

# Consistent colour per model so the same colour means the same model
# across all comparison plots.
MODEL_COLORS: dict[str, str] = {
    "s_learner": "#1f77b4",
    "t_learner": "#ff7f0e",
    "x_learner": "#2ca02c",
    "r_learner": "#d62728",
    "dr_learner": "#9467bd",
    "class_transformation": "#8c564b",
    "causal_forest": "#e377c2",
}


@functools.cache
def _refit_one(
    dataset: str,
    model_name: str,
    seed: int,
    data_dir: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Refit one (dataset, model, seed) and return (preds, t_test, y_test, true_tau).

    Cached so the 4 comparison plot types (qini / calibration / cumulative
    gain / policy value) reuse the same fit instead of training the model
    from scratch four times. Saves ~28-70s on a Hillstrom + synthetic run.
    """
    seed_everything(seed)
    loader = make_loader(
        dataset,
        data_dir=data_dir,
        **loader_params(dataset, seed=seed, criteo_subsample=1_000_000),
    )
    ds = loader.load()
    splits = make_splits(ds, train_frac=0.7, val_frac=0.15, seed=seed)
    X, t, y = ds.X, ds.t, ds.y
    X_train, t_train, y_train = X.iloc[splits.train], t[splits.train], y[splits.train]
    X_test, t_test, y_test = X.iloc[splits.test], t[splits.test], y[splits.test]

    model = make_model(model_name, **fast_model_kwargs(model_name, seed=seed))
    model.fit(X_train, t_train, y_train)
    preds = model.predict_uplift(X_test)
    # true_tau placeholder kept for signature stability with synthetic-only callers.
    return preds, t_test, y_test, np.zeros_like(preds)


def plot_qini_curves_overlay(
    dataset: str,
    models: list[str],
    seed: int,
    data_dir: str,
    save_path: Path,
) -> None:
    """Refit every model on `dataset/seed` and overlay their Qini curves.

    Thin wrapper: delegates plotting to `viz.qini_curve.plot_qini_curves`
    (with the per-model colour map) so we don't carry a second copy of
    matplotlib boilerplate.
    """
    curves = {m: qini_curve(*_refit_one(dataset, m, seed, data_dir)[:3]) for m in models}
    plot_qini_curves_canonical(
        curves,
        title=f"Qini curves — all models on {dataset} (seed {seed})",
        colors=MODEL_COLORS,
        save_path=save_path,
    )
    log.info("wrote_overlay", path=str(save_path))


def plot_decile_overlay(
    dataset: str,
    outputs_root: Path,
    seed: int,
    save_path: Path,
) -> None:
    """Grouped bar chart: per-decile uplift by model."""
    rows: list[pd.DataFrame] = []
    for run_dir in sorted(outputs_root.glob(f"*_{dataset}_seed{seed}")):
        model = run_dir.name.replace(f"_{dataset}_seed{seed}", "")
        decile_csv = run_dir / "deciles.csv"
        if not decile_csv.exists():
            continue
        df = pd.read_csv(decile_csv)
        df["model"] = model
        rows.append(df)
    if not rows:
        log.warning("no_decile_csvs_found", dataset=dataset)
        return
    full = pd.concat(rows, ignore_index=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    pivot = full.pivot(index="bucket", columns="model", values="uplift")
    width = 0.11
    x = np.arange(len(pivot.index))
    for i, m in enumerate(pivot.columns):
        ax.bar(
            x + (i - len(pivot.columns) / 2) * width,
            pivot[m],
            width,
            label=m,
            color=MODEL_COLORS.get(m, "grey"),
            edgecolor="black",
            alpha=0.85,
        )
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in pivot.index])
    ax.set_xlabel("decile (1 = highest predicted uplift)")
    ax.set_ylabel("realised uplift in bucket")
    ax.set_title(f"Per-decile uplift by model — {dataset} (seed {seed})")
    ax.legend(loc="upper right", fontsize=8, ncol=2, frameon=True)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=140)
    plt.close(fig)
    log.info("wrote_overlay", path=str(save_path))


def plot_calibration_overlay(
    dataset: str,
    models: list[str],
    seed: int,
    data_dir: str,
    save_path: Path,
    n_buckets: int = 10,
) -> None:
    """Overlay calibration scatter for every model — points + identity line."""
    fig, ax = plt.subplots(figsize=(7, 6.5))
    all_x: list[float] = []
    all_y: list[float] = []
    for m in models:
        preds, t_test, y_test, _ = _refit_one(dataset, m, seed, data_dir)
        order = np.argsort(-preds, kind="stable")
        bucket_idx = make_bucket_indices(len(preds), n_buckets)
        s_ord = preds[order]
        t_ord = t_test[order].astype(bool)
        y_ord = y_test[order].astype(np.float64)
        xs: list[float] = []
        ys: list[float] = []
        for b in range(1, n_buckets + 1):
            mask = bucket_idx == b
            tb = t_ord[mask]
            yb = y_ord[mask]
            if tb.any() and (~tb).any():
                xs.append(float(s_ord[mask].mean()))
                ys.append(float(yb[tb].mean() - yb[~tb].mean()))
        ax.plot(
            xs, ys, marker="o", lw=1.5, ms=7, color=MODEL_COLORS.get(m, "grey"), label=m, alpha=0.9
        )
        all_x.extend(xs)
        all_y.extend(ys)
    if all_x and all_y:
        lim = max(abs(min(all_x + all_y)), abs(max(all_x + all_y))) * 1.1
        ax.plot([-lim, lim], [-lim, lim], "--", color="grey", lw=1, label="perfect calibration")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
    ax.axhline(0, color="black", lw=0.4)
    ax.axvline(0, color="black", lw=0.4)
    ax.set_xlabel("predicted uplift (bucket mean)")
    ax.set_ylabel("realised uplift (bucket diff-of-means)")
    ax.set_title(f"Calibration: predicted vs realised uplift — {dataset} (seed {seed})")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=9, frameon=True)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=140)
    plt.close(fig)
    log.info("wrote_overlay", path=str(save_path))


def plot_cumulative_gain_overlay(
    dataset: str,
    models: list[str],
    seed: int,
    data_dir: str,
    save_path: Path,
) -> None:
    """One cumulative-gain curve per model (Radcliffe 2007)."""
    fig, ax = plt.subplots(figsize=(8, 5.5))
    endpoints: list[float] = []
    for m in models:
        preds, t_test, y_test, _ = _refit_one(dataset, m, seed, data_dir)
        cg = cumulative_gain_curve(preds, t_test, y_test)
        ax.plot(
            cg.population_share,
            cg.cumulative_responders_per_capita,
            lw=2.2,
            color=MODEL_COLORS.get(m, "grey"),
            label=f"{m}  AUC={cg.auc:.4f}",
        )
        endpoints.append(float(cg.cumulative_responders_per_capita[-1]))
    if endpoints:
        ax.plot([0, 1], [0, max(endpoints)], "--", color="grey", lw=1, label="random")
    ax.set_xlabel("targeted fraction of population")
    ax.set_ylabel("cumulative treated responders / N")
    ax.set_title(f"Cumulative gain — all models on {dataset} (seed {seed})")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9, frameon=True)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=140)
    plt.close(fig)
    log.info("wrote_overlay", path=str(save_path))


def plot_policy_value_overlay(
    dataset: str,
    models: list[str],
    seed: int,
    data_dir: str,
    save_path: Path,
) -> None:
    """One policy-value curve per model (Manski 2004 / Athey & Wager 2021)."""
    fig, ax = plt.subplots(figsize=(8, 5.5))
    budgets = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0]
    for m in models:
        preds, t_test, y_test, _ = _refit_one(dataset, m, seed, data_dir)
        pv = policy_value_curve(preds, t_test, y_test, budgets=budgets)
        ax.plot(
            pv.budgets,
            pv.policy_values,
            marker="o",
            lw=2,
            ms=5,
            color=MODEL_COLORS.get(m, "grey"),
            label=m,
        )
    ax.set_xlabel("budget (fraction of population treated)")
    ax.set_ylabel("expected outcome E[Y(pi(X))]")
    ax.set_title(f"Policy value vs budget — all models on {dataset} (seed {seed})")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=9, frameon=True)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=140)
    plt.close(fig)
    log.info("wrote_overlay", path=str(save_path))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["hillstrom", "synthetic"])
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "s_learner",
            "t_learner",
            "x_learner",
            "r_learner",
            "dr_learner",
            "class_transformation",
            "causal_forest",
        ],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--outputs-dir", default="outputs/v0.2.0")
    parser.add_argument("--figures-dir", default="results/figures")
    args = parser.parse_args()

    configure(level="INFO")
    figures_dir = Path(args.figures_dir)
    outputs_root = Path(args.outputs_dir)

    for ds in args.datasets:
        plot_qini_curves_overlay(
            ds,
            args.models,
            args.seed,
            args.data_dir,
            figures_dir / f"comparison_qini_curves_{ds}.png",
        )
        plot_decile_overlay(
            ds,
            outputs_root,
            args.seed,
            figures_dir / f"comparison_deciles_{ds}.png",
        )
        plot_calibration_overlay(
            ds,
            args.models,
            args.seed,
            args.data_dir,
            figures_dir / f"comparison_calibration_{ds}.png",
        )
        plot_cumulative_gain_overlay(
            ds,
            args.models,
            args.seed,
            args.data_dir,
            figures_dir / f"comparison_cumulative_gain_{ds}.png",
        )
        plot_policy_value_overlay(
            ds,
            args.models,
            args.seed,
            args.data_dir,
            figures_dir / f"comparison_policy_value_{ds}.png",
        )

    # JSON manifest of what was produced — handy for the README to
    # discover the exact set of comparison figures committed.
    manifest = {
        "datasets": args.datasets,
        "models": args.models,
        "seed": args.seed,
        "figures": sorted(p.name for p in figures_dir.glob("comparison_*.png")),
    }
    (figures_dir / "comparison_manifest.json").write_text(
        json.dumps(manifest, indent=2),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
