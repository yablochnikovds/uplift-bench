"""Diagnostic plots beyond the basic Qini curve.

Each function below renders a single, publication-quality figure that
reviewers familiar with uplift modeling expect to see in a benchmark
report. References:

* **Calibration plot** — compares predicted uplift bucket means to the
  realised uplift in the same bucket. Idea: Gutierrez & Gerardy 2017,
  "Causal Inference and Uplift Modeling: A Review of the Literature".
* **Per-decile uplift bar** — Athey & Imbens 2016, decile decomposition.
* **Propensity overlap histogram** — Imbens & Rubin 2015, "Causal
  Inference for Statistics, Social, and Biomedical Sciences", §12.
* **Learning curve** — Perlich et al. 2003, "Tree induction vs logistic
  regression: a learning-curve analysis".
* **Permutation importance** — Breiman 2001 (RF) adapted to uplift in
  this repo's robustness module.
* **Heatmap** — model x dataset Qini grid for at-a-glance comparison.

We use matplotlib's "Agg" backend so figures render without a display
(CI / Docker). Every function takes an optional `save_path` and returns
it for chaining.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from uplift_bench.metrics._common import NDArray1D, make_bucket_indices, sort_by_score_desc


def plot_calibration(
    score: NDArray1D,
    treatment: NDArray1D,
    outcome: NDArray1D,
    *,
    n_buckets: int = 10,
    title: str = "Uplift calibration",
    save_path: Path | None = None,
) -> Path | None:
    """Predicted-vs-realised uplift, bucketed.

    For each of `n_buckets` equal-size buckets (sorted by predicted
    uplift), we plot:
      x-axis: mean predicted uplift in the bucket
      y-axis: realised uplift = mean(y|T=1, bucket) - mean(y|T=0, bucket)

    A perfectly calibrated model lies on the diagonal y=x. Systematic
    over- or under-estimation shows up as a tilt or vertical shift.

    NaN buckets (no treated or no control rows) are skipped.
    """
    score = np.asarray(score)
    treatment = np.asarray(treatment).astype(bool)
    outcome = np.asarray(outcome).astype(np.float64)
    n = len(score)

    order = sort_by_score_desc(score)
    bucket_idx = make_bucket_indices(n, n_buckets)

    s_ord = score[order]
    t_ord = treatment[order]
    y_ord = outcome[order]

    pred_means: list[float] = []
    real_means: list[float] = []
    for b in range(1, n_buckets + 1):
        mask = bucket_idx == b
        t_b = t_ord[mask]
        y_b = y_ord[mask]
        if t_b.any() and (~t_b).any():
            pred_means.append(float(s_ord[mask].mean()))
            real_means.append(float(y_b[t_b].mean() - y_b[~t_b].mean()))

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(
        pred_means, real_means, s=80, alpha=0.85, color="#1f77b4", edgecolors="black", zorder=3
    )
    if pred_means and real_means:
        lim = (
            max(
                max(pred_means, default=0),
                max(real_means, default=0),
                -min(pred_means, default=0),
                -min(real_means, default=0),
            )
            * 1.1
        )
        ax.plot([-lim, lim], [-lim, lim], "--", color="grey", lw=1, label="perfect calibration")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
    ax.axhline(0, color="black", lw=0.5)
    ax.axvline(0, color="black", lw=0.5)
    ax.set_xlabel("predicted uplift (bucket mean)")
    ax.set_ylabel("realised uplift (bucket diff-of-means)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return _save(fig, save_path)


def plot_decile_uplift(
    decile_table: pd.DataFrame,
    *,
    title: str = "Per-decile uplift",
    save_path: Path | None = None,
) -> Path | None:
    """Bar chart of per-decile uplift from a decile_table DataFrame.

    Bucket 1 is the highest predicted uplift; a healthy model has
    a roughly monotone decreasing pattern.
    """
    df = decile_table.sort_values("bucket")
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#2ca02c" if u > 0 else "#d62728" for u in df["uplift"]]
    ax.bar(df["bucket"].astype(str), df["uplift"], color=colors, alpha=0.85, edgecolor="black")
    ax.axhline(0, color="black", lw=0.6)
    overall = float(df["uplift"].mean())
    ax.axhline(overall, color="grey", ls="--", lw=1, label=f"overall = {overall:+.4f}")
    ax.set_xlabel("decile (1 = highest predicted uplift)")
    ax.set_ylabel("realised uplift in bucket")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    return _save(fig, save_path)


def plot_propensity_histogram(
    treated: NDArray1D,
    control: NDArray1D,
    *,
    title: str = "Propensity score overlap",
    save_path: Path | None = None,
) -> Path | None:
    """Mirrored histogram of propensity scores by treatment arm.

    Heavy mass near 0 in the treated arm (or near 1 in the control arm)
    flags poor overlap — IPW-flavoured estimators (X / R / DR) become
    unreliable there.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = list(np.linspace(0, 1, 41))
    ax.hist(
        treated,
        bins=bins,
        alpha=0.6,
        label=f"treated (n={len(treated)})",
        color="#1f77b4",
        edgecolor="black",
    )
    ax.hist(
        control,
        bins=bins,
        alpha=0.6,
        label=f"control (n={len(control)})",
        color="#ff7f0e",
        edgecolor="black",
    )
    ax.axvspan(0.0, 0.05, color="grey", alpha=0.2, label="clipping tail")
    ax.axvspan(0.95, 1.0, color="grey", alpha=0.2)
    ax.set_xlabel("estimated propensity P(T=1|X)")
    ax.set_ylabel("count")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    return _save(fig, save_path)


def plot_learning_curve(
    df: pd.DataFrame,
    *,
    title: str = "Learning curve",
    save_path: Path | None = None,
) -> Path | None:
    """Plot Qini vs train size from a learning_curve DataFrame."""
    df = df.sort_values("n_train")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df["n_train"], df["qini"], marker="o", color="#1f77b4", lw=2)
    ax.set_xlabel("train size")
    ax.set_ylabel("Qini on held-out fold")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return _save(fig, save_path)


def plot_permutation_importance(
    df: pd.DataFrame,
    *,
    top_k: int = 20,
    title: str = "Permutation importance for uplift",
    save_path: Path | None = None,
) -> Path | None:
    """Horizontal bar of mean Qini drop per feature (top-k features)."""
    df = df.sort_values("mean_qini_drop", ascending=True).tail(top_k)
    fig, ax = plt.subplots(figsize=(6, max(2.5, 0.3 * len(df) + 1)))
    err = df["std_qini_drop"].to_numpy() if "std_qini_drop" in df else None
    colors = ["#2ca02c" if v > 0 else "#d62728" for v in df["mean_qini_drop"]]
    ax.barh(
        df["feature"],
        df["mean_qini_drop"],
        xerr=err,
        color=colors,
        alpha=0.85,
        edgecolor="black",
        capsize=3,
    )
    ax.axvline(0, color="black", lw=0.6)
    ax.set_xlabel("mean Qini drop when feature is shuffled")
    ax.set_title(title)
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    return _save(fig, save_path)


def plot_qini_heatmap(
    summary: pd.DataFrame,
    *,
    value_col: str = "qini_mean",
    title: str = "Qini by model x dataset",
    save_path: Path | None = None,
) -> Path | None:
    """Heatmap of mean Qini across (model, dataset) pairs.

    `summary` should have columns 'model', 'dataset', and `value_col`.
    """
    pivot = summary.pivot_table(
        index="model",
        columns="dataset",
        values=value_col,
        aggfunc="mean",
    ).sort_index()

    fig, ax = plt.subplots(figsize=(0.9 * len(pivot.columns) + 4, 0.4 * len(pivot.index) + 2))
    im = ax.imshow(
        pivot.values,
        aspect="auto",
        cmap="RdYlGn",
        vmin=-pivot.abs().to_numpy().max(),
        vmax=pivot.abs().to_numpy().max(),
    )
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=20, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            v = pivot.values[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:+.3f}", ha="center", va="center", color="black", fontsize=9)
    fig.colorbar(im, ax=ax, label=value_col)
    ax.set_title(title)
    fig.tight_layout()
    return _save(fig, save_path)


def plot_qini_curves_overlay(
    curves: dict[str, tuple[NDArray1D, NDArray1D, float]],
    *,
    title: str = "Qini curves",
    save_path: Path | None = None,
) -> Path | None:
    """Multiple Qini curves on one axis.

    `curves` maps model_label -> (population_share, cumulative_uplift, qini_value).
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    for label, (share, uplift, q) in curves.items():
        ax.plot(share, uplift, lw=2, label=f"{label}  Q={q:+.4f}")
    if curves:
        max_endpoint = max(uplift[-1] for _, uplift, _ in curves.values())
        ax.plot([0, 1], [0, max_endpoint], "--", color="grey", lw=1, label="random")
    ax.set_xlabel("targeted fraction of population")
    ax.set_ylabel("cumulative uplift")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    return _save(fig, save_path)


def plot_cumulative_gain_overlay(
    curves: dict[str, tuple[NDArray1D, NDArray1D, float]],
    *,
    title: str = "Cumulative gain curves",
    save_path: Path | None = None,
) -> Path | None:
    """Multiple cumulative-gain curves on one axis.

    `curves` maps model_label -> (population_share, cumulative_per_capita, auc).
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    endpoints: list[float] = []
    for label, (share, gain, auc) in curves.items():
        ax.plot(share, gain, lw=2, label=f"{label}  AUC={auc:.4f}")
        endpoints.append(float(gain[-1]))
    if endpoints:
        ax.plot([0, 1], [0, max(endpoints)], "--", color="grey", lw=1, label="random")
    ax.set_xlabel("targeted fraction of population")
    ax.set_ylabel("cumulative treated responders / N")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    return _save(fig, save_path)


def plot_policy_value_overlay(
    curves: dict[str, tuple[NDArray1D, NDArray1D]],
    *,
    title: str = "Policy value vs budget",
    save_path: Path | None = None,
) -> Path | None:
    """Multiple policy-value curves on one axis."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for label, (budgets, values) in curves.items():
        ax.plot(budgets, values, marker="o", lw=2, ms=5, label=label)
    ax.set_xlabel("budget (fraction of population treated)")
    ax.set_ylabel("expected outcome E[Y(pi(X))]")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    return _save(fig, save_path)


def plot_bootstrap_distribution(
    boot_values: NDArray1D,
    *,
    point: float | None = None,
    ci: tuple[float, float] | None = None,
    title: str = "Bootstrap distribution",
    save_path: Path | None = None,
) -> Path | None:
    """Histogram of bootstrap statistics with point estimate and CI lines."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(boot_values, bins=50, color="#1f77b4", alpha=0.85, edgecolor="black")
    if point is not None:
        ax.axvline(point, color="red", lw=2, label=f"point = {point:+.4f}")
    if ci is not None:
        ax.axvline(
            ci[0], color="black", ls="--", lw=1, label=f"95% CI = [{ci[0]:+.4f}, {ci[1]:+.4f}]"
        )
        ax.axvline(ci[1], color="black", ls="--", lw=1)
    ax.set_xlabel("bootstrap statistic")
    ax.set_ylabel("count")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    return _save(fig, save_path)


def _save(fig: matplotlib.figure.Figure, save_path: Path | None) -> Path | None:
    if save_path is None:
        return None
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return save_path
