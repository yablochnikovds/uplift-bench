"""Bar chart comparing model Qini coefficients with CIs."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_qini_comparison(
    df: pd.DataFrame,
    *,
    point_col: str = "qini",
    lower_col: str = "qini_ci_lower",
    upper_col: str = "qini_ci_upper",
    label_col: str = "model",
    title: str = "Qini by model",
    save_path: Path | None = None,
) -> Path | None:
    """Sorted horizontal bar chart with error bars from BCa CIs."""
    df_sorted = df.sort_values(point_col)
    fig, ax = plt.subplots(figsize=(7, max(2.5, 0.45 * len(df_sorted) + 1)))
    y = range(len(df_sorted))
    point = df_sorted[point_col].to_numpy()
    err = [point - df_sorted[lower_col].to_numpy(), df_sorted[upper_col].to_numpy() - point]
    ax.barh(list(y), point, xerr=err, color="#1f77b4", alpha=0.85, ecolor="black", capsize=4)
    ax.set_yticks(list(y))
    ax.set_yticklabels(df_sorted[label_col].astype(str))
    ax.axvline(0.0, color="black", lw=0.8)
    ax.set_xlabel(point_col)
    ax.set_title(title)
    fig.tight_layout()

    if save_path is None:
        return None
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    return save_path
