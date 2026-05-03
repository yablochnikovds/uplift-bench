"""Histogram of predicted individual uplift."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from uplift_bench.metrics._common import NDArray1D


def plot_uplift_distribution(
    score: NDArray1D,
    *,
    title: str = "Predicted uplift distribution",
    save_path: Path | None = None,
) -> Path | None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(score, bins=60, color="#1f77b4", alpha=0.85)
    ax.axvline(0.0, color="black", ls="--", lw=1, label="zero uplift")
    ax.axvline(float(np.mean(score)), color="red", ls=":", lw=1, label="mean")
    ax.set_xlabel("predicted treatment effect")
    ax.set_ylabel("count")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    if save_path is None:
        return None
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    return save_path
