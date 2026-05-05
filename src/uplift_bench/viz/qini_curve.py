"""Qini curve plotting.

We use matplotlib directly — seaborn would add a dep for almost no value
(the plots are five-line affairs). Pyplot's "Agg" backend is fine for
file output and doesn't need a display.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from uplift_bench.metrics.qini import QiniCurve


def plot_qini_curve(
    curve: QiniCurve, *, title: str = "Qini curve", save_path: Path | None = None
) -> Path | None:
    """Render and save (or return) a Qini curve plot."""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(
        curve.population_share,
        curve.cumulative_uplift,
        lw=2,
        color="#1f77b4",
        label=f"model (Q={curve.qini_coefficient:+.4f})",
    )
    # Random-targeting baseline = straight line from (0,0) to (1, ATE).
    ax.plot([0, 1], [0, curve.cumulative_uplift[-1]], "--", color="grey", lw=1, label="random")
    ax.set_xlabel("targeted fraction of population")
    ax.set_ylabel("cumulative uplift")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()

    if save_path is None:
        return None
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    return save_path


def plot_qini_curves(
    curves: dict[str, QiniCurve],
    *,
    title: str = "Qini curves",
    colors: dict[str, str] | None = None,
    save_path: Path | None = None,
) -> Path | None:
    """Overlay multiple model curves on one axis.

    `colors` optionally maps model label → matplotlib color so the same
    model uses the same colour across multiple plots. Unset labels fall
    back to matplotlib's default cycle.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    color_map = colors or {}
    for label, curve in curves.items():
        ax.plot(
            curve.population_share,
            curve.cumulative_uplift,
            lw=2,
            color=color_map.get(label),
            label=f"{label}  Q={curve.qini_coefficient:+.4f}",
        )
    # Use the largest endpoint for the random baseline.
    if curves:
        max_endpoint = max(c.cumulative_uplift[-1] for c in curves.values())
        ax.plot([0, 1], [0, max_endpoint], "--", color="grey", lw=1, label="random")
    ax.set_xlabel("targeted fraction of population")
    ax.set_ylabel("cumulative uplift")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()

    if save_path is None:
        return None
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    return save_path
