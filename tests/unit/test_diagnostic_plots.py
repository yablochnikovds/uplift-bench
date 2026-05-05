"""Smoke tests for the diagnostic plot module.

Plots are visual artefacts; we verify only that each function runs
without exception, produces a non-empty PNG, and accepts the documented
shapes. Visual correctness is checked by eye in the notebooks.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from uplift_bench.data.synthetic import make_uplift_dataset
from uplift_bench.metrics.decile import decile_table
from uplift_bench.viz.diagnostic_plots import (
    plot_bootstrap_distribution,
    plot_calibration,
    plot_decile_uplift,
    plot_learning_curve,
    plot_permutation_importance,
    plot_propensity_histogram,
    plot_qini_curves_overlay,
    plot_qini_heatmap,
)


def _toy_data(n: int = 800) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ds = make_uplift_dataset(n_samples=n, seed=0)
    return (
        ds.true_uplift,
        ds.df["treatment"].to_numpy(),
        ds.df["outcome"].to_numpy(),
    )


def test_plot_calibration_writes_png(tmp_path: Path) -> None:
    score, t, y = _toy_data()
    out = plot_calibration(score, t, y, save_path=tmp_path / "cal.png")
    assert out is not None
    assert out.exists()
    assert out.stat().st_size > 1000


def test_plot_decile_uplift(tmp_path: Path) -> None:
    score, t, y = _toy_data(n=2000)
    df = decile_table(score, t, y, n_buckets=10)
    out = plot_decile_uplift(df, save_path=tmp_path / "dec.png")
    assert out is not None
    assert out.exists()


def test_plot_propensity_histogram(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    treated = rng.beta(5, 3, 500)
    control = rng.beta(3, 5, 500)
    out = plot_propensity_histogram(treated, control, save_path=tmp_path / "prop.png")
    assert out is not None
    assert out.exists()


def test_plot_learning_curve(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "fraction": [0.25, 0.5, 0.75, 1.0],
            "n_train": [250, 500, 750, 1000],
            "qini": [0.05, 0.07, 0.09, 0.10],
        }
    )
    out = plot_learning_curve(df, save_path=tmp_path / "lc.png")
    assert out is not None
    assert out.exists()


def test_plot_permutation_importance(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "feature": [f"f{i}" for i in range(8)],
            "mean_qini_drop": np.linspace(0.001, 0.05, 8),
            "std_qini_drop": np.full(8, 0.005),
            "baseline_qini": np.full(8, 0.1),
            "n_repeats": np.full(8, 5),
        }
    )
    out = plot_permutation_importance(df, save_path=tmp_path / "perm.png")
    assert out is not None
    assert out.exists()


def test_plot_qini_heatmap(tmp_path: Path) -> None:
    summary = pd.DataFrame(
        {
            "model": ["s_learner", "t_learner", "x_learner", "s_learner", "t_learner", "x_learner"],
            "dataset": ["hillstrom"] * 3 + ["criteo"] * 3,
            "qini_mean": [0.2, 0.18, 0.22, 0.15, 0.14, 0.17],
        }
    )
    out = plot_qini_heatmap(summary, save_path=tmp_path / "heatmap.png")
    assert out is not None
    assert out.exists()


def test_plot_qini_curves_overlay(tmp_path: Path) -> None:
    share = np.linspace(0, 1, 50)
    curves = {
        "s_learner": (share, share * 0.04, 0.012),
        "dr_learner": (share, share * 0.05, 0.018),
    }
    out = plot_qini_curves_overlay(curves, save_path=tmp_path / "overlay.png")
    assert out is not None
    assert out.exists()


def test_plot_bootstrap_distribution(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    boots = rng.normal(0.05, 0.01, 500)
    out = plot_bootstrap_distribution(
        boots,
        point=0.05,
        ci=(0.03, 0.07),
        save_path=tmp_path / "boot.png",
    )
    assert out is not None
    assert out.exists()
