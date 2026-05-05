from __future__ import annotations

import numpy as np
import pytest

from uplift_bench.data.synthetic import make_uplift_dataset
from uplift_bench.metrics.bootstrap import bootstrap_ci, paired_bootstrap_test
from uplift_bench.metrics.qini import qini_coefficient


def test_ci_brackets_point_estimate() -> None:
    ds = make_uplift_dataset(n_samples=2000, seed=0)
    ci = bootstrap_ci(
        qini_coefficient,
        ds.true_uplift,
        ds.df["treatment"].to_numpy(),
        ds.df["outcome"].to_numpy(),
        n_boot=200,
        method="bca",
        seed=42,
    )
    assert ci.lower <= ci.point <= ci.upper
    # Width is positive but not absurd.
    assert 0 < (ci.upper - ci.lower) < 0.2


def test_percentile_and_bca_close_on_well_behaved_metric() -> None:
    """For ~symmetric bootstrap distributions, BCa ≈ percentile."""
    ds = make_uplift_dataset(n_samples=2000, seed=1)
    score = ds.true_uplift
    treatment = ds.df["treatment"].to_numpy()
    outcome = ds.df["outcome"].to_numpy()
    pct = bootstrap_ci(
        qini_coefficient, score, treatment, outcome, n_boot=200, seed=7, method="percentile"
    )
    bca = bootstrap_ci(
        qini_coefficient, score, treatment, outcome, n_boot=200, seed=7, method="bca"
    )
    # BCa shifts the bounds but on a clean DGP they should overlap heavily.
    assert abs(pct.lower - bca.lower) < 0.05
    assert abs(pct.upper - bca.upper) < 0.05


def test_seed_reproducibility() -> None:
    ds = make_uplift_dataset(n_samples=1500, seed=2)
    a = bootstrap_ci(
        qini_coefficient,
        ds.true_uplift,
        ds.df["treatment"].to_numpy(),
        ds.df["outcome"].to_numpy(),
        n_boot=100,
        seed=11,
        method="percentile",
    )
    b = bootstrap_ci(
        qini_coefficient,
        ds.true_uplift,
        ds.df["treatment"].to_numpy(),
        ds.df["outcome"].to_numpy(),
        n_boot=100,
        seed=11,
        method="percentile",
    )
    assert a.lower == b.lower
    assert a.upper == b.upper
    assert a.point == b.point


def test_paired_test_detects_real_difference() -> None:
    """Oracle should significantly beat random under paired bootstrap.

    Need a reasonable n and bigger signal — Qini values on binary outcome
    fixtures are tiny and the bootstrap variance can swallow small gaps.
    """
    ds = make_uplift_dataset(
        n_samples=8000,
        n_informative_uplift=5,
        seed=0,
    )
    t = ds.df["treatment"].to_numpy()
    y = ds.df["outcome"].to_numpy()
    rng = np.random.default_rng(0)
    res = paired_bootstrap_test(
        qini_coefficient,
        score_a=ds.true_uplift,
        score_b=rng.standard_normal(ds.n),
        treatment=t,
        outcome=y,
        n_boot=400,
        seed=3,
    )
    assert res["observed_diff"] > 0
    assert res["p_value_one_sided"] < 0.05


def test_paired_test_returns_high_p_for_no_real_difference() -> None:
    ds = make_uplift_dataset(n_samples=2000, seed=1)
    t = ds.df["treatment"].to_numpy()
    y = ds.df["outcome"].to_numpy()
    rng = np.random.default_rng(0)
    a = rng.standard_normal(ds.n)
    b = rng.standard_normal(ds.n)
    res = paired_bootstrap_test(
        qini_coefficient,
        score_a=a,
        score_b=b,
        treatment=t,
        outcome=y,
        n_boot=200,
        seed=3,
    )
    # Two random rankings — no reason to prefer one over the other.
    assert res["p_value_one_sided"] > 0.1


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.1])
def test_invalid_alpha(alpha: float) -> None:
    ds = make_uplift_dataset(n_samples=200, seed=0)
    with pytest.raises(ValueError, match="alpha"):
        bootstrap_ci(
            qini_coefficient,
            ds.true_uplift,
            ds.df["treatment"].to_numpy(),
            ds.df["outcome"].to_numpy(),
            n_boot=100,
            alpha=alpha,
        )


def test_n_boot_too_small() -> None:
    ds = make_uplift_dataset(n_samples=200, seed=0)
    with pytest.raises(ValueError, match="n_boot"):
        bootstrap_ci(
            qini_coefficient,
            ds.true_uplift,
            ds.df["treatment"].to_numpy(),
            ds.df["outcome"].to_numpy(),
            n_boot=10,
        )


def test_paired_lengths_must_match() -> None:
    with pytest.raises(ValueError, match="same length"):
        paired_bootstrap_test(
            qini_coefficient,
            score_a=np.array([1.0]),
            score_b=np.array([1.0, 2.0]),
            treatment=np.array([0, 1]),
            outcome=np.array([1, 0]),
            n_boot=100,
        )
