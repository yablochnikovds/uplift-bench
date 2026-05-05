from __future__ import annotations

import numpy as np
import pytest

from uplift_bench.data.synthetic import make_uplift_dataset
from uplift_bench.metrics.uplift_at_k import uplift_at_k


def test_oracle_top_decile_lifts_more_than_random_top_decile() -> None:
    ds = make_uplift_dataset(n_samples=8000, seed=0)
    t = ds.df["treatment"].to_numpy()
    y = ds.df["outcome"].to_numpy()
    rng = np.random.default_rng(0)
    u_oracle = uplift_at_k(ds.true_uplift, t, y, k=0.10)
    u_rand = uplift_at_k(rng.standard_normal(ds.n), t, y, k=0.10)
    assert u_oracle > u_rand


def test_uplift_at_full_population_equals_overall_diff_of_means() -> None:
    """k=1 should reduce to the trivial mean(y|T=1) - mean(y|T=0)."""
    ds = make_uplift_dataset(n_samples=2000, seed=2)
    t = ds.df["treatment"].to_numpy()
    y = ds.df["outcome"].to_numpy()
    score = np.zeros(ds.n)  # ranking irrelevant when k=1
    naive = float(y[t == 1].mean() - y[t == 0].mean())
    assert uplift_at_k(score, t, y, k=1.0) == pytest.approx(naive)


@pytest.mark.parametrize("k", [-0.1, 0.0, 1.5])
def test_invalid_k_raises(k: float) -> None:
    with pytest.raises(ValueError, match="k must be"):
        uplift_at_k(np.array([1.0]), np.array([1]), np.array([1]), k=k)


def test_returns_nan_when_top_k_has_only_one_arm() -> None:
    score = np.arange(100, dtype=float)
    # Top 5% is rows 95..99; force them all to treatment.
    t = np.zeros(100, dtype=int)
    t[95:] = 1
    y = np.zeros(100, dtype=int)
    assert np.isnan(uplift_at_k(score, t, y, k=0.05))


def test_empty_input_raises() -> None:
    with pytest.raises(ValueError, match="at least one"):
        uplift_at_k(np.array([]), np.array([]), np.array([]), k=0.1)
