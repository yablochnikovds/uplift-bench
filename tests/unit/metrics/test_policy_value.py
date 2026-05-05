from __future__ import annotations

import numpy as np
import pytest

from uplift_bench.data.synthetic import make_uplift_dataset
from uplift_bench.metrics.policy_value import policy_value_at, policy_value_curve


def test_curve_endpoints_match_treat_none_and_treat_all() -> None:
    ds = make_uplift_dataset(n_samples=3000, seed=0)
    t = ds.df["treatment"].to_numpy()
    y = ds.df["outcome"].to_numpy()
    res = policy_value_curve(
        ds.true_uplift,
        t,
        y,
        budgets=[0.0, 0.5, 1.0],
    )
    assert res.policy_values[0] == res.treat_none_value
    assert res.policy_values[-1] == res.treat_all_value


def test_oracle_policy_beats_random_at_top_decile() -> None:
    """A targeting policy based on the true uplift should produce higher
    expected outcome than random targeting at the same budget."""
    ds = make_uplift_dataset(n_samples=5000, seed=2)
    t = ds.df["treatment"].to_numpy()
    y = ds.df["outcome"].to_numpy()
    rng = np.random.default_rng(0)

    oracle_v = policy_value_at(ds.true_uplift, t, y, budget=0.10)
    random_v = policy_value_at(rng.standard_normal(ds.n), t, y, budget=0.10)
    assert oracle_v > random_v


def test_invalid_budget_raises() -> None:
    ds = make_uplift_dataset(n_samples=200, seed=0)
    t = ds.df["treatment"].to_numpy()
    y = ds.df["outcome"].to_numpy()
    with pytest.raises(ValueError, match="budget"):
        policy_value_curve(ds.true_uplift, t, y, budgets=[1.5])


def test_one_arm_only_raises() -> None:
    score = np.arange(50, dtype=float)
    t = np.zeros(50, dtype=int)  # nobody treated
    y = np.zeros(50, dtype=int)
    with pytest.raises(ValueError, match="both arms"):
        policy_value_curve(score, t, y, budgets=[0.5])
