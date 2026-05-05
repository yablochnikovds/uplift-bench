from __future__ import annotations

import numpy as np
import pytest

from uplift_bench.data.synthetic import make_uplift_dataset
from uplift_bench.metrics.cumulative_gain import cumulative_gain_curve


def test_endpoints() -> None:
    ds = make_uplift_dataset(n_samples=2000, seed=0)
    res = cumulative_gain_curve(
        ds.true_uplift,
        ds.df["treatment"].to_numpy(),
        ds.df["outcome"].to_numpy(),
    )
    assert res.population_share[0] == 0.0
    assert res.population_share[-1] == 1.0
    assert res.cumulative_responders_per_capita[0] == 0.0
    # Endpoint = total treated responders / N.
    expected_end = float(
        ((ds.df["treatment"] == 1) & (ds.df["outcome"] == 1)).sum() / ds.n,
    )
    np.testing.assert_allclose(
        res.cumulative_responders_per_capita[-1],
        expected_end,
        atol=1e-9,
    )


def test_oracle_beats_random() -> None:
    """Ranking by true uplift should give a higher cumulative-gain AUC
    than random ranking, when there's any treatment effect signal."""
    ds = make_uplift_dataset(n_samples=4000, seed=1)
    t = ds.df["treatment"].to_numpy()
    y = ds.df["outcome"].to_numpy()
    rng = np.random.default_rng(0)
    oracle = cumulative_gain_curve(ds.true_uplift, t, y).auc
    rand = cumulative_gain_curve(rng.standard_normal(ds.n), t, y).auc
    assert oracle > rand


def test_empty_input_raises() -> None:
    with pytest.raises(ValueError, match="at least one"):
        cumulative_gain_curve(np.array([]), np.array([]), np.array([]))
