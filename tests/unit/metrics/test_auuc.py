from __future__ import annotations

import numpy as np
import pytest

from uplift_bench.data.synthetic import make_uplift_dataset
from uplift_bench.metrics.auuc import auuc


def test_random_model_has_lower_auuc_than_oracle() -> None:
    """Random ranking shouldn't be near oracle on AUUC.

    Note we don't bound |normalized| < epsilon — the per-sample
    normalisation by the perfect-ranking area is itself noisy on small
    samples, so the right comparison is relative, not absolute.
    """
    ds = make_uplift_dataset(n_samples=5000, seed=0)
    t = ds.df["treatment"].to_numpy()
    y = ds.df["outcome"].to_numpy()
    rng = np.random.default_rng(0)
    rand = auuc(rng.standard_normal(ds.n), t, y).auuc_normalized
    oracle = auuc(ds.true_uplift, t, y).auuc_normalized
    assert oracle > rand + 0.05


def test_oracle_beats_random_in_auuc() -> None:
    ds = make_uplift_dataset(n_samples=3000, seed=1)
    t = ds.df["treatment"].to_numpy()
    y = ds.df["outcome"].to_numpy()
    a_oracle = auuc(ds.true_uplift, t, y).auuc_normalized
    a_rand = auuc(np.random.default_rng(0).standard_normal(ds.n), t, y).auuc_normalized
    assert a_oracle > a_rand


def test_curve_endpoints() -> None:
    ds = make_uplift_dataset(n_samples=1500, seed=2)
    res = auuc(ds.true_uplift, ds.df["treatment"].to_numpy(), ds.df["outcome"].to_numpy())
    assert res.population_share[0] == 0.0
    assert res.population_share[-1] == 1.0


def test_empty_input_raises() -> None:
    with pytest.raises(ValueError, match="at least one"):
        auuc(np.array([]), np.array([]), np.array([]))
