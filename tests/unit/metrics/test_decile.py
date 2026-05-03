from __future__ import annotations

import numpy as np
import pytest

from tests.fixtures.synthetic import make_uplift_dataset
from uplift_bench.metrics.decile import decile_table


def test_table_shape_and_columns() -> None:
    ds = make_uplift_dataset(n_samples=2000, seed=0)
    df = decile_table(
        ds.true_uplift, ds.df["treatment"].to_numpy(), ds.df["outcome"].to_numpy(), n_buckets=10
    )
    assert len(df) == 10
    assert list(df["bucket"]) == list(range(1, 11))
    assert {"n_treat", "n_ctrl", "uplift"} <= set(df.columns)


def test_oracle_buckets_are_roughly_monotone_decreasing() -> None:
    ds = make_uplift_dataset(n_samples=10_000, seed=2)
    df = decile_table(
        ds.true_uplift, ds.df["treatment"].to_numpy(), ds.df["outcome"].to_numpy(), n_buckets=10
    )
    upl = df["uplift"].to_numpy()
    # Spearman-style monotone test: top bucket > bottom bucket by a
    # comfortable margin.
    assert upl[0] > upl[-1] + 0.02


def test_buckets_partition_population() -> None:
    ds = make_uplift_dataset(n_samples=997, seed=0)  # not divisible by 10
    df = decile_table(
        ds.true_uplift, ds.df["treatment"].to_numpy(), ds.df["outcome"].to_numpy(), n_buckets=10
    )
    assert df["n_total"].sum() == ds.n


@pytest.mark.parametrize("nb", [1, 0, -1])
def test_invalid_n_buckets(nb: int) -> None:
    with pytest.raises(ValueError, match="n_buckets"):
        decile_table(np.array([1.0]), np.array([1]), np.array([1]), n_buckets=nb)


def test_too_few_rows_raises() -> None:
    score = np.arange(5, dtype=float)
    t = np.array([0, 1, 0, 1, 0])
    y = np.array([1, 0, 1, 0, 1])
    with pytest.raises(ValueError, match="not enough"):
        decile_table(score, t, y, n_buckets=10)
