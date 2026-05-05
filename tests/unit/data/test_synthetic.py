"""Sanity checks on the synthetic generator.

If these fail, every test that depends on the fixture is meaningless — so
we treat them as the foundation layer.
"""

from __future__ import annotations

import numpy as np
import pytest

from uplift_bench.data.synthetic import make_uplift_dataset


def test_shape_and_columns() -> None:
    ds = make_uplift_dataset(n_samples=500, n_features=8, seed=0)
    assert ds.n == 500
    assert len(ds.feature_names) == 8
    expected_cols = {*ds.feature_names, "treatment", "outcome"}
    assert set(ds.df.columns) == expected_cols
    assert ds.true_uplift.shape == (500,)
    assert ds.true_propensity.shape == (500,)


def test_seed_is_reproducible() -> None:
    a = make_uplift_dataset(n_samples=200, seed=7)
    b = make_uplift_dataset(n_samples=200, seed=7)
    np.testing.assert_array_equal(a.df.values, b.df.values)
    np.testing.assert_array_equal(a.true_uplift, b.true_uplift)


def test_treatment_is_binary_int8() -> None:
    ds = make_uplift_dataset(n_samples=300, seed=1)
    assert ds.df["treatment"].dtype == np.int8
    assert set(ds.df["treatment"].unique()) <= {0, 1}


def test_binary_outcome_is_binary_int8() -> None:
    ds = make_uplift_dataset(n_samples=300, outcome="binary", seed=1)
    assert ds.df["outcome"].dtype == np.int8
    assert set(ds.df["outcome"].unique()) <= {0, 1}


def test_continuous_outcome_is_float() -> None:
    ds = make_uplift_dataset(n_samples=300, outcome="continuous", seed=1)
    assert ds.df["outcome"].dtype == np.float64


def test_treatment_share_respected_when_no_drift() -> None:
    ds = make_uplift_dataset(n_samples=20_000, treatment_share=0.3, propensity_drift=0.0, seed=2)
    realised = ds.df["treatment"].mean()
    # ±2 percentage points at n=20k is well within sampling noise.
    assert 0.28 <= realised <= 0.32


def test_propensity_drift_creates_dependence_on_f0() -> None:
    ds = make_uplift_dataset(n_samples=10_000, propensity_drift=1.5, seed=3)
    # Treated rows should have systematically higher f0 when drift > 0.
    f0_treated = ds.df.loc[ds.df["treatment"] == 1, "f0"].mean()
    f0_control = ds.df.loc[ds.df["treatment"] == 0, "f0"].mean()
    assert f0_treated > f0_control + 0.3


def test_true_uplift_has_meaningful_variance() -> None:
    ds = make_uplift_dataset(n_samples=2000, n_informative_uplift=3, seed=4)
    # Without variance there's nothing to learn; this guards against me
    # accidentally clipping tau in the future.
    assert ds.true_uplift.std() > 0.1
    # Sign flips should exist — a chunk of the population genuinely
    # responds negatively. This is what distinguishes uplift from response.
    neg_share = (ds.true_uplift < 0).mean()
    assert 0.05 < neg_share < 0.5


def test_uninformative_features_carry_no_uplift_signal() -> None:
    ds = make_uplift_dataset(
        n_samples=4000, n_features=10, n_informative_uplift=2, seed=5,
    )
    # f0/f1 drive tau; f5..f9 should be ~uncorrelated with it.
    for j in range(5, 10):
        corr = np.corrcoef(ds.df[f"f{j}"].to_numpy(), ds.true_uplift)[0, 1]
        assert abs(corr) < 0.05


def test_average_treatment_effect_is_positive() -> None:
    ds = make_uplift_dataset(n_samples=10_000, seed=6)
    # The DGP is built so the marginal ATE is positive — checking it stays
    # that way means future tweaks to coefficients won't silently invert
    # what models are supposed to learn.
    assert ds.true_uplift.mean() > 0


@pytest.mark.parametrize("share", [0.0, 1.0, -0.1, 1.5])
def test_rejects_invalid_treatment_share(share: float) -> None:
    with pytest.raises(ValueError, match="treatment_share"):
        make_uplift_dataset(treatment_share=share)


def test_rejects_n_informative_above_n_features() -> None:
    with pytest.raises(ValueError, match="n_informative_uplift"):
        make_uplift_dataset(n_features=3, n_informative_uplift=5)
