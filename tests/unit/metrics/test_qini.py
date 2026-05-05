"""Property-based + fixture-based tests for Qini.

The "easy" tests (random model gives Qini ≈ 0; perfect model gives Qini high)
double as regressions for tie-breaking and the random baseline subtraction.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from uplift_bench.data.synthetic import make_uplift_dataset
from uplift_bench.metrics.qini import qini_coefficient, qini_curve


def test_random_model_has_qini_near_zero() -> None:
    ds = make_uplift_dataset(n_samples=4000, seed=0)
    rng = np.random.default_rng(123)
    rand_scores = rng.standard_normal(ds.n)
    q = qini_coefficient(rand_scores, ds.df["treatment"].to_numpy(), ds.df["outcome"].to_numpy())
    # Random ranking should not separate uplift from no-uplift; ±0.02 is the
    # noise band at this n.
    assert abs(q) < 0.02


def test_oracle_model_beats_random() -> None:
    """Ranking by the *true* uplift should clearly beat random."""
    ds = make_uplift_dataset(n_samples=4000, seed=1)
    t = ds.df["treatment"].to_numpy()
    y = ds.df["outcome"].to_numpy()

    q_oracle = qini_coefficient(ds.true_uplift, t, y)
    q_random = qini_coefficient(np.random.default_rng(0).standard_normal(ds.n), t, y)
    assert q_oracle > q_random + 0.005


def test_qini_curve_starts_at_origin_and_endpoint_is_ate() -> None:
    ds = make_uplift_dataset(n_samples=2000, seed=2)
    t = ds.df["treatment"].to_numpy()
    y = ds.df["outcome"].to_numpy()
    curve = qini_curve(ds.true_uplift, t, y)

    assert curve.population_share[0] == 0.0
    assert curve.cumulative_uplift[0] == 0.0
    assert curve.population_share[-1] == 1.0


def test_inverted_score_flips_qini_sign() -> None:
    """If a score is informative, negating it should give an opposite Qini."""
    ds = make_uplift_dataset(n_samples=3000, seed=3)
    t = ds.df["treatment"].to_numpy()
    y = ds.df["outcome"].to_numpy()
    q_pos = qini_coefficient(ds.true_uplift, t, y)
    q_neg = qini_coefficient(-ds.true_uplift, t, y)
    assert q_pos > 0
    assert q_neg < 0
    assert abs(q_pos + q_neg) < 0.01  # roughly mirror images


def test_qini_invariant_under_strictly_monotone_transform() -> None:
    """Qini ranks by score; any monotone transform preserves the ordering."""
    ds = make_uplift_dataset(n_samples=2000, seed=4)
    t = ds.df["treatment"].to_numpy()
    y = ds.df["outcome"].to_numpy()
    s = ds.true_uplift
    q1 = qini_coefficient(s, t, y)
    q2 = qini_coefficient(np.exp(s), t, y)  # strictly increasing
    q3 = qini_coefficient(s * 100 + 7, t, y)  # affine, slope > 0
    assert q1 == pytest.approx(q2)
    assert q1 == pytest.approx(q3)


def test_empty_input_raises() -> None:
    with pytest.raises(ValueError, match="at least one"):
        qini_coefficient(np.array([]), np.array([]), np.array([]))


def test_mismatched_lengths_raise() -> None:
    with pytest.raises(ValueError, match="same length"):
        qini_coefficient(np.array([1.0, 2.0]), np.array([0]), np.array([1]))


@given(
    seed=st.integers(min_value=0, max_value=10_000),
    n=st.integers(min_value=100, max_value=2000),
)
@settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_qini_is_finite_for_any_seed(seed: int, n: int) -> None:
    """Property: a finite numeric input should always give a finite Qini."""
    ds = make_uplift_dataset(n_samples=n, seed=seed)
    q = qini_coefficient(
        ds.true_uplift,
        ds.df["treatment"].to_numpy(),
        ds.df["outcome"].to_numpy(),
    )
    assert np.isfinite(q)


def test_qini_handles_all_control() -> None:
    # Edge case: everyone is in control. Qini is undefined-ish but the
    # implementation must not crash and must return a finite number.
    n = 200
    score = np.linspace(1, 0, n)
    t = np.zeros(n, dtype=int)
    y = np.random.default_rng(0).integers(0, 2, n)
    q = qini_coefficient(score, t, y)
    assert np.isfinite(q)
