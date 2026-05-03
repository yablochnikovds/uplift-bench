"""Repro tests — boring but they have caught real regressions before."""

from __future__ import annotations

import random

import numpy as np
import pytest

from uplift_bench.utils.reproducibility import SeedBundle, seed_everything


def test_returns_seed_bundle_with_all_fields() -> None:
    bundle = seed_everything(123)
    assert isinstance(bundle, SeedBundle)
    assert bundle.python == bundle.numpy == bundle.catboost == 123


def test_numpy_reproducible_after_seed() -> None:
    seed_everything(7)
    a = np.random.rand(5)
    seed_everything(7)
    b = np.random.rand(5)
    np.testing.assert_array_equal(a, b)


def test_python_random_reproducible_after_seed() -> None:
    seed_everything(7)
    a = [random.random() for _ in range(5)]
    seed_everything(7)
    b = [random.random() for _ in range(5)]
    assert a == b


def test_different_seeds_produce_different_streams() -> None:
    seed_everything(1)
    a = np.random.rand(10)
    seed_everything(2)
    b = np.random.rand(10)
    # Astronomically unlikely to collide; if it does we want to know.
    assert not np.allclose(a, b)


@pytest.mark.parametrize("bad", [-1, 2**32, 2**40])
def test_rejects_seeds_outside_numpy_range(bad: int) -> None:
    with pytest.raises(ValueError, match="out of range"):
        seed_everything(bad)
