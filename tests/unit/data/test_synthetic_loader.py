"""Tests for the SyntheticLoader shim."""

from __future__ import annotations

import numpy as np

from uplift_bench.data.factory import make_loader
from uplift_bench.data.synthetic_loader import SyntheticLoader


def test_synthetic_loader_returns_validated_dataset() -> None:
    loader = SyntheticLoader(n_samples=400, n_features=4, seed=0)
    ds = loader.load()
    assert ds.n == 400
    assert set(ds.df["treatment"].unique()) <= {0, 1}
    assert set(ds.df["outcome"].unique()) <= {0, 1}
    assert tuple(ds.schema.feature_cols) == ("f0", "f1", "f2", "f3")
    # Synthetic loader produces a deterministic-from-seed fingerprint.
    assert ds.source_hash != ""
    assert len(ds.source_hash) == 64


def test_synthetic_loader_via_factory() -> None:
    loader = make_loader(
        "synthetic", data_dir="data", n_samples=200, n_features=3, seed=1,
    )
    assert isinstance(loader, SyntheticLoader)
    ds = loader.load()
    assert ds.n == 200


def test_synthetic_loader_seeded_reproducibility() -> None:
    a = SyntheticLoader(n_samples=300, seed=7).load()
    b = SyntheticLoader(n_samples=300, seed=7).load()
    np.testing.assert_array_equal(a.df.values, b.df.values)
    assert a.source_hash == b.source_hash


def test_synthetic_loader_can_train_a_real_model() -> None:
    """End-to-end: synthetic loader → S-learner → predict. Sanity check."""
    from uplift_bench.models.s_learner import SLearner  # noqa: PLC0415

    loader = SyntheticLoader(n_samples=600, n_features=5, seed=2)
    ds = loader.load()
    model = SLearner(base_learner="logreg", base_params={"max_iter": 300})
    model.fit(ds.X, ds.t, ds.y)
    preds = model.predict_uplift(ds.X)
    assert preds.shape == (600,)
    assert np.all(np.isfinite(preds))
