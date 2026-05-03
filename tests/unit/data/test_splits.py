from __future__ import annotations

import numpy as np
import pytest

from tests.fixtures.synthetic import SyntheticDataset, make_uplift_dataset
from uplift_bench.data.splits import make_splits
from uplift_bench.data.validation import DatasetSchema, UpliftDataset, validate_dataframe


def _to_uplift(df_synth: SyntheticDataset, name: str = "toy") -> UpliftDataset:
    schema = DatasetSchema(
        treatment_col="treatment",
        outcome_col="outcome",
        feature_cols=tuple(c for c in df_synth.df.columns if c not in {"treatment", "outcome"}),
    )
    df = validate_dataframe(df_synth.df, schema)
    return UpliftDataset(df=df, schema=schema, name=name)


def test_splits_are_disjoint_and_cover_all_rows() -> None:
    ds = _to_uplift(make_uplift_dataset(n_samples=1000, seed=0))
    s = make_splits(ds, train_frac=0.7, val_frac=0.15, seed=0)

    assert len(np.intersect1d(s.train, s.val)) == 0
    assert len(np.intersect1d(s.train, s.test)) == 0
    assert len(np.intersect1d(s.val, s.test)) == 0

    union = np.concatenate([s.train, s.val, s.test])
    np.testing.assert_array_equal(np.sort(union), np.arange(ds.n))


def test_splits_size_close_to_requested_fractions() -> None:
    ds = _to_uplift(make_uplift_dataset(n_samples=10_000, seed=0))
    s = make_splits(ds, train_frac=0.6, val_frac=0.2, seed=0)
    n_train, n_val, n_test = s.sizes
    assert abs(n_train / 10_000 - 0.60) < 0.01
    assert abs(n_val / 10_000 - 0.20) < 0.01
    assert abs(n_test / 10_000 - 0.20) < 0.01


def test_stratification_preserves_treatment_outcome_marginals() -> None:
    ds = _to_uplift(make_uplift_dataset(n_samples=5000, seed=0))
    s = make_splits(ds, seed=0)
    df = ds.df

    # Each fold's P(T=1) and P(Y=1) should be within ~1.5 pp of full-data marginals.
    full_pt = df["treatment"].mean()
    full_py = df["outcome"].mean()
    for idx in (s.train, s.val, s.test):
        sub = df.iloc[idx]
        assert abs(sub["treatment"].mean() - full_pt) < 0.02
        assert abs(sub["outcome"].mean() - full_py) < 0.02


def test_seed_is_reproducible() -> None:
    ds = _to_uplift(make_uplift_dataset(n_samples=1000, seed=0))
    a = make_splits(ds, seed=11)
    b = make_splits(ds, seed=11)
    np.testing.assert_array_equal(a.train, b.train)
    np.testing.assert_array_equal(a.val, b.val)
    np.testing.assert_array_equal(a.test, b.test)


def test_different_seeds_produce_different_splits() -> None:
    ds = _to_uplift(make_uplift_dataset(n_samples=1000, seed=0))
    a = make_splits(ds, seed=1)
    b = make_splits(ds, seed=2)
    assert not np.array_equal(a.train, b.train)


@pytest.mark.parametrize(
    ("train_frac", "val_frac"),
    [(0.0, 0.2), (1.0, 0.2), (0.6, 0.0), (0.6, 1.0), (0.7, 0.4)],
)
def test_invalid_fractions(train_frac: float, val_frac: float) -> None:
    ds = _to_uplift(make_uplift_dataset(n_samples=200, seed=0))
    with pytest.raises(ValueError):
        make_splits(ds, train_frac=train_frac, val_frac=val_frac)
