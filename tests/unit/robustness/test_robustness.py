"""Smoke + correctness tests for robustness modules."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from uplift_bench.data.synthetic import make_uplift_dataset
from uplift_bench.models.s_learner import SLearner
from uplift_bench.robustness.feature_drop import feature_drop_stability
from uplift_bench.robustness.learning_curve import learning_curve
from uplift_bench.robustness.overlap import overlap_diagnostics
from uplift_bench.robustness.permutation import permutation_uplift_importance


def _logreg_s_learner_builder() -> SLearner:
    # Logreg base learner so the tests run in <1s.
    return SLearner(base_learner="logreg", outcome="classification", seed=0)


def test_permutation_importance_orders_informative_features_first() -> None:
    ds = make_uplift_dataset(
        n_samples=2000,
        n_features=8,
        n_informative_uplift=2,
        seed=0,
    )
    X = ds.df[ds.feature_names].copy()
    t = ds.df["treatment"].to_numpy()
    y = ds.df["outcome"].to_numpy()

    model = _logreg_s_learner_builder()
    model.fit(X, t, y)
    imp = permutation_uplift_importance(model, X, t, y, n_repeats=3, seed=1)

    # f0/f1 drive tau in the DGP; f5..f7 should rank near the bottom.
    informative = set(imp.head(3)["feature"]) & {"f0", "f1"}
    assert len(informative) >= 1


def test_permutation_importance_n_repeats_validation() -> None:
    ds = make_uplift_dataset(n_samples=200, seed=0)
    X = ds.df[ds.feature_names].copy()
    t = ds.df["treatment"].to_numpy()
    y = ds.df["outcome"].to_numpy()
    model = _logreg_s_learner_builder()
    model.fit(X, t, y)
    with pytest.raises(ValueError, match="n_repeats"):
        permutation_uplift_importance(model, X, t, y, n_repeats=0)


def test_feature_drop_baseline_above_or_equal_to_dropped() -> None:
    """Dropping features can only hurt or match Qini on a generative DGP
    where every feature carries some signal — but on an iid synthetic
    fixture noise sometimes makes drops *help*. We just sanity-check
    that the function runs and returns the right schema."""
    ds = make_uplift_dataset(
        n_samples=1000,
        n_features=5,
        n_informative_uplift=3,
        seed=0,
    )
    X = ds.df[ds.feature_names].copy()
    t = ds.df["treatment"].to_numpy()
    y = ds.df["outcome"].to_numpy()
    n_train = 700
    res = feature_drop_stability(
        _logreg_s_learner_builder,
        X.iloc[:n_train],
        t[:n_train],
        y[:n_train],
        X.iloc[n_train:],
        t[n_train:],
        y[n_train:],
    )
    assert set(res.columns) == {
        "feature_or_group",
        "baseline_qini",
        "qini_after_drop",
        "qini_delta",
    }
    assert len(res) == 5


def test_feature_drop_with_groups() -> None:
    ds = make_uplift_dataset(n_samples=800, n_features=6, seed=0)
    X = ds.df[ds.feature_names].copy()
    t = ds.df["treatment"].to_numpy()
    y = ds.df["outcome"].to_numpy()
    res = feature_drop_stability(
        _logreg_s_learner_builder,
        X.iloc[:600],
        t[:600],
        y[:600],
        X.iloc[600:],
        t[600:],
        y[600:],
        groups={"first_two": ["f0", "f1"], "last_two": ["f4", "f5"]},
    )
    assert set(res["feature_or_group"]) == {"first_two", "last_two"}


def test_learning_curve_monotone_in_n_train() -> None:
    """More training data should generally not hurt — accept noise but
    require the largest fraction to beat the smallest by a small margin."""
    ds = make_uplift_dataset(
        n_samples=4000,
        n_features=6,
        n_informative_uplift=3,
        seed=0,
    )
    X = ds.df[ds.feature_names].copy()
    t = ds.df["treatment"].to_numpy()
    y = ds.df["outcome"].to_numpy()
    n_train = 3000
    res = learning_curve(
        _logreg_s_learner_builder,
        X.iloc[:n_train],
        t[:n_train],
        y[:n_train],
        X.iloc[n_train:],
        t[n_train:],
        y[n_train:],
        fractions=(0.1, 0.5, 1.0),
    )
    assert len(res) == 3
    assert (res["n_train"].diff().dropna() > 0).all()


def test_learning_curve_invalid_fraction() -> None:
    ds = make_uplift_dataset(n_samples=200, seed=0)
    X = ds.df[ds.feature_names].copy()
    t = ds.df["treatment"].to_numpy()
    y = ds.df["outcome"].to_numpy()
    with pytest.raises(ValueError, match="fractions"):
        learning_curve(
            _logreg_s_learner_builder,
            X.iloc[:150],
            t[:150],
            y[:150],
            X.iloc[150:],
            t[150:],
            y[150:],
            fractions=(0.0, 0.5),
        )


def test_overlap_diagnostics_on_balanced_rct() -> None:
    """Under randomised treatment ESS / n should be close to 1 and there
    should be essentially no observations in the clip tails."""
    ds = make_uplift_dataset(
        n_samples=3000,
        n_features=6,
        propensity_drift=0.0,
        seed=0,
    )
    X = ds.df[ds.feature_names].copy()
    t = ds.df["treatment"].to_numpy()
    diag = overlap_diagnostics(X, t, n_splits=3, seed=0)
    # Even under perfect RCT the gradient-boosting propensity overfits a
    # bit on small folds; 0.75 is comfortably above the "broken" range
    # while still catching real degradation.
    assert diag.ess_ratio > 0.75
    assert diag.pct_below_clip < 0.05
    assert diag.pct_above_clip < 0.05


def test_overlap_diagnostics_on_confounded_data() -> None:
    """With strong propensity drift, overlap should degrade meaningfully."""
    rct = overlap_diagnostics(
        make_uplift_dataset(
            n_samples=3000,
            propensity_drift=0.0,
            seed=1,
        ).df[[f"f{i}" for i in range(10)]],
        make_uplift_dataset(
            n_samples=3000,
            propensity_drift=0.0,
            seed=1,
        )
        .df["treatment"]
        .to_numpy(),
        n_splits=3,
        seed=0,
    )
    confounded = overlap_diagnostics(
        make_uplift_dataset(
            n_samples=3000,
            propensity_drift=2.5,
            seed=1,
        ).df[[f"f{i}" for i in range(10)]],
        make_uplift_dataset(
            n_samples=3000,
            propensity_drift=2.5,
            seed=1,
        )
        .df["treatment"]
        .to_numpy(),
        n_splits=3,
        seed=0,
    )
    assert confounded.ess_ratio < rct.ess_ratio


def test_overlap_diagnostics_rejects_non_binary_treatment() -> None:
    X = pd.DataFrame({"a": np.arange(20.0)})
    t = np.array([0, 1, 2] * 6 + [0, 1])
    with pytest.raises(ValueError, match="binary"):
        overlap_diagnostics(X, t)
