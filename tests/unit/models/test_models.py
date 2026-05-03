"""Smoke + correctness tests for every meta-learner.

The bar each model must clear:
  1. fit() on a synthetic dataset with known true uplift.
  2. predict_uplift() returns finite values of the right shape.
  3. The Spearman rank correlation between predicted and true uplift is
     >= a per-model threshold. Thresholds are intentionally lenient — we
     are testing that the implementation is *plausible*, not winning a
     leaderboard. A regression below threshold means something broke.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from tests.fixtures.synthetic import make_uplift_dataset
from uplift_bench.models.factory import MODEL_REGISTRY, make_model
from uplift_bench.models.s_learner import SLearner
from uplift_bench.models.t_learner import TLearner

# Per-model minimum Spearman rank correlation between predicted and true
# uplift on a 4k-sample synthetic dataset with strong signal. These are
# generous; they catch broken models, not weak ones.
RANK_CORR_THRESHOLDS = {
    "s_learner": 0.20,
    "t_learner": 0.25,
    "x_learner": 0.25,
    # Cross-fit learners need more boosting iterations to shine; the
    # 120-iteration fast preset isn't enough for them to win on a small
    # synthetic. Threshold here is a regression guard, not a leaderboard.
    "r_learner": 0.20,
    "dr_learner": 0.20,
    "class_transformation": 0.20,
    "causal_forest": 0.25,
}


@pytest.fixture(scope="module")
def synth_train_test() -> tuple[
    pd.DataFrame,
    np.ndarray,
    np.ndarray,
    pd.DataFrame,
    np.ndarray,
]:
    """Pre-build a single train/test pair shared across model tests."""
    train = make_uplift_dataset(
        n_samples=4000,
        n_features=8,
        n_informative_uplift=4,
        treatment_share=0.5,
        seed=42,
    )
    test = make_uplift_dataset(
        n_samples=2000,
        n_features=8,
        n_informative_uplift=4,
        treatment_share=0.5,
        seed=43,
    )
    feature_cols = train.feature_names
    X_train = train.df[feature_cols].copy()
    X_test = test.df[feature_cols].copy()
    return (
        X_train,
        train.df["treatment"].to_numpy(),
        train.df["outcome"].to_numpy(),
        X_test,
        test.true_uplift,
    )


@pytest.mark.parametrize("model_name", sorted(MODEL_REGISTRY))
def test_each_model_ranks_uplift_above_threshold(
    model_name: str,
    synth_train_test: tuple[
        pd.DataFrame,
        np.ndarray,
        np.ndarray,
        pd.DataFrame,
        np.ndarray,
    ],
) -> None:
    X_train, t_train, y_train, X_test, true_tau = synth_train_test
    # Fast settings so the test suite stays under a minute. CatBoost +
    # LightGBM both honour reduced n_estimators via base_params.
    if model_name in {"causal_forest"}:
        model = make_model(model_name, seed=42, n_estimators=80, min_samples_leaf=20)
    else:
        model = make_model(
            model_name,
            seed=42,
            base_params={"iterations": 120, "n_estimators": 120},
        )

    model.fit(X_train, t_train, y_train)
    preds = model.predict_uplift(X_test)

    assert len(preds) == len(X_test)
    assert np.all(np.isfinite(preds))

    rho, _ = spearmanr(preds, true_tau)
    assert rho >= RANK_CORR_THRESHOLDS[model_name], (
        f"{model_name}: rank correlation {rho:.3f} < threshold {RANK_CORR_THRESHOLDS[model_name]}"
    )


def test_predict_before_fit_raises() -> None:
    model = SLearner(base_learner="logreg", outcome="classification")
    X = pd.DataFrame({"a": [1.0, 2.0]})
    with pytest.raises(RuntimeError, match="fit"):
        model.predict_uplift(X)


def test_t_learner_requires_both_arms() -> None:
    model = TLearner(base_learner="logreg")
    X = pd.DataFrame({"a": np.arange(20.0)})
    t = np.zeros(20, dtype=int)
    y = np.zeros(20, dtype=int)
    with pytest.raises(ValueError, match="each arm"):
        model.fit(X, t, y)


def test_factory_unknown_model_raises() -> None:
    with pytest.raises(ValueError, match="unknown model"):
        make_model("not_a_real_learner")


def test_factory_returns_correct_class() -> None:
    m = make_model("s_learner", base_learner="logreg")
    assert isinstance(m, SLearner)
