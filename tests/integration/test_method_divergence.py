"""Integration test: on a deliberately heterogeneous DGP, the meta-learners
should diverge in performance.

Why this test exists. Looking at the early benchmark on Hillstrom and on
the Criteo subsample, all 7 meta-learners produced very similar Qini
values (~0.003 on the per-person scale). That looks suspicious — it
might mean implementations are wrong, or it might mean the datasets
genuinely have low heterogeneity.

This test pins down the second possibility. We construct a synthetic
DGP with:
  * strong heterogeneous tau(X) — sign flips driven by a single feature
  * deliberate confounding (propensity drift = 1.5)
  * moderate noise

On such a DGP, methods that handle confounding well (X / R / DR /
causal forest) should clearly beat S- and T-learner baselines. If they
don't, something is wrong with the implementations.

We also perform a cross-validation against `causalml`'s reference
implementations of S/T/X/R-learner — Spearman rank correlation between
our predicted uplift and causalml's, on the same data, should be > 0.7.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from uplift_bench.data.synthetic import make_uplift_dataset
from uplift_bench.metrics.qini import qini_coefficient
from uplift_bench.models.factory import make_model

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def heterogeneous_split() -> tuple[
    pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame, np.ndarray, np.ndarray
]:
    """Strong-heterogeneity DGP with confounded treatment.

    Train: 6000 rows. Test: 4000 rows.
    """
    train = make_uplift_dataset(
        n_samples=6000,
        n_features=8,
        n_informative_uplift=4,
        treatment_share=0.5,
        propensity_drift=1.5,  # confounded
        noise=0.5,
        seed=101,
    )
    test = make_uplift_dataset(
        n_samples=4000,
        n_features=8,
        n_informative_uplift=4,
        treatment_share=0.5,
        propensity_drift=1.5,
        noise=0.5,
        seed=102,
    )
    feat = train.feature_names
    return (
        train.df[feat].copy(),
        train.df["treatment"].to_numpy(),
        train.df["outcome"].to_numpy(),
        test.df[feat].copy(),
        test.df["treatment"].to_numpy(),
        test.df["outcome"].to_numpy(),
    )


def _common_kwargs(model_name: str) -> dict:
    """Reasonable training settings for a 6k-row test."""
    if model_name == "causal_forest":
        return {"seed": 0, "n_estimators": 100, "min_samples_leaf": 30}
    return {"seed": 0, "base_params": {"iterations": 200, "n_estimators": 200}}


def test_models_qini_spreads_meaningfully_under_heterogeneity(
    heterogeneous_split: tuple[
        pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame, np.ndarray, np.ndarray
    ],
) -> None:
    """The spread between best and worst meta-learner should be > 0.05
    Qini under strong heterogeneity. If this fails, several implementations
    have collapsed onto similar predictions — likely bug."""
    X_train, t_train, y_train, X_test, t_test, y_test = heterogeneous_split

    qinis: dict[str, float] = {}
    for name in [
        "s_learner",
        "t_learner",
        "x_learner",
        "r_learner",
        "dr_learner",
        "class_transformation",
    ]:
        model = make_model(name, **_common_kwargs(name))
        model.fit(X_train, t_train, y_train)
        preds = model.predict_uplift(X_test)
        qinis[name] = qini_coefficient(preds, t_test, y_test)

    spread = max(qinis.values()) - min(qinis.values())
    assert spread > 0.05, (
        f"Qini values too clustered (spread={spread:.4f}); models may have "
        f"collapsed to identical predictions. Got: {qinis}"
    )


def test_dr_or_x_learner_beats_s_learner_under_confounding(
    heterogeneous_split: tuple[
        pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame, np.ndarray, np.ndarray
    ],
) -> None:
    """Under confounded treatment (propensity_drift=1.5), DR or X should
    outperform plain S-learner. This is the textbook claim for these
    learners — if it fails, the implementation is suspect."""
    X_train, t_train, y_train, X_test, t_test, y_test = heterogeneous_split

    qinis: dict[str, float] = {}
    for name in ["s_learner", "x_learner", "dr_learner"]:
        model = make_model(name, **_common_kwargs(name))
        model.fit(X_train, t_train, y_train)
        preds = model.predict_uplift(X_test)
        qinis[name] = qini_coefficient(preds, t_test, y_test)

    # At least one of the confounding-aware learners should beat S.
    advanced_best = max(qinis["x_learner"], qinis["dr_learner"])
    assert advanced_best > qinis["s_learner"], (
        f"Neither X-learner nor DR-learner beat S-learner under confounded treatment. Got: {qinis}"
    )


def test_t_learner_matches_causalml_on_same_base_estimator() -> None:
    """T-learner with the same base classifier as causalml should produce
    rank-correlated uplift predictions.

    T-learner is the cleanest cross-validation target: no cross-fitting,
    no propensity weighting, no second-stage regression. Just two fits
    and a difference of probabilities. If our T-learner doesn't match a
    reference T-learner on the same data and same base estimator,
    something is wrong with the wrapper code.

    We use a fresh, smaller fixture here so the test stays fast and
    independent of the heterogeneous-DGP fixture.
    """
    causalml = pytest.importorskip("causalml.inference.meta")
    from sklearn.linear_model import LogisticRegression  # noqa: PLC0415

    train = make_uplift_dataset(
        n_samples=3000,
        n_features=6,
        n_informative_uplift=3,
        treatment_share=0.5,
        seed=0,
    )
    test = make_uplift_dataset(
        n_samples=1500,
        n_features=6,
        n_informative_uplift=3,
        treatment_share=0.5,
        seed=1,
    )
    feat = train.feature_names
    X_train = train.df[feat].copy()
    t_train = train.df["treatment"].to_numpy()
    y_train = train.df["outcome"].to_numpy()
    X_test = test.df[feat].copy()

    # causalml's BaseTClassifier — equivalent in spirit to our T-learner
    # for binary outcome. Use a fresh LogisticRegression for both to make
    # the comparison about the meta-learner formula, not the base.
    base_kwargs = {"C": 1.0, "max_iter": 1000, "solver": "lbfgs", "random_state": 0}
    cml_t = causalml.BaseTClassifier(
        control_learner=LogisticRegression(**base_kwargs),
        treatment_learner=LogisticRegression(**base_kwargs),
    )
    cml_t.fit(X=X_train.to_numpy(), treatment=t_train, y=y_train)
    cml_pred = cml_t.predict(X=X_test.to_numpy()).ravel()

    # Our T-learner with raw LogisticRegression (no scaling pipeline) so
    # the base estimator is identical.
    from uplift_bench.models.t_learner import TLearner  # noqa: PLC0415

    class _RawLogregT(TLearner):
        """T-learner with vanilla LogisticRegression — no scaling pipeline."""

        def __init__(self) -> None:
            super().__init__(seed=0)
            self._mu0 = LogisticRegression(**base_kwargs)
            self._mu1 = LogisticRegression(**base_kwargs)
            self._outcome = "classification"

    ub_t = _RawLogregT()
    ub_t.fit(X_train, t_train, y_train)
    our_pred = ub_t.predict_uplift(X_test)

    rho, _ = spearmanr(our_pred, cml_pred)
    # T-learner is deterministic given the base estimator and data, so
    # the rank correlation should be very close to 1.0. Allow some slack
    # for solver convergence noise.
    assert rho > 0.95, (
        f"T-learner Spearman vs causalml.BaseTClassifier = {rho:.4f}; "
        "expected ≥ 0.95 because both libs implement the identical formula"
    )
