"""Causal forest wrapper around econml's CausalForestDML.

We wrap rather than re-export because (a) econml's API is opinionated
(treatment must be float, predictions return shape (n, 1, 1)…), and
(b) we want a uniform `predict_uplift(X) -> 1-D ndarray` contract across
all seven meta-learners.

Reference: Athey & Wager (2019), "Estimation and Inference of
Heterogeneous Treatment Effects Using Random Forests" — JASA.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from econml.dml import CausalForestDML
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from uplift_bench.metrics._common import NDArray1D
from uplift_bench.models.base import UpliftModel

# These defaults follow Athey & Wager's recommended ranges. The big knob is
# `n_estimators` — bumping it from 100 to 500 typically buys 0.005-0.01 Qini
# at the cost of 5x training time.
_DEFAULTS: dict[str, Any] = {
    "n_estimators": 200,
    "min_samples_leaf": 30,
    "max_depth": None,
    "max_samples": 0.45,  # subsample per tree (Wager & Athey)
    "honest": True,
    "discrete_treatment": True,
    "cv": 3,
}


class CausalForestModel(UpliftModel):
    name = "causal_forest"

    def __init__(
        self,
        seed: int = 42,
        n_estimators: int = _DEFAULTS["n_estimators"],
        min_samples_leaf: int = _DEFAULTS["min_samples_leaf"],
        max_depth: int | None = _DEFAULTS["max_depth"],
        max_samples: float = _DEFAULTS["max_samples"],
    ) -> None:
        super().__init__(
            seed=seed,
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            max_depth=max_depth,
            max_samples=max_samples,
        )
        # CausalForestDML wants explicit nuisance models. We use sklearn
        # random forests because they're simpler to seed than CatBoost
        # inside econml's CV machinery and give comparable results on tabular.
        outcome_model = RandomForestRegressor(
            n_estimators=200,
            min_samples_leaf=20,
            n_jobs=-1,
            random_state=seed,
        )
        propensity_model = RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=20,
            n_jobs=-1,
            random_state=seed,
        )

        self._cf = CausalForestDML(
            model_y=outcome_model,
            model_t=propensity_model,
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            max_depth=max_depth,
            max_samples=max_samples,
            honest=_DEFAULTS["honest"],
            discrete_treatment=_DEFAULTS["discrete_treatment"],
            cv=_DEFAULTS["cv"],
            random_state=seed,
        )

    def fit(self, X: pd.DataFrame, t: NDArray1D, y: NDArray1D) -> CausalForestModel:
        t, y = self._as_arrays(t, y)
        # econml expects (Y, T, X=X, W=None) and float treatments.
        self._cf.fit(Y=y.astype(np.float64), T=t.astype(np.float64), X=X)
        self._fitted = True
        return self

    def predict_uplift(self, X: pd.DataFrame) -> NDArray1D:
        self._check_fitted()
        # econml returns shape (n, 1) for a binary treatment; flatten.
        return np.asarray(self._cf.effect(X)).ravel()
