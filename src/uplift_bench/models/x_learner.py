"""X-learner (Künzel, Sekhon, Bickel, Yu 2019).

Two-stage:

  Stage 1 — fit mu_0 on control, mu_1 on treated (same as T-learner).
  Stage 2 — impute counterfactual differences:
      D_treated  = Y_1 - mu_0(X_1)
      D_control  = mu_1(X_0) - Y_0
    then fit tau_0 on D_control (X_0) and tau_1 on D_treated (X_1).
  Combine via propensity weights:
      tau(X) = e(X) * tau_0(X) + (1 - e(X)) * tau_1(X)

The propensity weights `e(X)` come from a separate model. We clip them to
[0.05, 0.95] — if you don't, X-learner explodes on small datasets where
propensity gets close to 0 or 1. Painful lesson learned on RetailHero.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

from uplift_bench.metrics._common import NDArray1D
from uplift_bench.models._base_learners import BaseLearnerName, make_base_learner
from uplift_bench.models.base import UpliftModel
from uplift_bench.models.s_learner import _predict_outcome

OutcomeKind = Literal["classification", "regression"]

PROPENSITY_CLIP = (0.05, 0.95)


class XLearner(UpliftModel):
    name = "x_learner"

    def __init__(
        self,
        base_learner: BaseLearnerName = "catboost",
        outcome: OutcomeKind = "classification",
        seed: int = 42,
        base_params: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            base_learner=base_learner, outcome=outcome, seed=seed, base_params=base_params
        )
        self._mu0 = make_base_learner(base_learner, task=outcome, seed=seed, params=base_params)
        self._mu1 = make_base_learner(base_learner, task=outcome, seed=seed + 1, params=base_params)
        self._tau0 = make_base_learner(
            base_learner, task="regression", seed=seed + 2, params=base_params
        )
        self._tau1 = make_base_learner(
            base_learner, task="regression", seed=seed + 3, params=base_params
        )
        # Propensity is always classification regardless of outcome kind.
        self._propensity = make_base_learner(
            base_learner, task="classification", seed=seed + 4, params=base_params
        )
        self._outcome = outcome

    def fit(self, X: pd.DataFrame, t: NDArray1D, y: NDArray1D) -> XLearner:
        t, y = self._as_arrays(t, y)
        mask_t = t == 1
        if mask_t.sum() == 0 or (~mask_t).sum() == 0:
            raise ValueError("X-learner needs at least one row in each arm")

        # Stage 1: fit outcome models per arm.
        self._mu0.fit(X.loc[~mask_t], y[~mask_t])
        self._mu1.fit(X.loc[mask_t], y[mask_t])

        # Stage 2: imputed individual treatment effects.
        d_treated = y[mask_t] - _predict_outcome(self._mu0, X.loc[mask_t], self._outcome)
        d_control = _predict_outcome(self._mu1, X.loc[~mask_t], self._outcome) - y[~mask_t]

        self._tau1.fit(X.loc[mask_t], d_treated)
        self._tau0.fit(X.loc[~mask_t], d_control)

        # Propensity model. Always trained on full X; predicts P(T=1|X).
        self._propensity.fit(X, t)
        self._fitted = True
        return self

    def predict_uplift(self, X: pd.DataFrame) -> NDArray1D:
        self._check_fitted()
        e: NDArray1D = np.clip(
            np.asarray(self._propensity.predict_proba(X))[:, 1],
            *PROPENSITY_CLIP,
        )
        tau1 = np.asarray(self._tau1.predict(X)).ravel()
        tau0 = np.asarray(self._tau0.predict(X)).ravel()
        # Künzel et al. weighting: use propensity to weight the two stage-2
        # models inversely to where their training data lived.
        return np.asarray(e * tau0 + (1 - e) * tau1)
