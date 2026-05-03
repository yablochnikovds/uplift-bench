"""T-learner.

Two separate models — one trained on treated rows, one on control rows.
Uplift = mu_1(X) - mu_0(X). The opposite trade-off to S-learner: never
shares signal across arms, so it overfits when one arm is small but
captures arm-specific structure perfectly when both arms are large.
"""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd

from uplift_bench.metrics._common import NDArray1D
from uplift_bench.models._base_learners import BaseLearnerName, make_base_learner
from uplift_bench.models.base import UpliftModel
from uplift_bench.models.s_learner import _predict_outcome  # shared helper

OutcomeKind = Literal["classification", "regression"]


class TLearner(UpliftModel):
    name = "t_learner"

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
        # +1 to the seed for the treated model so the two estimators don't
        # see literally the same bagging permutations. Tiny effect in
        # practice but it keeps "perfectly seeded" honest.
        self._mu0 = make_base_learner(base_learner, task=outcome, seed=seed, params=base_params)
        self._mu1 = make_base_learner(base_learner, task=outcome, seed=seed + 1, params=base_params)
        self._outcome = outcome

    def fit(self, X: pd.DataFrame, t: NDArray1D, y: NDArray1D) -> TLearner:
        t, y = self._as_arrays(t, y)
        mask_t = t == 1
        if mask_t.sum() == 0 or (~mask_t).sum() == 0:
            raise ValueError("T-learner needs at least one row in each arm")
        self._mu0.fit(X.loc[~mask_t], y[~mask_t])
        self._mu1.fit(X.loc[mask_t], y[mask_t])
        self._fitted = True
        return self

    def predict_uplift(self, X: pd.DataFrame) -> NDArray1D:
        self._check_fitted()
        return _predict_outcome(self._mu1, X, self._outcome) - _predict_outcome(
            self._mu0, X, self._outcome
        )
