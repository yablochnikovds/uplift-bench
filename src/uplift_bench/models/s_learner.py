"""S-learner.

Single model trained on (X, T) → Y. Uplift estimate is the difference of
predictions when T is set to 1 vs 0:

    tau_hat(X) = mu(X, T=1) - mu(X, T=0)

The simplest meta-learner. Tends to win on data where the treatment effect
is small and the model is highly regularised — because in that regime the
shared parameter pool acts as a useful prior. Loses badly when the treated
and control groups have very different X distributions.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

from uplift_bench.metrics._common import NDArray1D
from uplift_bench.models._base_learners import BaseLearnerName, make_base_learner
from uplift_bench.models.base import UpliftModel

OutcomeKind = Literal["classification", "regression"]


class SLearner(UpliftModel):
    name = "s_learner"

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
        self._model = make_base_learner(base_learner, task=outcome, seed=seed, params=base_params)
        self._outcome = outcome
        self._treatment_col = "_t_internal"

    def fit(self, X: pd.DataFrame, t: NDArray1D, y: NDArray1D) -> SLearner:
        t, y = self._as_arrays(t, y)
        X_aug = X.copy()
        X_aug[self._treatment_col] = t
        self._model.fit(X_aug, y)
        self._fitted = True
        return self

    def predict_uplift(self, X: pd.DataFrame) -> NDArray1D:
        self._check_fitted()
        X1 = X.copy()
        X0 = X.copy()
        X1[self._treatment_col] = 1
        X0[self._treatment_col] = 0
        return _predict_outcome(self._model, X1, self._outcome) - _predict_outcome(
            self._model, X0, self._outcome
        )


def _predict_outcome(model: Any, X: pd.DataFrame, outcome: OutcomeKind) -> NDArray1D:
    """Return P(Y=1|X,T) for classification, raw mean for regression."""
    if outcome == "classification":
        proba = model.predict_proba(X)
        # Sklearn pipelines and tree models both return shape (n, 2)
        return np.asarray(proba)[:, 1]
    return np.asarray(model.predict(X)).ravel()
