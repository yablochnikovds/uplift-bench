"""Class Variable Transformation (Jaskowski & Jaroszewicz 2012).

Trick: under randomised treatment assignment with P(T=1) = 0.5, define

    Z = T * Y + (1 - T) * (1 - Y)

i.e. Z = 1 iff (treated and responded) OR (not treated and not responded).
Then 2 * P(Z=1|X) - 1 = E[Y|T=1, X] - E[Y|T=0, X] = tau(X).

So a single classifier on (X, Z) recovers an unbiased uplift estimator —
without ever fitting two outcome models. Catches:

* Requires balanced treatment, ideally close to 50/50. We re-weight rows
  by `1 / P(T=t_i)` to make this work for arbitrary marginal propensities,
  which is the modern adjustment used by scikit-uplift.
* Only defined for binary outcomes. Calling with continuous outcome raises.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from uplift_bench.metrics._common import NDArray1D
from uplift_bench.models._base_learners import BaseLearnerName, make_base_learner
from uplift_bench.models.base import UpliftModel


class ClassTransformationLearner(UpliftModel):
    name = "class_transformation"

    def __init__(
        self,
        base_learner: BaseLearnerName = "catboost",
        seed: int = 42,
        base_params: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(base_learner=base_learner, seed=seed, base_params=base_params)
        self._model = make_base_learner(
            base_learner, task="classification", seed=seed, params=base_params
        )

    def fit(self, X: pd.DataFrame, t: NDArray1D, y: NDArray1D) -> ClassTransformationLearner:
        t, y = self._as_arrays(t, y)
        if not set(np.unique(y)) <= {0, 1}:
            raise ValueError("class transformation requires binary outcome 0/1")

        z = (t * y + (1 - t) * (1 - y)).astype(int)
        # Re-weight to handle non-balanced propensity. For balanced data the
        # weights collapse to 1.
        p_t = float(t.mean())
        if not 0.05 < p_t < 0.95:
            raise ValueError(
                f"class transformation expects roughly balanced treatment; "
                f"observed P(T=1)={p_t:.3f} is outside [0.05, 0.95]"
            )
        weights = np.where(t == 1, 1.0 / p_t, 1.0 / (1.0 - p_t))
        self._model.fit(X, z, sample_weight=weights)
        self._fitted = True
        return self

    def predict_uplift(self, X: pd.DataFrame) -> NDArray1D:
        self._check_fitted()
        proba = np.asarray(self._model.predict_proba(X))[:, 1]
        return 2.0 * proba - 1.0
