"""R-learner (Nie & Wager 2021).

Fits propensity e(X) and outcome m(X) = E[Y|X] in stage 1 (with cross-fitting
to avoid overfitting bias), then minimises the R-loss in stage 2:

    L(tau) = sum_i ( (Y_i - m(X_i)) - (T_i - e(X_i)) * tau(X_i) )^2

Equivalent to a weighted regression of the outcome residual on the
treatment residual times the candidate tau. We express this as:

    target_i = (Y_i - m_hat(X_i)) / (T_i - e_hat(X_i))
    weight_i = (T_i - e_hat(X_i))^2

then fit a regressor on (X, target) with sample_weight=weight.

Implementation note: R-learner is the meta-learner where cross-fitting
matters most. We use 5-fold by default; less and stage-1 leakage shows up
as overoptimistic Qini on training folds.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from uplift_bench.metrics._common import NDArray1D
from uplift_bench.models._base_learners import BaseLearnerName, make_base_learner
from uplift_bench.models.base import UpliftModel
from uplift_bench.models.s_learner import _predict_outcome

OutcomeKind = Literal["classification", "regression"]

PROPENSITY_CLIP = (0.05, 0.95)


class RLearner(UpliftModel):
    name = "r_learner"

    def __init__(
        self,
        base_learner: BaseLearnerName = "catboost",
        outcome: OutcomeKind = "classification",
        n_splits: int = 5,
        seed: int = 42,
        base_params: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            base_learner=base_learner,
            outcome=outcome,
            n_splits=n_splits,
            seed=seed,
            base_params=base_params,
        )
        self._n_splits = n_splits
        self._seed = seed
        self._base_learner = base_learner
        self._outcome = outcome
        self._base_params = base_params
        # Stage-2 final tau model — regression of residuals on X.
        self._tau = make_base_learner(
            base_learner, task="regression", seed=seed + 100, params=base_params
        )

    def fit(self, X: pd.DataFrame, t: NDArray1D, y: NDArray1D) -> RLearner:
        t, y = self._as_arrays(t, y)
        m_hat, e_hat = self._cross_fit_nuisances(X, t, y)

        residual_y = y - m_hat
        residual_t = t.astype(np.float64) - e_hat

        # With propensity clipped to [0.05, 0.95] we have |residual_t| ≥ 0.05
        # for every row, so the division is safe. Algebraically equivalent to
        # the canonical weighted-residual form (Nie & Wager 2021 Algorithm 1):
        #   minimize sum_i (residual_y_i - residual_t_i * tau(X_i))^2
        # which we encode as label = residual_y / residual_t and weight =
        # residual_t^2 — the algebra cancels out exactly.
        target = residual_y / residual_t
        weight = residual_t**2

        self._tau.fit(X, target, sample_weight=weight)
        self._fitted = True
        return self

    def predict_uplift(self, X: pd.DataFrame) -> NDArray1D:
        self._check_fitted()
        return np.asarray(self._tau.predict(X)).ravel()

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #

    def _cross_fit_nuisances(
        self,
        X: pd.DataFrame,
        t: NDArray1D,
        y: NDArray1D,
    ) -> tuple[NDArray1D, NDArray1D]:
        """K-fold OOF predictions for m(X)=E[Y|X] and e(X)=P(T=1|X).

        Uses StratifiedKFold on the *treatment* indicator so every fold
        has both arms, avoiding one-class folds that crash propensity fits.
        """
        kf = StratifiedKFold(
            n_splits=self._n_splits,
            shuffle=True,
            random_state=self._seed,
        )
        m_hat = np.zeros(len(X), dtype=np.float64)
        e_hat = np.zeros(len(X), dtype=np.float64)

        for fold, (train_idx, hold_idx) in enumerate(kf.split(X, t)):
            outcome_model = make_base_learner(
                self._base_learner,
                task=self._outcome,
                seed=self._seed + 200 + fold,
                params=self._base_params,
            )
            propensity_model = make_base_learner(
                self._base_learner,
                task="classification",
                seed=self._seed + 300 + fold,
                params=self._base_params,
            )
            outcome_model.fit(X.iloc[train_idx], y[train_idx])
            propensity_model.fit(X.iloc[train_idx], t[train_idx])
            m_hat[hold_idx] = _predict_outcome(outcome_model, X.iloc[hold_idx], self._outcome)
            e_hat[hold_idx] = propensity_model.predict_proba(X.iloc[hold_idx])[:, 1]

        e_hat = np.clip(e_hat, *PROPENSITY_CLIP)
        return m_hat, e_hat
