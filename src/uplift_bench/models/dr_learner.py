"""DR-learner — Doubly Robust meta-learner (Kennedy 2023).

Same skeleton as R-learner (cross-fit nuisances) but the stage-2 target is
the doubly-robust pseudo-outcome:

    psi_i = mu_1(X_i) - mu_0(X_i)
          + (T_i / e(X_i))     * (Y_i - mu_1(X_i))
          - ((1-T_i)/(1-e(X_i))) * (Y_i - mu_0(X_i))

This is unbiased for tau(X) when *either* the outcome model *or* the
propensity model is correctly specified — hence "doubly robust". On the
benchmark this is the meta-learner I expect to win on Criteo with a
flexible base learner like CatBoost; the propensity scores from CatBoost
are usually well-calibrated enough that even when the outcome model is
slightly off, the IPW correction recovers the bias.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from uplift_bench.metrics._common import NDArray1D
from uplift_bench.models._base_learners import BaseLearnerName, make_base_learner
from uplift_bench.models.base import UpliftModel
from uplift_bench.models.s_learner import _predict_outcome

OutcomeKind = Literal["classification", "regression"]

PROPENSITY_CLIP = (0.05, 0.95)


class DRLearner(UpliftModel):
    name = "dr_learner"

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
        # Stage-2 final tau model — regression of pseudo-outcome on X.
        self._tau = make_base_learner(
            base_learner, task="regression", seed=seed + 100, params=base_params
        )

    def fit(self, X: pd.DataFrame, t: NDArray1D, y: NDArray1D) -> DRLearner:
        t, y = self._as_arrays(t, y)
        mu0_hat, mu1_hat, e_hat = self._cross_fit_nuisances(X, t, y)

        psi = (
            (mu1_hat - mu0_hat)
            + (t / e_hat) * (y - mu1_hat)
            - ((1 - t) / (1 - e_hat)) * (y - mu0_hat)
        )
        self._tau.fit(X, psi)
        self._fitted = True
        return self

    def predict_uplift(self, X: pd.DataFrame) -> NDArray1D:
        self._check_fitted()
        return np.asarray(self._tau.predict(X)).ravel()

    def _cross_fit_nuisances(
        self,
        X: pd.DataFrame,
        t: NDArray1D,
        y: NDArray1D,
    ) -> tuple[NDArray1D, NDArray1D, NDArray1D]:
        kf = KFold(n_splits=self._n_splits, shuffle=True, random_state=self._seed)
        n = len(X)
        mu0 = np.zeros(n, dtype=np.float64)
        mu1 = np.zeros(n, dtype=np.float64)
        e_hat = np.zeros(n, dtype=np.float64)

        for fold, (train_idx, hold_idx) in enumerate(kf.split(X)):
            t_tr = t[train_idx]
            mask_t = t_tr == 1
            X_tr = X.iloc[train_idx]
            y_tr = y[train_idx]

            mu0_model = make_base_learner(
                self._base_learner,
                task=self._outcome,
                seed=self._seed + 200 + fold,
                params=self._base_params,
            )
            mu1_model = make_base_learner(
                self._base_learner,
                task=self._outcome,
                seed=self._seed + 300 + fold,
                params=self._base_params,
            )
            propensity_model = make_base_learner(
                self._base_learner,
                task="classification",
                seed=self._seed + 400 + fold,
                params=self._base_params,
            )

            # If a fold accidentally has only one arm we'd silently produce
            # garbage — fail loud instead.
            if mask_t.sum() == 0 or (~mask_t).sum() == 0:
                raise RuntimeError(
                    f"DR-learner fold {fold} has only one treatment arm; "
                    "increase n_splits or use a stratified splitter."
                )

            mu0_model.fit(X_tr.loc[~mask_t], y_tr[~mask_t])
            mu1_model.fit(X_tr.loc[mask_t], y_tr[mask_t])
            propensity_model.fit(X_tr, t_tr)

            mu0[hold_idx] = _predict_outcome(mu0_model, X.iloc[hold_idx], self._outcome)
            mu1[hold_idx] = _predict_outcome(mu1_model, X.iloc[hold_idx], self._outcome)
            e_hat[hold_idx] = propensity_model.predict_proba(X.iloc[hold_idx])[:, 1]

        e_hat = np.clip(e_hat, *PROPENSITY_CLIP)
        return mu0, mu1, e_hat
