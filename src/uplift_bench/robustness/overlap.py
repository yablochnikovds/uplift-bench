"""Propensity overlap diagnostics.

Two checks:

1. **Propensity histogram**: estimate e(X) with a calibrated classifier
   and look at the distributions for treated vs control. Heavy tails near
   0 or 1 mean we have observations with no comparable counterpart in the
   other arm — IPW-flavoured estimators (X / R / DR) will explode there.

2. **Effective sample size**: ESS = (sum w_i)^2 / sum w_i^2 with
   w_i = 1 / e(x_i) for treated, 1 / (1 - e(x_i)) for control. ESS / n
   is a single-number summary of the same problem.

We compute these with a fresh, simple classifier (sklearn HistGradientBoosting
by default) so the diagnostic doesn't depend on which meta-learner the
user picked.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_predict

from uplift_bench.metrics._common import NDArray1D


@dataclass(frozen=True, slots=True)
class OverlapDiagnostics:
    propensity: NDArray1D  # OOF estimates, length n
    treated_propensity: NDArray1D  # subset
    control_propensity: NDArray1D  # subset
    ess_ratio: float  # effective sample size / n, in [0, 1]
    pct_below_clip: float  # fraction of e(x) below 0.05
    pct_above_clip: float  # fraction of e(x) above 0.95


def overlap_diagnostics(
    X: pd.DataFrame,
    treatment: NDArray1D,
    *,
    seed: int = 0,
    n_splits: int = 5,
    clip: tuple[float, float] = (0.05, 0.95),
) -> OverlapDiagnostics:
    """Estimate propensity OOF and report overlap diagnostics."""
    treatment = np.asarray(treatment).ravel()
    if not set(np.unique(treatment)) <= {0, 1}:
        raise ValueError("treatment must be binary 0/1")

    clf = HistGradientBoostingClassifier(
        max_iter=200,
        max_depth=6,
        learning_rate=0.05,
        random_state=seed,
    )
    proba = cross_val_predict(
        clf,
        X,
        treatment,
        cv=n_splits,
        method="predict_proba",
        n_jobs=1,
    )[:, 1]

    treated = proba[treatment == 1]
    control = proba[treatment == 0]

    # IPW weights for ESS calculation.
    w = np.where(treatment == 1, 1.0 / np.clip(proba, *clip), 1.0 / np.clip(1 - proba, *clip))
    ess = (w.sum() ** 2) / (w**2).sum()
    ess_ratio = float(ess / len(w))

    pct_below = float((proba < clip[0]).mean())
    pct_above = float((proba > clip[1]).mean())

    return OverlapDiagnostics(
        propensity=proba,
        treated_propensity=treated,
        control_propensity=control,
        ess_ratio=ess_ratio,
        pct_below_clip=pct_below,
        pct_above_clip=pct_above,
    )
