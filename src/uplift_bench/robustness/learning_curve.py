"""Learning curve: Qini vs training-set size.

Refit the same model on increasing fractions of the training set and
report Qini on a fixed eval set. Useful for two questions:

1. "Will more data help?" — flat curve at the high end means we've
   saturated; rising curve means we're data-limited.
2. "Is the model overfitting at small N?" — early Qini > late Qini is the
   pathological signature of meta-learners that haven't been regularised
   for sample size (often X-learner with deep CatBoost).

We always sort the train set by a fixed permutation (seeded) and take
prefixes — so smaller training sets are subsets of bigger ones, which
makes the curve interpretable.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

from uplift_bench.metrics._common import NDArray1D
from uplift_bench.metrics.qini import qini_coefficient
from uplift_bench.models.base import UpliftModel

ModelBuilder = Callable[[], UpliftModel]


def learning_curve(
    model_builder: ModelBuilder,
    X_train: pd.DataFrame,
    t_train: NDArray1D,
    y_train: NDArray1D,
    X_eval: pd.DataFrame,
    t_eval: NDArray1D,
    y_eval: NDArray1D,
    *,
    fractions: Sequence[float] = (0.1, 0.25, 0.5, 0.75, 1.0),
    seed: int = 0,
) -> pd.DataFrame:
    """Compute Qini at each training-set fraction.

    Returns a DataFrame with columns: fraction, n_train, qini.
    """
    if any(not 0 < f <= 1 for f in fractions):
        raise ValueError(f"fractions must be in (0, 1], got {fractions}")

    n = len(X_train)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    X_train = X_train.iloc[perm].reset_index(drop=True)
    t_train = np.asarray(t_train)[perm]
    y_train = np.asarray(y_train)[perm]

    rows: list[dict[str, float | int]] = []
    for frac in fractions:
        n_use = max(2, round(n * frac))
        model = model_builder()
        model.fit(X_train.iloc[:n_use], t_train[:n_use], y_train[:n_use])
        preds = model.predict_uplift(X_eval)
        rows.append(
            {
                "fraction": float(frac),
                "n_train": int(n_use),
                "qini": float(qini_coefficient(preds, t_eval, y_eval)),
            }
        )

    return pd.DataFrame(rows)
