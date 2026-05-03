"""Drop-feature stability of Qini.

For each feature (or group of features), refit the model with that column
removed and measure the change in Qini on a held-out set.

Different from permutation importance:
* Permutation: how much does the *fitted* model rely on this feature?
* Drop-feature: how much does removing the feature change the model
  *that gets fit*? Captures interaction effects permutation misses.

Slow — costs N_features model fits — so we expose `n_jobs` for joblib
parallelism and recommend running it only on the final candidate models,
not during sweep.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd
from joblib import Parallel, delayed

from uplift_bench.metrics._common import NDArray1D
from uplift_bench.metrics.qini import qini_coefficient
from uplift_bench.models.base import UpliftModel

ModelBuilder = Callable[[], UpliftModel]


@dataclass(frozen=True, slots=True)
class FeatureDropResult:
    feature_or_group: str
    baseline_qini: float
    qini_after_drop: float
    qini_delta: float  # baseline - after; positive = feature mattered


def _fit_and_score(
    builder: ModelBuilder,
    X_train_dropped: pd.DataFrame,
    t_train: NDArray1D,
    y_train: NDArray1D,
    X_eval_dropped: pd.DataFrame,
    t_eval: NDArray1D,
    y_eval: NDArray1D,
) -> float:
    model = builder()
    model.fit(X_train_dropped, t_train, y_train)
    preds = model.predict_uplift(X_eval_dropped)
    return float(qini_coefficient(preds, t_eval, y_eval))


def feature_drop_stability(
    model_builder: ModelBuilder,
    X_train: pd.DataFrame,
    t_train: NDArray1D,
    y_train: NDArray1D,
    X_eval: pd.DataFrame,
    t_eval: NDArray1D,
    y_eval: NDArray1D,
    *,
    groups: dict[str, list[str]] | None = None,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Refit the model dropping each feature (or group) and report Qini delta.

    Parameters
    ----------
    model_builder
        Zero-arg factory returning a fresh, *unfit* UpliftModel. We need a
        factory rather than a model instance because we have to re-fit
        from scratch for each drop.
    X_train, t_train, y_train, X_eval, t_eval, y_eval
        Training and held-out folds.
    groups
        Optional mapping of group_name → list of column names to drop together.
        If None, drops one feature at a time.
    n_jobs
        joblib parallelism. Each fit is independent.

    Returns
    -------
    pd.DataFrame
        Sorted descending by qini_delta (most-important groups first).
    """
    # Baseline: train on the full feature set.
    baseline = _fit_and_score(
        model_builder,
        X_train,
        t_train,
        y_train,
        X_eval,
        t_eval,
        y_eval,
    )

    if groups is None:
        groups = {col: [col] for col in X_train.columns}

    drop_specs: list[tuple[str, list[str]]] = list(groups.items())

    qinis = Parallel(n_jobs=n_jobs)(
        delayed(_fit_and_score)(
            model_builder,
            X_train.drop(columns=cols),
            t_train,
            y_train,
            X_eval.drop(columns=cols),
            t_eval,
            y_eval,
        )
        for _, cols in drop_specs
    )

    rows = [
        {
            "feature_or_group": name,
            "baseline_qini": baseline,
            "qini_after_drop": q,
            "qini_delta": baseline - q,
        }
        for (name, _), q in zip(drop_specs, qinis, strict=True)
    ]
    return pd.DataFrame(rows).sort_values("qini_delta", ascending=False).reset_index(drop=True)
