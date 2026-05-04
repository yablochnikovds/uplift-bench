"""Factory that turns a config name into a fresh sklearn-compatible estimator.

Why this lives here, not next to the meta-learners: the same base learners
(catboost, lightgbm, logreg) are reused across S/T/X/R/DR/CT/CF, so it
deserves a single source of truth. Each meta-learner just calls
`make_base_learner('catboost', task='regression')`.
"""

from __future__ import annotations

from typing import Any, Literal

from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BaseLearnerName = Literal["catboost", "lightgbm", "logreg"]
TaskKind = Literal["classification", "regression"]

# We don't try to express the full sklearn estimator interface in a
# Protocol — fit / predict / predict_proba have wildly different signatures
# across libraries. `Any` here is the honest answer; mypy strict still
# checks every call site, just not the estimator object.
SklearnEstimator = Any


# Reasonable defaults that don't OOM on Criteo and don't underfit on
# Hillstrom. These can be overridden via the `params` kwarg.
_CATBOOST_DEFAULTS: dict[str, Any] = {
    "iterations": 500,
    "depth": 6,
    "learning_rate": 0.05,
    "l2_leaf_reg": 3.0,
    "verbose": False,
    "allow_writing_files": False,
    "thread_count": -1,
}

_LGBM_DEFAULTS: dict[str, Any] = {
    "n_estimators": 500,
    "num_leaves": 63,
    "learning_rate": 0.05,
    "min_child_samples": 50,
    "verbose": -1,
    "n_jobs": -1,
}

_LOGREG_DEFAULTS: dict[str, Any] = {
    "max_iter": 1000,
    "n_jobs": -1,
    "solver": "lbfgs",
}


def make_base_learner(
    name: BaseLearnerName,
    *,
    task: TaskKind,
    seed: int = 42,
    params: dict[str, Any] | None = None,
) -> SklearnEstimator:
    """Create a fresh estimator instance.

    Always returns a *new* object — meta-learners that need multiple base
    models must call this multiple times rather than cloning, since
    cloning a CatBoost model loses the random_state we just set.
    """
    overrides = dict(params or {})

    if name == "catboost":
        # CatBoost rejects passing both "iterations" and "n_estimators" even
        # though they're aliases. We accept either, but only forward the
        # canonical one ("iterations") downstream.
        if "n_estimators" in overrides and "iterations" not in overrides:
            overrides["iterations"] = overrides.pop("n_estimators")
        elif "n_estimators" in overrides and "iterations" in overrides:
            overrides.pop("n_estimators")
        merged = {**_CATBOOST_DEFAULTS, "random_seed": seed, **overrides}
        if task == "classification":
            return CatBoostClassifier(**merged)
        return CatBoostRegressor(**merged)

    if name == "lightgbm":
        # Mirror the alias trick — accept "iterations" and forward as "n_estimators".
        if "iterations" in overrides and "n_estimators" not in overrides:
            overrides["n_estimators"] = overrides.pop("iterations")
        elif "iterations" in overrides and "n_estimators" in overrides:
            overrides.pop("iterations")
        merged = {**_LGBM_DEFAULTS, "random_state": seed, **overrides}
        if task == "classification":
            return LGBMClassifier(**merged)
        return LGBMRegressor(**merged)

    if name == "logreg":
        # Tree-model knobs leak in through shared configs; drop them so
        # logreg/Ridge don't TypeError on unknown kwargs.
        for k in (
            "iterations",
            "n_estimators",
            "depth",
            "num_leaves",
            "min_child_samples",
            "l2_leaf_reg",
            "learning_rate",
            "verbose",
            "allow_writing_files",
            "thread_count",
        ):
            overrides.pop(k, None)
        merged = {**_LOGREG_DEFAULTS, "random_state": seed, **overrides}
        # Logistic without scaling can be misleading on Criteo features that
        # span ten orders of magnitude. Pipeline keeps it honest.
        if task == "classification":
            return Pipeline(
                [
                    ("scale", StandardScaler(with_mean=True)),
                    ("clf", LogisticRegression(**merged)),
                ]
            )
        # For regression we use Ridge (logreg's regression analogue) and
        # the same scaling pipeline.
        ridge_params = {k: v for k, v in merged.items() if k in {"random_state", "alpha"}}
        return Pipeline(
            [
                ("scale", StandardScaler(with_mean=True)),
                ("reg", Ridge(**ridge_params)),
            ]
        )

    raise ValueError(f"unknown base learner {name!r}")
