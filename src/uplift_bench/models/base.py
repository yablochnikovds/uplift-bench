"""Abstract base for all uplift meta-learners.

The contract is intentionally tiny — `fit` and `predict_uplift`. Anything
fancier (CV-based hyper-tuning, per-group calibration) is the job of a
wrapper, not the base class.

Naming: I use `X`, `t`, `y` as parameters because these are the universal
short names in the causal-inference literature. The lint rule that bans
single-letter names is per-file-disabled in pyproject for the same reason.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd

from uplift_bench.metrics._common import NDArray1D


class UpliftModel(ABC):
    """Common interface every meta-learner implements.

    Subclasses store fitted state on `self`; we don't enforce a particular
    layout because each learner needs different things (S-learner: one
    model; T-learner: two; X-learner: 3+).
    """

    name: str  # short slug used in MLflow run tags

    def __init__(self, **kwargs: Any) -> None:
        # Accepted hyper-parameters live in `self.params` so they're easy
        # to log without subclasses needing to enumerate them.
        self.params: dict[str, Any] = dict(kwargs)
        self._fitted: bool = False

    @abstractmethod
    def fit(self, X: pd.DataFrame, t: NDArray1D, y: NDArray1D) -> UpliftModel:
        """Train the model. Must mark `self._fitted = True` before returning."""

    @abstractmethod
    def predict_uplift(self, X: pd.DataFrame) -> NDArray1D:
        """Return per-row uplift estimate (same length as X)."""

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(f"{type(self).__name__} must be fit() before predict_uplift()")

    @staticmethod
    def _as_arrays(t: NDArray1D, y: NDArray1D) -> tuple[NDArray1D, NDArray1D]:
        """Coerce treatment and outcome to numpy arrays of expected shapes."""
        t_arr = np.asarray(t).ravel()
        y_arr = np.asarray(y).ravel()
        if len(t_arr) != len(y_arr):
            raise ValueError(f"t and y must align: {len(t_arr)} vs {len(y_arr)}")
        if not set(np.unique(t_arr)) <= {0, 1}:
            raise ValueError("treatment must be binary 0/1")
        return t_arr, y_arr
