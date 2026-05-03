"""Permutation feature importance for *uplift*.

Standard permutation importance (Breiman 2001) shuffles a feature column
and measures how much the loss degrades. For uplift the obvious knob to
turn is the Qini coefficient: a feature that genuinely drives heterogeneous
treatment effect should hurt Qini when shuffled. A feature that only drives
the *outcome* (not uplift) shouldn't.

That distinction is why we don't reuse sklearn's permutation_importance:
sklearn shuffles relative to a model's score on Y, which conflates the two.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from uplift_bench.metrics._common import NDArray1D
from uplift_bench.metrics.qini import qini_coefficient
from uplift_bench.models.base import UpliftModel


@dataclass(frozen=True, slots=True)
class PermutationImportance:
    feature: str
    baseline_qini: float
    mean_qini_drop: float
    std_qini_drop: float
    n_repeats: int


def permutation_uplift_importance(
    model: UpliftModel,
    X: pd.DataFrame,
    treatment: NDArray1D,
    outcome: NDArray1D,
    *,
    n_repeats: int = 5,
    seed: int = 0,
) -> pd.DataFrame:
    """Per-feature permutation importance computed against Qini.

    Parameters
    ----------
    model
        A *fitted* UpliftModel.
    X, treatment, outcome
        Held-out evaluation set (don't reuse the training fold).
    n_repeats
        How many shuffles per feature. 5 is enough for ranking; bump to 30+
        for tight error bars.
    seed
        RNG seed. Each (feature, repeat) pair derives a child seed via
        SeedSequence so the shuffles are independent but reproducible.

    Returns
    -------
    pd.DataFrame
        Columns: feature, baseline_qini, mean_qini_drop, std_qini_drop, n_repeats.
        Sorted descending by mean_qini_drop. Positive drop = important feature.
    """
    if n_repeats < 1:
        raise ValueError(f"n_repeats must be >= 1, got {n_repeats}")

    base_preds = model.predict_uplift(X)
    baseline = qini_coefficient(base_preds, treatment, outcome)
    seed_seq = np.random.SeedSequence(seed)

    rows: list[dict[str, float | str | int]] = []
    feature_seeds = seed_seq.generate_state(len(X.columns) * n_repeats).reshape(
        len(X.columns),
        n_repeats,
    )

    for j, feature in enumerate(X.columns):
        drops = np.empty(n_repeats, dtype=np.float64)
        for r in range(n_repeats):
            rng = np.random.default_rng(int(feature_seeds[j, r]))
            X_shuf = X.copy()
            X_shuf[feature] = rng.permutation(X_shuf[feature].to_numpy())
            preds = model.predict_uplift(X_shuf)
            drops[r] = baseline - qini_coefficient(preds, treatment, outcome)

        rows.append(
            {
                "feature": str(feature),
                "baseline_qini": float(baseline),
                "mean_qini_drop": float(drops.mean()),
                "std_qini_drop": float(drops.std(ddof=1)) if n_repeats > 1 else 0.0,
                "n_repeats": n_repeats,
            }
        )

    return pd.DataFrame(rows).sort_values("mean_qini_drop", ascending=False).reset_index(drop=True)
