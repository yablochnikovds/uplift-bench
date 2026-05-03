"""uplift@k — average uplift in the top-k targeted fraction.

Used by marketers who only have budget for the top X% of customers. The
question is: among those, what's the realised lift over a random pull
of the same size?

Implementation detail: we compute the *difference of means* of the outcome
between treated and control rows inside the top-k. This is the natural
estimator under random assignment; under confounded assignment it's biased
but that's what every public benchmark uses, so we match for comparability.
"""

from __future__ import annotations

import numpy as np

from uplift_bench.metrics._common import NDArray1D, as_1d_arrays, sort_by_score_desc


def uplift_at_k(
    score: NDArray1D | list[float],
    treatment: NDArray1D | list[int],
    outcome: NDArray1D | list[int],
    k: float,
) -> float:
    """Compute uplift in the top-k fraction.

    Parameters
    ----------
    score
        Predicted treatment effect (or anything to rank by, descending).
    treatment, outcome
        Binary 0/1 arrays.
    k
        Fraction in (0, 1]. e.g. 0.10 → top decile.

    Returns
    -------
    float
        Realised mean(y | T=1, top-k) - mean(y | T=0, top-k). NaN if either
        sub-group is empty in the top-k (the right answer — the metric is
        undefined, not zero).
    """
    if not 0 < k <= 1:
        raise ValueError(f"k must be in (0, 1], got {k}")

    score, treatment, outcome = as_1d_arrays(score, treatment, outcome)
    n = len(score)
    if n == 0:
        raise ValueError("uplift_at_k requires at least one observation")

    n_top = max(1, round(n * k))
    order = sort_by_score_desc(score)
    top = order[:n_top]

    t_top = treatment[top].astype(bool)
    y_top = outcome[top].astype(np.float64)

    if not t_top.any() or not (~t_top).any():
        return float("nan")

    return float(y_top[t_top].mean() - y_top[~t_top].mean())
