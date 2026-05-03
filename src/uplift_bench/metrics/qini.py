"""Qini coefficient and Qini curve.

Definition (Radcliffe 2007). Sort observations by predicted uplift descending.
Walking down that ordered list, plot:

    x-axis : cumulative population share k / N
    y-axis : (n_treated_responders[:k] - n_control_responders[:k]
             * n_treated[:k] / n_control[:k])  / N

The Qini coefficient is the area between this curve and the random-targeting
diagonal, normalised so that a perfect-ordering model has Q == 1.

Two implementation notes that bit me before:

1. The ratio `n_treated[:k] / n_control[:k]` is undefined at k=0 and at the
   edge case where a prefix has zero control rows. We clamp using `np.where`
   instead of try/except to keep the function vectorised.

2. Ties in the score must be broken consistently across runs — otherwise
   `qini(score, t, y, seed=0)` and the same call later return slightly
   different numbers. We resolve ties by pre-sorting on (score desc, row index).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from uplift_bench.metrics._common import NDArray1D, as_1d_arrays, sort_by_score_desc


@dataclass(frozen=True, slots=True)
class QiniCurve:
    """The cumulative uplift curve.

    `population_share` is in [0, 1]. `cumulative_uplift` is in absolute
    counts divided by N (so the value at share=1 equals the realised ATE
    on the sample, when treated and control sizes are the same).
    """

    population_share: NDArray1D
    cumulative_uplift: NDArray1D
    qini_coefficient: float
    random_baseline_area: float


def qini_curve(
    score: NDArray1D | list[float],
    treatment: NDArray1D | list[int],
    outcome: NDArray1D | list[int],
) -> QiniCurve:
    """Compute the Qini curve and the Qini coefficient.

    Parameters
    ----------
    score
        Predicted individual treatment effect (or any scalar to rank by).
    treatment
        Binary 0/1 array of treatment indicators.
    outcome
        Binary 0/1 array of observed outcomes.

    Returns
    -------
    QiniCurve
        Curve plus scalar coefficient.
    """
    score, treatment, outcome = as_1d_arrays(score, treatment, outcome)
    n = len(score)
    if n == 0:
        raise ValueError("qini_curve requires at least one observation")

    order = sort_by_score_desc(score)
    t = treatment[order].astype(np.float64)
    y = outcome[order].astype(np.float64)

    cum_t = np.cumsum(t)
    cum_c = np.cumsum(1.0 - t)
    cum_yt = np.cumsum(t * y)
    cum_yc = np.cumsum((1.0 - t) * y)

    # Avoid /0 when a prefix has no control rows. Where cum_c == 0 the
    # imputed control response is also 0, so the contribution is just cum_yt.
    safe_c = np.where(cum_c > 0, cum_c, 1.0)
    incremental = cum_yt - cum_yc * (cum_t / safe_c)
    incremental = np.where(cum_c > 0, incremental, cum_yt)

    # Prepend the origin (0, 0) so the trapezoidal area is well-defined.
    share = np.concatenate(([0.0], np.arange(1, n + 1) / n))
    uplift = np.concatenate(([0.0], incremental / n))

    # Random-targeting baseline: a straight line from (0,0) to (1, ATE_sample).
    ate_sample = float(uplift[-1])
    baseline_area = 0.5 * ate_sample  # area of the triangle under the diagonal

    auc_curve = float(np.trapezoid(uplift, share))
    qini = auc_curve - baseline_area

    return QiniCurve(
        population_share=share,
        cumulative_uplift=uplift,
        qini_coefficient=qini,
        random_baseline_area=baseline_area,
    )


def qini_coefficient(
    score: NDArray1D | list[float],
    treatment: NDArray1D | list[int],
    outcome: NDArray1D | list[int],
) -> float:
    """Convenience: returns just the scalar coefficient."""
    return qini_curve(score, treatment, outcome).qini_coefficient
