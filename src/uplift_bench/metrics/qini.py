"""Qini coefficient and Qini curve.

Definition (Radcliffe 2007, "Using control groups to target on predicted
lift", Direct Marketing Analytics Journal). Sort observations by predicted
uplift descending. Walking down that ordered list, plot:

    x-axis : cumulative population share k / N
    y-axis : (n_treated_responders[:k] - n_control_responders[:k]
             * n_treated[:k] / n_control[:k])  / N

The **raw** Qini area is the integral between this curve and the
random-targeting diagonal. The **normalised** Qini coefficient divides
that by the area achievable by a perfect-ordering model, so it sits in
roughly [-1, 1] and is comparable across datasets — this is the
convention used by `scikit-uplift.metrics.qini_auc_score` and what we
expose as `qini_coefficient`.

The perfect-ordering curve is constructed by sorting observations on
`y * (2t - 1)`: positive responders in the treated group first, negative
responders in the control group last. This is the standard sklift
convention (Gutierrez & Gerardy 2017, "Causal Inference and Uplift
Modeling: A Review of the Literature").

Two implementation notes that bit me before:

1. The ratio `n_treated[:k] / n_control[:k]` is undefined at k=0 and at
   the edge case where a prefix has zero control rows. We clamp using
   `np.where` instead of try/except to keep the function vectorised.

2. Ties in the score must be broken consistently across runs — otherwise
   `qini(score, t, y, seed=0)` and the same call later return slightly
   different numbers. We resolve ties via numpy's stable sort on the
   negated score, which preserves insertion order.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from uplift_bench.metrics._common import NDArray1D, as_1d_arrays, sort_by_score_desc


@dataclass(frozen=True, slots=True)
class QiniCurve:
    """The cumulative uplift curve and its Qini coefficient.

    `population_share` and `cumulative_uplift` describe the curve itself;
    use them for plotting. `qini_coefficient` is the normalised scalar
    (raw area / perfect area), bounded in roughly [-1, 1]. `qini_raw`
    is the un-normalised area for cross-package comparability.
    """

    population_share: NDArray1D
    cumulative_uplift: NDArray1D
    qini_coefficient: float
    qini_raw: float
    perfect_area: float
    random_baseline_area: float


def _cumulative_uplift_curve(
    score: NDArray1D,
    t: NDArray1D,
    y: NDArray1D,
) -> tuple[NDArray1D, NDArray1D, float]:
    """Helper: build the cumulative-uplift curve under a given ranking.

    Returns (population_share, cumulative_uplift_per_capita, sample_ate).
    """
    n = len(score)
    order = sort_by_score_desc(score)
    t_ord = t[order].astype(np.float64)
    y_ord = y[order].astype(np.float64)

    cum_t = np.cumsum(t_ord)
    cum_c = np.cumsum(1.0 - t_ord)
    cum_yt = np.cumsum(t_ord * y_ord)
    cum_yc = np.cumsum((1.0 - t_ord) * y_ord)

    # Avoid /0 when a prefix has no control rows yet.
    safe_c = np.where(cum_c > 0, cum_c, 1.0)
    incremental = cum_yt - cum_yc * (cum_t / safe_c)
    incremental = np.where(cum_c > 0, incremental, cum_yt)

    share = np.concatenate(([0.0], np.arange(1, n + 1) / n))
    uplift = np.concatenate(([0.0], incremental / n))
    return share, uplift, float(uplift[-1])


def qini_curve(
    score: NDArray1D | list[float],
    treatment: NDArray1D | list[int],
    outcome: NDArray1D | list[int],
) -> QiniCurve:
    """Compute the Qini curve and the normalised Qini coefficient.

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
        Curve plus the normalised coefficient `qini_coefficient` and the
        un-normalised `qini_raw`.
    """
    score_arr, t_arr, y_arr = as_1d_arrays(score, treatment, outcome)
    n = len(score_arr)
    if n == 0:
        raise ValueError("qini_curve requires at least one observation")

    share, uplift, ate_sample = _cumulative_uplift_curve(score_arr, t_arr, y_arr)
    auc_curve = float(np.trapezoid(uplift, share))

    # Random-targeting baseline: a straight line from (0,0) to (1, ATE).
    random_baseline_area = 0.5 * ate_sample
    raw = auc_curve - random_baseline_area

    # Perfect-ordering curve: rank by y * (2t - 1) — treated responders
    # first, then everyone else, then control responders last.
    # This matches `scikit-uplift.metrics.perfect_uplift_curve`.
    perfect_score = y_arr * (2.0 * t_arr - 1.0)
    _, perfect_uplift, _ = _cumulative_uplift_curve(perfect_score, t_arr, y_arr)
    perfect_auc = float(np.trapezoid(perfect_uplift, share))
    perfect_area = perfect_auc - random_baseline_area

    # Degenerate case (no responders / all-treated / all-control) gives
    # perfect_area == 0 and raw ≈ 0; callers can detect via `perfect_area`.
    qini = raw / perfect_area if perfect_area > 0 else raw

    return QiniCurve(
        population_share=share,
        cumulative_uplift=uplift,
        qini_coefficient=qini,
        qini_raw=raw,
        perfect_area=perfect_area,
        random_baseline_area=random_baseline_area,
    )


def qini_coefficient(
    score: NDArray1D | list[float],
    treatment: NDArray1D | list[int],
    outcome: NDArray1D | list[int],
) -> float:
    """Normalised Qini coefficient — bounded in roughly [-1, 1].

    Equivalent to `qini_curve(...).qini_coefficient`.
    """
    return qini_curve(score, treatment, outcome).qini_coefficient


def qini_raw(
    score: NDArray1D | list[float],
    treatment: NDArray1D | list[int],
    outcome: NDArray1D | list[int],
) -> float:
    """Un-normalised Qini area (Radcliffe 2007 raw definition).

    Useful when comparing to legacy packages that don't normalise.
    """
    return qini_curve(score, treatment, outcome).qini_raw
