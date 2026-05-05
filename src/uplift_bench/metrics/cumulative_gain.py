"""Cumulative gain curve.

Distinguished from Qini in one important way. The Qini curve plots the
*incremental* uplift (treated responders minus reweighted control
responders) at each top-k. The cumulative gain curve plots the absolute
*responder count* (or rate) at each top-k for the *treated* arm only —
no reweighting, no incremental calculation.

It answers a different question than Qini: "if I target the top-k
predicted-uplift customers and treat them all, how many responders do I
get?" That's the relevant question for marketers who don't run a hold-out
on the targeted set.

Reference: Radcliffe 2007, "Using control groups to target on predicted
lift", Direct Marketing Analytics Journal.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from uplift_bench.metrics._common import NDArray1D, as_1d_arrays, sort_by_score_desc


@dataclass(frozen=True, slots=True)
class CumulativeGainCurve:
    """Cumulative responders by predicted-uplift quantile."""

    population_share: NDArray1D
    cumulative_responders_per_capita: NDArray1D
    auc: float  # area under the curve
    random_baseline_auc: float  # area achievable by random targeting


def cumulative_gain_curve(
    score: NDArray1D | list[float],
    treatment: NDArray1D | list[int],
    outcome: NDArray1D | list[int],
) -> CumulativeGainCurve:
    """Cumulative responder rate among treated, ordered by predicted uplift.

    For each top-k fraction we count: how many of the targeted (treated)
    rows responded? Divide by N for a per-capita scale.
    """
    score_arr, t_arr, y_arr = as_1d_arrays(score, treatment, outcome)
    n = len(score_arr)
    if n == 0:
        raise ValueError("cumulative_gain_curve requires at least one observation")

    order = sort_by_score_desc(score_arr)
    t_ord = t_arr[order].astype(np.float64)
    y_ord = y_arr[order].astype(np.float64)

    # Cumulative responders among the treated subset of the top-k prefix.
    # We count only treated responders because that's what the "if I treat
    # the top k" intuition asks for.
    cum_treated_responders = np.cumsum(t_ord * y_ord)
    share = np.concatenate(([0.0], np.arange(1, n + 1) / n))
    gain = np.concatenate(([0.0], cum_treated_responders / n))

    auc = float(np.trapezoid(gain, share))
    # Random baseline: a straight line from (0,0) to (1, gain[-1]).
    baseline_auc = 0.5 * float(gain[-1])

    return CumulativeGainCurve(
        population_share=share,
        cumulative_responders_per_capita=gain,
        auc=auc,
        random_baseline_auc=baseline_auc,
    )
