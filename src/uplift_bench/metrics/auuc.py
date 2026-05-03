"""Area Under the Uplift Curve (AUUC).

Distinguished from Qini by the y-axis: AUUC plots the *raw* difference of
treated and control responder counts at each prefix, rather than the
re-weighted incremental uplift used by Qini. The two coincide when the
treatment/control sample sizes are perfectly balanced; they diverge as the
imbalance grows.

We report AUUC normalised by the theoretical max (perfect-ranking AUUC)
so the reported number lives in roughly [-1, 1] and is comparable across
datasets. The unnormalised area is also returned for users who need it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from uplift_bench.metrics._common import NDArray1D, as_1d_arrays, sort_by_score_desc


@dataclass(frozen=True, slots=True)
class AUUCResult:
    auuc_raw: float  # area under the absolute-counts uplift curve
    auuc_normalized: float  # auuc_raw / area_under_perfect_ranking
    population_share: NDArray1D
    cumulative_uplift: NDArray1D


def auuc(
    score: NDArray1D | list[float],
    treatment: NDArray1D | list[int],
    outcome: NDArray1D | list[int],
) -> AUUCResult:
    """Compute AUUC and the underlying curve."""
    score, treatment, outcome = as_1d_arrays(score, treatment, outcome)
    n = len(score)
    if n == 0:
        raise ValueError("auuc requires at least one observation")

    order = sort_by_score_desc(score)
    t = treatment[order].astype(np.float64)
    y = outcome[order].astype(np.float64)

    cum_yt = np.cumsum(t * y)
    cum_yc = np.cumsum((1.0 - t) * y)
    incremental = cum_yt - cum_yc

    share = np.concatenate(([0.0], np.arange(1, n + 1) / n))
    uplift = np.concatenate(([0.0], incremental / n))
    raw = float(np.trapezoid(uplift, share))

    # Perfect-ranking baseline: the same arrays sorted by *true* y * (2t - 1)
    # (i.e. responders treated first, then non-responders, etc.). On
    # observed data this overshoots the true ceiling but it's the standard
    # normalisation used in the uplift literature.
    perfect_order = sort_by_score_desc(y * (2.0 * t - 1.0))
    pt = treatment[order][perfect_order].astype(np.float64)
    py = outcome[order][perfect_order].astype(np.float64)
    perfect_cum = np.cumsum(pt * py) - np.cumsum((1.0 - pt) * py)
    perfect_share = np.concatenate(([0.0], np.arange(1, n + 1) / n))
    perfect_uplift = np.concatenate(([0.0], perfect_cum / n))
    perfect_area = float(np.trapezoid(perfect_uplift, perfect_share))

    norm = raw / perfect_area if perfect_area > 0 else 0.0

    return AUUCResult(
        auuc_raw=raw,
        auuc_normalized=norm,
        population_share=share,
        cumulative_uplift=uplift,
    )
