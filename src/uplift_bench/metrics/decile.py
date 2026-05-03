"""Per-decile uplift table.

Splits the population into k equal-size buckets ordered by predicted uplift,
reports the realised uplift in each bucket. The natural diagnostic to look
at after Qini/AUUC: monotone-decreasing buckets confirm the model has
useful ordering; a flat or zigzagging table tells you the "good" Qini was
luck on the head and tail.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from uplift_bench.metrics._common import NDArray1D, as_1d_arrays, sort_by_score_desc


def decile_table(
    score: NDArray1D | list[float],
    treatment: NDArray1D | list[int],
    outcome: NDArray1D | list[int],
    n_buckets: int = 10,
) -> pd.DataFrame:
    """Build a per-decile (or n-tile) uplift table.

    Parameters
    ----------
    score, treatment, outcome
        Same shapes as elsewhere.
    n_buckets
        Number of equal-size buckets. Default 10 (deciles).

    Returns
    -------
    pd.DataFrame
        Columns: bucket (1..n_buckets, 1 = highest score), n_treat, n_ctrl,
        mean_y_treat, mean_y_ctrl, uplift, n_total.
    """
    if n_buckets < 2:
        raise ValueError(f"n_buckets must be >= 2, got {n_buckets}")

    score, treatment, outcome = as_1d_arrays(score, treatment, outcome)
    n = len(score)
    if n < n_buckets:
        raise ValueError(f"n={n} < n_buckets={n_buckets}; not enough rows")

    order = sort_by_score_desc(score)
    t = treatment[order].astype(bool)
    y = outcome[order].astype(np.float64)

    # np.array_split handles non-divisible n by giving the early buckets one
    # extra row. That matches what most published uplift packages do.
    sizes = [len(s) for s in np.array_split(np.arange(n), n_buckets)]
    bucket_idx: NDArray1D = np.repeat(np.arange(1, n_buckets + 1), sizes)

    # Per-bucket aggregates via vectorised masking — much friendlier to
    # pandas-stubs than groupby.apply with conditional returns.
    rows: list[dict[str, float | int]] = []
    for b in range(1, n_buckets + 1):
        mask = bucket_idx == b
        y_b = y[mask]
        t_b = t[mask]
        n_treat = int(t_b.sum())
        n_ctrl = int((~t_b).sum())
        mean_t = float(y_b[t_b].mean()) if n_treat else float("nan")
        mean_c = float(y_b[~t_b].mean()) if n_ctrl else float("nan")
        rows.append(
            {
                "bucket": b,
                "n_treat": n_treat,
                "n_ctrl": n_ctrl,
                "mean_y_treat": mean_t,
                "mean_y_ctrl": mean_c,
                "uplift": mean_t - mean_c,
                "n_total": n_treat + n_ctrl,
            }
        )

    return pd.DataFrame(rows)
