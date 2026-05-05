"""Shared helpers for metric implementations.

Kept private (`_common`) because the only consumers are the metric modules
themselves. Pulling these into a public utility would tempt callers to
build on internal contracts.
"""

from __future__ import annotations

from typing import Any

import numpy as np

NDArray1D = np.ndarray[Any, np.dtype[Any]]


def as_1d_arrays(*arrays: NDArray1D | list[Any]) -> tuple[NDArray1D, ...]:
    """Coerce inputs to 1-D numpy arrays of equal length.

    Returns a tuple in input order. Raises ValueError on shape mismatch.
    """
    out = tuple(np.asarray(a).ravel() for a in arrays)
    n0 = len(out[0])
    for i, arr in enumerate(out[1:], start=1):
        if len(arr) != n0:
            raise ValueError(
                f"input arrays must have the same length; arr[0]={n0}, arr[{i}]={len(arr)}"
            )
    return out


def sort_by_score_desc(score: NDArray1D) -> NDArray1D:
    """Stable sort indices by `score` descending, breaking ties by row order.

    Numpy's sort is stable for kind='stable', and we sort the negated score
    so descending order falls out naturally. The original row index breaks
    ties — this is what gives us reproducibility across runs even when many
    scores are exactly equal (common with tree models on small data).
    """
    return np.argsort(-score, kind="stable")


def make_bucket_indices(n: int, n_buckets: int) -> NDArray1D:
    """1-indexed bucket assignment for `n` items split into `n_buckets`.

    Earlier buckets get the extra row when n isn't divisible by n_buckets,
    matching `np.array_split`. The output is positional, intended to apply
    AFTER `sort_by_score_desc` — so bucket 1 is the highest-scoring band.
    """
    sizes = [len(s) for s in np.array_split(np.arange(n), n_buckets)]
    return np.repeat(np.arange(1, n_buckets + 1), sizes)
