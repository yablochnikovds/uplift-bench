"""Bootstrap confidence intervals.

We support two variants:

* **Percentile** — the simple one. Take the empirical 2.5/97.5 quantiles
  of the bootstrap distribution. Cheap and good enough for symmetric
  metrics on big samples.
* **BCa (bias-corrected accelerated)** — Efron's improvement on percentile.
  Adjusts for bias and skewness using a jackknife estimate of acceleration.
  Substantially more accurate on small samples or skewed metrics like
  Qini, which is why we recommend it as the default.

Plus `paired_bootstrap_test`: given two metric values computed from the
same data (e.g. Qini for model A vs Qini for model B), is A significantly
better than B at level alpha? Implemented as a bootstrap on the *difference*
of metrics, sharing the resampled indices between A and B (paired) so the
comparison cancels out shared noise.

Implementation notes:

* We use joblib for parallelism. For metrics that take ~1 ms each and a
  default n_boot=1000, the overhead of a process pool dominates; we default
  `n_jobs=1` and let the user opt in.
* The bootstrap RNG is seeded so reruns are bit-identical. Inside parallel
  workers we derive child seeds via SeedSequence.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
from joblib import Parallel, delayed
from scipy import stats

from uplift_bench.metrics._common import NDArray1D

MetricFn = Callable[[NDArray1D, NDArray1D, NDArray1D], float]
CIMethod = Literal["percentile", "bca"]


@dataclass(frozen=True, slots=True)
class BootstrapCI:
    point: float
    lower: float
    upper: float
    method: CIMethod
    alpha: float
    n_boot: int

    def as_dict(self) -> dict[str, float | str | int]:
        return {
            "point": self.point,
            "lower": self.lower,
            "upper": self.upper,
            "method": self.method,
            "alpha": self.alpha,
            "n_boot": self.n_boot,
        }


def _resample_metric(
    metric: MetricFn,
    score: NDArray1D,
    t: NDArray1D,
    y: NDArray1D,
    indices: NDArray1D,
) -> float:
    return metric(score[indices], t[indices], y[indices])


def _bootstrap_indices(n: int, n_boot: int, seed: int) -> NDArray1D:
    """Generate (n_boot, n) array of bootstrap row indices."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=(n_boot, n))


def bootstrap_ci(
    metric: MetricFn,
    score: NDArray1D,
    treatment: NDArray1D,
    outcome: NDArray1D,
    *,
    n_boot: int = 1000,
    alpha: float = 0.05,
    method: CIMethod = "bca",
    seed: int = 0,
    n_jobs: int = 1,
) -> BootstrapCI:
    """Bootstrap a confidence interval for a single metric.

    Parameters
    ----------
    metric
        Callable taking (score, treatment, outcome) → float.
    score, treatment, outcome
        Aligned arrays.
    n_boot
        Number of bootstrap resamples. 1000 is the textbook minimum for
        a 95% CI; bump to 5000 for tighter Qini CIs on small datasets.
    alpha
        Significance level. 0.05 → 95% CI.
    method
        'percentile' or 'bca'.
    seed
        Bootstrap RNG seed for reproducibility.
    n_jobs
        joblib parallelism. >1 helps when `metric` itself takes more than
        a few ms; for cheap metrics the IPC overhead dominates.

    Returns
    -------
    BootstrapCI
        Point estimate (computed on the original data), lower, upper.
    """
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if n_boot < 50:
        raise ValueError(f"n_boot must be >= 50 for a meaningful CI, got {n_boot}")

    score = np.asarray(score)
    treatment = np.asarray(treatment)
    outcome = np.asarray(outcome)
    n = len(score)
    if n == 0:
        raise ValueError("bootstrap_ci needs at least one observation")

    point = float(metric(score, treatment, outcome))
    indices = _bootstrap_indices(n, n_boot, seed)

    if n_jobs == 1:
        boots = np.array(
            [_resample_metric(metric, score, treatment, outcome, idx) for idx in indices]
        )
    else:
        boots_list = Parallel(n_jobs=n_jobs)(
            delayed(_resample_metric)(metric, score, treatment, outcome, idx) for idx in indices
        )
        boots = np.asarray(boots_list, dtype=np.float64)

    if method == "percentile":
        lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    elif method == "bca":
        lo, hi = _bca_interval(point, boots, score, treatment, outcome, metric, alpha)
    else:
        raise ValueError(f"unknown method {method!r}")

    return BootstrapCI(
        point=point,
        lower=float(lo),
        upper=float(hi),
        method=method,
        alpha=alpha,
        n_boot=n_boot,
    )


def _bca_interval(
    point: float,
    boots: NDArray1D,
    score: NDArray1D,
    treatment: NDArray1D,
    outcome: NDArray1D,
    metric: MetricFn,
    alpha: float,
) -> tuple[float, float]:
    """Efron's bias-corrected accelerated interval.

    Reference: Efron & Tibshirani (1993), An Introduction to the Bootstrap,
    chapter 14.
    """
    # Bias correction: z0 = Phi^{-1}(P(boots < point))
    n_boot = len(boots)
    p_below = float(np.mean(boots < point))
    # Clip to avoid Phi^{-1}(0) = -inf when the point is at the boundary.
    p_below = np.clip(p_below, 1e-10, 1 - 1e-10)
    z0 = stats.norm.ppf(p_below)

    # Acceleration via jackknife.
    n = len(score)
    jack = np.empty(n)
    full_idx = np.arange(n)
    for i in range(n):
        mask = full_idx != i
        jack[i] = metric(score[mask], treatment[mask], outcome[mask])
    jack_mean = jack.mean()
    diffs = jack_mean - jack
    denom = 6.0 * (np.sum(diffs**2)) ** 1.5
    if denom <= 0:  # all jackknife replicates equal — fall back to percentile
        lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
        return float(lo), float(hi)
    a = float(np.sum(diffs**3) / denom)

    z_lo = stats.norm.ppf(alpha / 2)
    z_hi = stats.norm.ppf(1 - alpha / 2)
    a1 = stats.norm.cdf(z0 + (z0 + z_lo) / (1 - a * (z0 + z_lo)))
    a2 = stats.norm.cdf(z0 + (z0 + z_hi) / (1 - a * (z0 + z_hi)))

    lo = float(np.quantile(boots, np.clip(a1, 0.0, 1.0)))
    hi = float(np.quantile(boots, np.clip(a2, 0.0, 1.0)))
    # In rare cases of extreme skew BCa can flip lo > hi — sort defensively.
    if lo > hi:
        lo, hi = hi, lo
    del n_boot  # silence unused-var; kept for readability of the formula
    return lo, hi


def paired_bootstrap_test(
    metric: MetricFn,
    score_a: NDArray1D,
    score_b: NDArray1D,
    treatment: NDArray1D,
    outcome: NDArray1D,
    *,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict[str, float]:
    """Test whether `metric(A) > metric(B)` significantly.

    Uses paired bootstrap (same indices for both) so the comparison
    cancels shared sampling noise. Returns:

    * `observed_diff`  - metric(A) - metric(B) on the original sample.
    * `ci_low`/`ci_high` — 95% percentile CI of the resampled difference.
    * `p_value_one_sided` — bootstrap test p-value for H0: metric(A) ≤
      metric(B), H1: metric(A) > metric(B). Computed via the centered
      bootstrap distribution (Efron & Tibshirani 1993, §16.4): we
      recenter `diffs` to be a draw from the null and ask how often the
      recentered statistic exceeds `observed_diff`.

    The recentered formula simplifies algebraically to
    `mean(diffs <= 0)` — but only if the centering shift `observed_diff`
    is the *only* shift between empirical and null distribution. We make
    that assumption explicit by computing it as such.
    """
    score_a_arr = np.asarray(score_a)
    score_b_arr = np.asarray(score_b)
    treatment_arr = np.asarray(treatment)
    outcome_arr = np.asarray(outcome)
    if not (len(score_a_arr) == len(score_b_arr) == len(treatment_arr) == len(outcome_arr)):
        raise ValueError("all arrays must have the same length")

    n = len(score_a_arr)
    point_a = metric(score_a_arr, treatment_arr, outcome_arr)
    point_b = metric(score_b_arr, treatment_arr, outcome_arr)
    observed_diff = float(point_a - point_b)

    indices = _bootstrap_indices(n, n_boot, seed)
    diffs = np.array(
        [
            metric(score_a_arr[idx], treatment_arr[idx], outcome_arr[idx])
            - metric(score_b_arr[idx], treatment_arr[idx], outcome_arr[idx])
            for idx in indices
        ]
    )

    lo, hi = np.quantile(diffs, [0.025, 0.975])
    # Recentered bootstrap p-value: the null distribution is `diffs - observed_diff`,
    # so P(null >= observed_diff) = P(diffs - observed_diff >= observed_diff)
    # = P(diffs >= 2 * observed_diff). For symmetric distributions this
    # roughly matches `mean(diffs <= 0)` but is the canonical Efron form
    # for skewed metrics.
    p_value = float(np.mean(diffs - observed_diff >= observed_diff))
    return {
        "metric_a": float(point_a),
        "metric_b": float(point_b),
        "observed_diff": observed_diff,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_value_one_sided": p_value,
    }
