"""Policy value of a treatment-by-threshold policy.

Given predicted uplift τ̂(X), define a policy
    π_c(X) = 1{ τ̂(X) > c }
that treats only customers above some threshold c. The **policy value**
at budget b is the expected outcome when we apply this policy:

    V(b) = E[ Y(π_c(X)) ]

where c is set so that π_c treats exactly the top-b fraction.

We estimate V(b) via the inverse-propensity-weighted estimator
(Manski 2004; Athey & Wager 2021, "Policy Learning with Observational
Data", Econometrica):

    V_hat(b) = (1/N) sum_i [ Y_i * I{T_i = π_c(X_i)} / P(T_i | X_i) ]

assuming randomised treatment with constant marginal propensity p,
this simplifies to a difference-in-means form on the targeted subset.

We provide it as a curve over budget — `policy_value_curve(...)` — so
practitioners can read off the expected outcome for any budget level.

Note: this is a single-step IPW estimator, not a doubly-robust one.
For unbiased policy evaluation under confounding use the AIPW form;
for our RCT-style benchmark datasets, plain IPW is sufficient.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from uplift_bench.metrics._common import NDArray1D, as_1d_arrays, sort_by_score_desc


@dataclass(frozen=True, slots=True)
class PolicyValueCurve:
    budgets: NDArray1D  # in [0, 1] — fraction of population treated
    policy_values: NDArray1D  # E[Y(pi_c(X))] estimate at each budget
    treat_none_value: float  # E[Y(0)]
    treat_all_value: float  # E[Y(1)]


def policy_value_curve(
    score: NDArray1D | list[float],
    treatment: NDArray1D | list[int],
    outcome: NDArray1D | list[int],
    *,
    budgets: Sequence[float] = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0),
) -> PolicyValueCurve:
    """Estimate policy value at each budget level.

    `budgets[0]` should be 0 (treat none) and `budgets[-1]` should be 1
    (treat everyone) — both yield natural baselines (control mean,
    treated mean).
    """
    score_arr, t_arr, y_arr = as_1d_arrays(score, treatment, outcome)
    n = len(score_arr)
    if n == 0:
        raise ValueError("policy_value_curve requires at least one observation")

    order = sort_by_score_desc(score_arr)
    t_ord = t_arr[order].astype(np.float64)
    y_ord = y_arr[order].astype(np.float64)

    # Marginal propensity (RCT assumption — for our datasets).
    p_t = float(t_arr.mean())
    if not 0 < p_t < 1:
        raise ValueError(f"policy_value needs both arms; got marginal P(T=1) = {p_t}")

    treat_none = float(y_ord[t_ord == 0].mean()) if (t_ord == 0).any() else float("nan")
    treat_all = float(y_ord[t_ord == 1].mean()) if (t_ord == 1).any() else float("nan")

    values = np.empty(len(budgets), dtype=np.float64)
    for i, b in enumerate(budgets):
        if not 0 <= b <= 1:
            raise ValueError(f"budget {b} not in [0, 1]")
        n_treat = round(n * b)
        if n_treat == 0:
            values[i] = treat_none
            continue
        if n_treat == n:
            values[i] = treat_all
            continue
        # Top-b are policy-treated; rest are policy-untreated.
        # IPW estimator: weight Y_i by 1/p_t for top rows that were
        # actually treated in the data, and by 1/(1-p_t) for bottom rows
        # that were actually controlled. Sum and divide by N.
        top = np.zeros(n, dtype=bool)
        top[:n_treat] = True
        # Indicator that observed T matches policy.
        match_top = top & (t_ord == 1)
        match_bot = (~top) & (t_ord == 0)
        ipw_sum = (y_ord[match_top] / p_t).sum() + (y_ord[match_bot] / (1 - p_t)).sum()
        values[i] = float(ipw_sum / n)

    return PolicyValueCurve(
        budgets=np.asarray(budgets, dtype=np.float64),
        policy_values=values,
        treat_none_value=treat_none,
        treat_all_value=treat_all,
    )


def policy_value_at(
    score: NDArray1D | list[float],
    treatment: NDArray1D | list[int],
    outcome: NDArray1D | list[int],
    budget: float,
) -> float:
    """Convenience: scalar policy value at a single budget."""
    curve = policy_value_curve(score, treatment, outcome, budgets=[budget])
    return float(curve.policy_values[0])
