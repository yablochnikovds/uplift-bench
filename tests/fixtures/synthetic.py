"""Synthetic uplift dataset generator.

Why this lives here, not in `src/`: nothing in production should ever depend
on synthetic data. The generator exists exclusively to test that our models
and metrics behave correctly under known ground truth — which is impossible
on real data because the true individual treatment effect is never observed.

The DGP follows Athey & Imbens (2016) "Recursive partitioning for heterogeneous
causal effects" — feature-driven heterogeneous treatment effects under
unconfounded treatment assignment.

    Y(0) = mu0(X) + epsilon0
    Y(1) = mu0(X) + tau(X) + epsilon1
    T    ~ Bernoulli(propensity(X))
    Y    = T * Y(1) + (1 - T) * Y(0)

`tau(X)` is the *true* individual uplift. Models should rank observations by
their score in roughly the same order tau(X) ranks them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

OutcomeKind = Literal["binary", "continuous"]


@dataclass(frozen=True, slots=True)
class SyntheticDataset:
    """Container around the materialized DataFrame plus ground-truth uplift."""

    df: pd.DataFrame  # columns: f0..fN, treatment, outcome
    true_uplift: np.ndarray  # tau(X_i) per row
    true_propensity: np.ndarray  # P(T=1 | X_i)
    feature_names: list[str]
    treatment_col: str = "treatment"
    outcome_col: str = "outcome"

    @property
    def n(self) -> int:
        return len(self.df)


def make_uplift_dataset(
    n_samples: int = 5000,
    n_features: int = 10,
    n_informative_uplift: int = 3,
    treatment_share: float = 0.5,
    propensity_drift: float = 0.0,
    noise: float = 1.0,
    outcome: OutcomeKind = "binary",
    seed: int = 0,
) -> SyntheticDataset:
    """Generate a dataset where individual treatment effect is known.

    Parameters
    ----------
    n_samples
        Number of rows.
    n_features
        Total number of covariates (X). Only the first `n_informative_uplift`
        of them affect tau(X); the rest are uninformative noise to make
        feature importance tests meaningful.
    n_informative_uplift
        Number of features that actually drive heterogeneous treatment effect.
    treatment_share
        Marginal probability of treatment when `propensity_drift == 0`.
        Used as the constant in the propensity logit.
    propensity_drift
        Coefficient on f0 in the propensity model. 0 → fully randomized
        (RCT-like), > 0 → confounded by f0 (observational-like). Useful for
        stressing IPS-style estimators.
    noise
        Std of additive Gaussian noise on the latent outcome. For binary
        outcomes this also controls how separable the classes are.
    outcome
        "binary" → Bernoulli outcome via a logistic link, "continuous" → raw
        latent value.
    seed
        Reproducibility.

    Returns
    -------
    SyntheticDataset
        With the realised data plus per-row true uplift.
    """
    if n_informative_uplift > n_features:
        raise ValueError("n_informative_uplift cannot exceed n_features")
    if not 0 < treatment_share < 1:
        raise ValueError("treatment_share must be in (0, 1)")

    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_features))

    # mu0(X): baseline outcome under control. A few features drive it.
    mu0 = 0.5 * X[:, 0] - 0.3 * X[:, 1] + 0.1 * X[:, 2] ** 2

    # tau(X): heterogeneous treatment effect. Built from the first
    # n_informative_uplift features. The signs and magnitudes are chosen so
    # that the marginal ATE is positive but tau(X) flips sign for a
    # meaningful minority of the population — that's the interesting
    # case for uplift modeling (without sign flips, S-learner trivially wins).
    tau = np.zeros(n_samples)
    for j in range(n_informative_uplift):
        coef = 1.0 if j == 0 else 0.7**j
        tau += coef * X[:, j]
    tau = 0.5 * tau / np.sqrt(n_informative_uplift)
    # Add a positive baseline so the marginal ATE > 0 and we still get a
    # meaningful but minority share of sign flips. Without this the DGP is
    # symmetric around 0 and S-learner has no signal direction to learn.
    tau = tau + 0.30

    # Propensity. logit(p) = a + b * f0. b=0 gives a randomized trial.
    logit_p = np.log(treatment_share / (1 - treatment_share)) + propensity_drift * X[:, 0]
    propensity = 1.0 / (1.0 + np.exp(-logit_p))
    treatment = (rng.uniform(size=n_samples) < propensity).astype(np.int8)

    # Realised potential outcomes.
    eps = rng.normal(0.0, noise, size=n_samples)
    y_latent = mu0 + treatment * tau + eps

    if outcome == "binary":
        prob = 1.0 / (1.0 + np.exp(-y_latent))
        y = (rng.uniform(size=n_samples) < prob).astype(np.int8)
    elif outcome == "continuous":
        y = y_latent
    else:  # pragma: no cover — Literal makes this unreachable in practice
        raise ValueError(f"unknown outcome kind: {outcome!r}")

    feature_names = [f"f{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df["treatment"] = treatment
    df["outcome"] = y

    return SyntheticDataset(
        df=df,
        true_uplift=tau,
        true_propensity=propensity,
        feature_names=feature_names,
    )
