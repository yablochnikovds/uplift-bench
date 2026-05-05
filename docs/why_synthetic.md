# Why a synthetic dataset

This page explains why uplift-bench ships and uses a synthetic
data-generating process (DGP) alongside the four real datasets
(Hillstrom, Criteo, Lenta, RetailHero, MegaFon).

## The fundamental problem of causal inference

For an individual $i$ with covariates $X_i$ and binary treatment
indicator $T_i \in \{0, 1\}$, define potential outcomes $Y_i(0)$ and
$Y_i(1)$ — what would happen with no treatment vs with treatment. The
**individual treatment effect (ITE)** is

$$
\tau(x) = \mathbb{E}\big[Y(1) - Y(0) \mid X = x\big].
$$

Reality always reveals only one of the two: we observe
$Y_i = T_i \cdot Y_i(1) + (1 - T_i) \cdot Y_i(0)$. The other is the
**counterfactual** and is *fundamentally unobservable* (Holland 1986).

Consequence: on real data, **we cannot directly verify that a
meta-learner is estimating $\tau$ correctly.** We can only check that
its score *ranks* observations consistently with realised outcomes
(via Qini, AUUC, etc.). A model could achieve a strong Qini for the
wrong reason — say, by picking up noise that happens to correlate with
the marginal outcome, not with the treatment effect.

## What synthetic data buys us

Synthetic data generated from a known DGP gives us the true $\tau(X_i)$
for every row. With that we can:

* Verify that a meta-learner's predicted uplift is **rank-correlated
  with the true τ**. Spearman ρ between predictions and ground-truth τ
  is a hard test that Qini on real data is not.
* Verify that the **oracle ranking by true τ** gives the highest
  achievable Qini (sanity-check the Qini implementation itself).
* Construct datasets with deliberately high heterogeneity, sign-flips,
  or confounding — situations where meta-learners *should* differ
  meaningfully. Real RCT-style datasets like Hillstrom and Criteo
  Uplift v2 have low heterogeneity, so they can't differentiate methods
  on their own.

## Precedent in the literature

Every reference uplift / heterogeneous-treatment-effect library does
the same thing:

* `causalml.dataset.synthetic_data` — Uber's [causalml](https://github.com/uber/causalml).
* `econml.utilities.dgp_data` — Microsoft's [EconML](https://github.com/py-why/EconML).
* `dowhy.datasets.linear_dataset` — PyWhy's [DoWhy](https://github.com/py-why/dowhy).
* All of Künzel et al. 2019, Athey & Wager 2019, Nie & Wager 2021,
  Kennedy 2023 include simulation studies on synthetic DGPs as the
  primary validation. Real-data tables come second.

## Where to find ours

The DGP itself: [`src/uplift_bench/data/synthetic.py`](https://github.com/yablochnikovds/uplift-bench/blob/main/src/uplift_bench/data/synthetic.py).
Documented as a Python function `make_uplift_dataset(...)`.

The same DGP wrapped as a `DatasetLoader` so it plugs into the
benchmark like a real dataset:
[`src/uplift_bench/data/synthetic_loader.py`](https://github.com/yablochnikovds/uplift-bench/blob/main/src/uplift_bench/data/synthetic_loader.py).

The Hydra config that runs the benchmark on it with confounding turned
on (`propensity_drift=1.5`):
[`configs/dataset/synthetic.yaml`](https://github.com/yablochnikovds/uplift-bench/blob/main/configs/dataset/synthetic.yaml).
