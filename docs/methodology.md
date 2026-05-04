# Methodology

## What is uplift?

Uplift modeling is the problem of estimating the **individual treatment
effect** (ITE) — for a given person, by how much does a treatment change
their outcome relative to no treatment?

Formally, for a binary treatment $T \in \{0, 1\}$ and outcome $Y$:

$$
\tau(x) = \mathbb{E}[Y(1) - Y(0) \mid X = x]
$$

where $Y(1)$ and $Y(0)$ are the potential outcomes under treatment and
control. We never observe both for the same individual — only one of
$Y(0)$ or $Y(1)$ is realised. That is the *fundamental problem of causal
inference*; uplift methods are different ways of solving it from
observational or randomised data.

## Meta-learners

A *meta-learner* is a recipe for combining one or more standard
supervised learners (CatBoost, LightGBM, …) into an ITE estimator. We
implement seven.

### S-learner

A single model fitted on $(X, T) \to Y$. Uplift estimate:

$$
\hat{\tau}(x) = \hat{\mu}(x, T=1) - \hat{\mu}(x, T=0)
$$

Sharing parameters across arms acts as a useful prior when the treatment
effect is small. Loses badly when treated and control distributions
diverge (no comparable observations to interpolate from).

### T-learner

Two models — one per arm:
$\hat{\mu}_0$ on control rows, $\hat{\mu}_1$ on treated.
Uplift = $\hat{\mu}_1(x) - \hat{\mu}_0(x)$. Opposite trade-off to
S-learner: never shares signal across arms, so tends to overfit when one
arm is small.

### X-learner (Künzel et al. 2019)

Stage 1: T-learner.
Stage 2: impute counterfactual differences

$$
D^{(1)}_i = Y^{(1)}_i - \hat{\mu}_0(X^{(1)}_i),\quad
D^{(0)}_i = \hat{\mu}_1(X^{(0)}_i) - Y^{(0)}_i
$$

then fit $\hat{\tau}_0$ on $(X^{(0)}, D^{(0)})$ and $\hat{\tau}_1$ on
$(X^{(1)}, D^{(1)})$. Combine via propensity:

$$
\hat{\tau}(x) = e(x) \cdot \hat{\tau}_0(x) + (1 - e(x)) \cdot \hat{\tau}_1(x)
$$

We clip $e(x) \in [0.05, 0.95]$ to keep stage-2 stable on small samples.

### R-learner (Nie & Wager 2021)

Cross-fit nuisances $\hat{m}(x) = \mathbb{E}[Y \mid X]$ and $\hat{e}(x)
= P(T=1 \mid X)$, then minimise

$$
\mathcal{L}(\tau) = \sum_i \left( (Y_i - \hat{m}(X_i)) - (T_i - \hat{e}(X_i)) \cdot \tau(X_i) \right)^2
$$

We re-express it as a weighted regression on residuals. K=5-fold
cross-fitting by default; less and stage-1 leakage shows up as inflated
training-fold Qini.

### DR-learner (Kennedy 2023)

Same cross-fit skeleton as R-learner, but stage-2 target is the
*doubly-robust pseudo-outcome*:

$$
\psi_i = \hat{\mu}_1(X_i) - \hat{\mu}_0(X_i)
       + \frac{T_i}{\hat{e}(X_i)} \big(Y_i - \hat{\mu}_1(X_i)\big)
       - \frac{1 - T_i}{1 - \hat{e}(X_i)} \big(Y_i - \hat{\mu}_0(X_i)\big)
$$

Unbiased for $\tau(X)$ when *either* the outcome model *or* the
propensity model is correct.

### Class transformation (Jaskowski & Jaroszewicz 2012)

Define $Z = T \cdot Y + (1 - T) \cdot (1 - Y)$. Under randomised treatment
with $P(T = 1) = 0.5$, $2 P(Z=1 \mid X) - 1 = \tau(x)$. We re-weight by
$1 / P(T = t_i)$ to handle non-balanced propensity.

### Causal forest (Athey & Wager 2019)

Forest of "honest" causal trees with sample splitting. We wrap
`econml.dml.CausalForestDML` for a uniform `predict_uplift` interface.

## Metrics

### Qini coefficient

Sort observations by predicted uplift descending. Walking down that list,
plot

$$
\text{cum}_k = \frac{1}{N} \left( n^{(1)}_{Y, k} - n^{(0)}_{Y, k} \cdot \frac{n^{(1)}_k}{n^{(0)}_k} \right)
$$

against $k / N$. Qini = area between this curve and the random-targeting
diagonal.

### AUUC

Same idea but with raw cumulative differences instead of the
propensity-reweighted version. We report the perfect-ranking-normalised
variant so values live in $[-1, 1]$ comparably across datasets.

### uplift@k

Realised mean(Y\|T=1, top-k) − mean(Y\|T=0, top-k). Useful when you
only target a fraction of the population.

### Bootstrap confidence intervals

* **Percentile** — empirical 2.5/97.5 quantiles. Cheap.
* **BCa (bias-corrected accelerated)** — Efron's adjustment for bias and
  skewness using a jackknife estimate of acceleration. Recommended for
  Qini, which is meaningfully skewed on small samples.

`paired_bootstrap_test` shares resampled indices between two models so
the comparison cancels shared sampling noise; we use it for "is model A
significantly better than model B?".

## Robustness

* **Permutation importance for uplift.** Shuffle each feature, measure
  the drop in Qini. Different from sklearn's permutation importance,
  which scores against $Y$ — we score against the uplift metric directly.
* **Drop-feature stability.** Refit dropping each feature (or group) and
  report Qini delta.
* **Learning curves.** Refit on increasing fractions; tells you whether
  more data would help.
* **Overlap diagnostics.** Estimate propensity, report ESS and the
  fraction of observations in the clipping tails. IPW-flavoured
  estimators (X / R / DR) need decent overlap to behave.

## References

* Künzel, S., Sekhon, J., Bickel, P., Yu, B. (2019). *Metalearners for
  estimating heterogeneous treatment effects using machine learning.*
  PNAS.
* Athey, S., Wager, S. (2019). *Estimation and Inference of Heterogeneous
  Treatment Effects Using Random Forests.* JASA.
* Nie, X., Wager, S. (2021). *Quasi-Oracle Estimation of Heterogeneous
  Treatment Effects.* Biometrika.
* Kennedy, E. H. (2023). *Towards optimal doubly robust estimation of
  heterogeneous causal effects.* Electronic Journal of Statistics.
* Jaskowski, M., Jaroszewicz, S. (2012). *Uplift modeling for clinical
  trial data.* ICML Workshop on Clinical Data Analysis.
* Radcliffe, N. (2007). *Using control groups to target on predicted lift.*
  Direct Marketing Analytics Journal.
