# Methodological validation

This page documents how each meta-learner and metric in uplift-bench is
verified to be a faithful implementation of its source paper or
reference library. The audit was triggered by surprising results in the
initial benchmark run (all seven learners produced very similar Qini
values on Hillstrom and Criteo subsample) — the cause turned out to be a
**Qini metric normalisation bug** plus the **Hillstrom / Criteo data
having genuinely low heterogeneity**, not bugs in the learners.

## Audit summary

Independent line-by-line review against the original papers and the
canonical reference implementations (`causalml`, `EconML`,
`scikit-uplift`, `scipy.stats._resampling`). Findings:

| component | severity | resolution |
|---|---|---|
| **Qini coefficient** | **critical** | docstring claimed normalised to [-1, 1] but code returned the raw Radcliffe area difference. **Fixed**: now divides by perfect-curve area as in `scikit-uplift.metrics.qini_auc_score`; both raw and normalised values are returned. |
| Class transformation | minor | reweighting trick works only under constant-propensity RCT — true for our 4 datasets but undocumented. **Fixed**: explicit RCT-only assumption in module docstring. |
| R-learner | minor | dead-code denominator clip; non-stratified KFold. **Fixed**: removed dead clip; switched to `StratifiedKFold` on T. |
| DR-learner | minor | non-stratified KFold (had a one-arm-fold guard so didn't crash, but produced unstable propensity estimates). **Fixed**: `StratifiedKFold` on T. |
| Paired bootstrap test | minor | p-value used CI-tail proxy instead of Efron's recentered statistic. **Fixed**: now uses `mean(diffs - observed_diff ≥ observed_diff)` per Efron & Tibshirani 1993 §16.4. |
| S-learner, T-learner, X-learner, BCa CI, Causal forest | OK | no discrepancy. |

## Cross-checks against reference implementations

[tests/integration/test_method_divergence.py](https://github.com/yablochnikovds/uplift-bench/blob/main/tests/integration/test_method_divergence.py) runs three integration tests every CI build:

1. **Qini spread under strong heterogeneity.** On a synthetic DGP with
   strong heterogeneous τ(X) and confounded treatment
   (`propensity_drift=1.5`), the spread between best and worst meta-learner
   must be > 0.05 Qini. Catches the case where multiple learners
   accidentally collapse onto the same prediction.

2. **X- or DR-learner beats S-learner under confounding.** Textbook
   claim from Künzel et al. 2019 and Kennedy 2023 — under non-random
   treatment assignment, propensity-aware learners should dominate the
   plain plug-in baseline. If not, something is wrong with our wrapper.

3. **T-learner matches `causalml.BaseTClassifier`** with Spearman ρ ≥ 0.95
   when given the same base estimator. T-learner is deterministic so this
   is essentially an identity check — failing it means our wrapper is
   off-by-something.

All three pass on the current implementation.

## Per-method references

* **S-learner** — folklore "single-model" approach; no canonical paper.
  Verified against `causalml.LRSRegressor` mechanics.
* **T-learner** — Künzel et al. 2019, [PNAS 116(10), 4156–4165](https://www.pnas.org/doi/10.1073/pnas.1804597116).
  Cross-checked against `causalml.BaseTClassifier`.
* **X-learner** — same Künzel et al. 2019 paper. Implementation cross-checked
  against both `causalml.BaseXClassifier` and `econml.metalearners.XLearner`.
  In particular, the weighting direction `tau = e * tau_C + (1 - e) * tau_T`
  (control-trained τ gets weight `e`, treated-trained τ gets weight `1 − e`)
  is verified to match both libraries.
* **R-learner** — Nie & Wager 2021, [Biometrika 108(2), 299–319](https://academic.oup.com/biomet/article/108/2/299/5911092).
  R-loss formulation cross-checked against `causalml.BaseRClassifier`.
* **DR-learner** — Kennedy 2023, [Electronic Journal of Statistics 17(2), 3008–3049](https://projecteuclid.org/journals/electronic-journal-of-statistics/volume-17/issue-2/Towards-optimal-doubly-robust-estimation-of-heterogeneous-causal-effects/10.1214/23-EJS2157.full).
  DR pseudo-outcome
  $\psi = \mu_1 - \mu_0 + (T/e)(Y - \mu_1) - ((1-T)/(1-e))(Y - \mu_0)$
  is the standard form going back to Robins, Rotnitzky & Zhao (1994).
* **Class transformation** — Jaskowski & Jaroszewicz 2012, "Uplift modeling
  for clinical trial data", ICML Workshop on Clinical Data Analysis. Z
  transformation `Z = T·Y + (1-T)·(1-Y)` and the result `2·P(Z=1|X) − 1 =
  τ(X)` under randomised treatment.
* **Causal forest** — Athey & Wager 2019, [JASA 114(528), 1611–1622](https://www.tandfonline.com/doi/full/10.1080/01621459.2017.1319839).
  Wrapper around `econml.dml.CausalForestDML`.

## Metric references

* **Qini** — Radcliffe 2007, "Using control groups to target on
  predicted lift", *Direct Marketing Analytics Journal*. We use the
  normalised form (raw area / perfect area) which is the convention in
  `scikit-uplift.metrics.qini_auc_score` and Gutierrez & Gerardy 2017.
* **AUUC** — same convention as `scikit-uplift.metrics.uplift_auc_score`.
* **Bootstrap BCa** — Efron & Tibshirani 1993, *An Introduction to the
  Bootstrap*, chapter 14. Implementation cross-checked against
  `scipy.stats._resampling._bca_interval`.

## Why Hillstrom and Criteo show small Qini values

Both Hillstrom (`Womens E-Mail` vs `No E-Mail`, `visit` outcome) and
Criteo Uplift v2 are randomised marketing experiments where the
*marginal* effect is modest:

* Hillstrom ATE on `visit` is ≈ 0.045 (15% control vs 30% treated).
* Criteo Uplift v2 ATE on `visit` is ≈ 0.005 (very small).

The normalised Qini for a perfect-ranking model is bounded by the area
under the ATE-weighted perfect curve, so even a perfect ranker on Criteo
gets a normalised Qini of ~0.7. Real models reach ~0.20–0.30 of that
ceiling, which translates to per-person Qini values around 0.001–0.005.
This is **expected behaviour for low-heterogeneity datasets** — see the
discussion in Diemert et al. 2018 ([arXiv:1810.04938](https://arxiv.org/abs/1810.04938))
about Criteo Uplift specifically.

For visualising real differences between learners, refer to the
synthetic-DGP figures in
[`results/figures/`](https://github.com/yablochnikovds/uplift-bench/tree/main/results/figures)
where heterogeneity is strong by construction.
