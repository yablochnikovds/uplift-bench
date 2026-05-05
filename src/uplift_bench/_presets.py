"""Shared helpers for the benchmark / comparison-plot scripts.

These two presets — dataset loader-params and a "fast" model-kwarg recipe —
need to be identical between `run_full_benchmark.py`, `build_comparison_plots.py`,
and `extend_results_with_new_metrics.py`. Putting them in one place removes
the silent-divergence risk when a new dataset or model is added.
"""

from __future__ import annotations

from typing import Any

from uplift_bench.models.utils import build_model_kwargs


def loader_params(
    dataset: str,
    *,
    seed: int = 0,
    criteo_subsample: int | None = 1_000_000,
) -> dict[str, Any]:
    """Per-dataset loader kwargs used by the benchmark scripts."""
    if dataset == "hillstrom":
        return {"treatment_arm": "Womens E-Mail", "outcome": "visit"}
    if dataset == "criteo":
        return {
            "outcome": "visit",
            "subsample": criteo_subsample,
            "subsample_seed": 42,
        }
    if dataset == "synthetic":
        # Confounded heterogeneous DGP — the standard setup we use to
        # differentiate meta-learners.
        return {
            "n_samples": 10_000,
            "n_features": 10,
            "n_informative_uplift": 4,
            "treatment_share": 0.5,
            "propensity_drift": 1.5,
            "noise": 0.5,
            "outcome": "binary",
            "seed": seed,
        }
    # lenta / retailhero / megafon: no extra params beyond what the
    # loader's __init__ defaults to.
    return {}


def fast_model_kwargs(model_name: str, *, seed: int = 0) -> dict[str, Any]:
    """Speed-tuned kwargs used by the comparison plotting + the
    metric-extension utility. CatBoost / LightGBM at 200 iterations,
    causal forest at 80 trees — fast enough for a re-fit pass without
    materially changing the metrics relative to the main benchmark.
    """
    if model_name == "causal_forest":
        return build_model_kwargs(
            model_name,
            extra_params={"n_estimators": 80, "min_samples_leaf": 30},
            seed=seed,
        )
    return build_model_kwargs(
        model_name,
        base_learner_cfg={
            "name": "catboost",
            "params": {"iterations": 200, "n_estimators": 200},
        },
        seed=seed,
    )
