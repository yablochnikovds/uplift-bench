"""End-to-end pipeline for one (model, dataset, seed) combination.

This is the function the CLI / Hydra entry point ultimately calls. Kept
free of Hydra-specific imports so it's also reachable from notebooks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from uplift_bench.data.factory import make_loader
from uplift_bench.data.splits import make_splits
from uplift_bench.metrics.auuc import auuc
from uplift_bench.metrics.bootstrap import bootstrap_ci
from uplift_bench.metrics.decile import decile_table
from uplift_bench.metrics.qini import qini_coefficient, qini_curve
from uplift_bench.metrics.uplift_at_k import uplift_at_k
from uplift_bench.models.factory import make_model
from uplift_bench.robustness.overlap import overlap_diagnostics
from uplift_bench.robustness.permutation import permutation_uplift_importance
from uplift_bench.tracking.mlflow_logger import start_run
from uplift_bench.utils.io import dump_json, ensure_dir
from uplift_bench.utils.logging import get_logger
from uplift_bench.utils.reproducibility import seed_everything
from uplift_bench.viz.diagnostic_plots import (
    plot_calibration,
    plot_decile_uplift,
    plot_permutation_importance,
    plot_propensity_histogram,
)
from uplift_bench.viz.qini_curve import plot_qini_curve
from uplift_bench.viz.uplift_distribution import plot_uplift_distribution

log = get_logger(__name__)

UPLIFT_AT_K_FRACTIONS = (0.10, 0.20, 0.30)


@dataclass
class TrainResult:
    """Returned by `run_one`. Used by the benchmark aggregator."""

    model_name: str
    dataset_name: str
    seed: int
    qini: float
    qini_ci_lower: float
    qini_ci_upper: float
    auuc: float
    metrics: dict[str, float] = field(default_factory=dict)
    artifacts_dir: Path | None = None


def _build_model_kwargs(
    model_name: str,
    base_learner_cfg: dict[str, Any],
    extra_params: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    """Translate the Hydra config layout into model __init__ kwargs."""
    if model_name == "causal_forest":
        # Causal forest takes its own knobs; doesn't accept base_learner.
        return {"seed": seed, **extra_params}
    return {
        "base_learner": base_learner_cfg["name"],
        "base_params": base_learner_cfg.get("params", {}) or None,
        "seed": seed,
        **extra_params,
    }


def run_one(
    *,
    dataset_cfg: dict[str, Any],
    model_cfg: dict[str, Any],
    base_learner_cfg: dict[str, Any],
    split_cfg: dict[str, Any],
    bootstrap_cfg: dict[str, Any],
    robustness_cfg: dict[str, Any],
    tracking_cfg: dict[str, Any],
    seed: int,
    output_dir: Path,
) -> TrainResult:
    """Train + evaluate + log + persist one (model, dataset) combination."""
    seed_everything(seed)
    ensure_dir(output_dir)

    log.info("pipeline_start", model=model_cfg["name"], dataset=dataset_cfg["name"], seed=seed)

    loader = make_loader(
        dataset_cfg["name"],
        data_dir=dataset_cfg["data_dir"],
        **(dataset_cfg.get("loader_params") or {}),
    )
    dataset = loader.load()

    splits = make_splits(
        dataset,
        train_frac=split_cfg["train_frac"],
        val_frac=split_cfg["val_frac"],
        seed=split_cfg.get("seed", seed),
    )
    X = dataset.X
    t = dataset.t
    y = dataset.y

    X_train, t_train, y_train = X.iloc[splits.train], t[splits.train], y[splits.train]
    X_test, t_test, y_test = X.iloc[splits.test], t[splits.test], y[splits.test]

    model_kwargs = _build_model_kwargs(
        model_cfg["name"],
        base_learner_cfg,
        model_cfg.get("extra_params") or {},
        seed,
    )
    model = make_model(model_cfg["name"], **model_kwargs)

    log.info("fitting_model", model=model_cfg["name"], n_train=len(X_train))
    model.fit(X_train, t_train, y_train)

    test_preds = model.predict_uplift(X_test)

    # Metrics with CIs.
    qini_ci = bootstrap_ci(
        qini_coefficient,
        test_preds,
        t_test,
        y_test,
        n_boot=bootstrap_cfg["n_boot"],
        method=bootstrap_cfg["method"],
        alpha=bootstrap_cfg["alpha"],
        n_jobs=bootstrap_cfg["n_jobs"],
        seed=bootstrap_cfg["seed"],
    )
    auuc_res = auuc(test_preds, t_test, y_test)
    extra_metrics: dict[str, float] = {
        "auuc_raw": auuc_res.auuc_raw,
        "auuc_normalized": auuc_res.auuc_normalized,
    }
    for k_frac in UPLIFT_AT_K_FRACTIONS:
        val = uplift_at_k(test_preds, t_test, y_test, k=k_frac)
        extra_metrics[f"uplift_at_{int(k_frac * 100)}"] = val

    # Build artifacts before opening the MLflow run so we can log files.
    deciles = decile_table(test_preds, t_test, y_test, n_buckets=10)
    qcurve = qini_curve(test_preds, t_test, y_test)
    qini_path = output_dir / "qini_curve.png"
    plot_qini_curve(
        qcurve, title=f"{model_cfg['name']} on {dataset_cfg['name']}", save_path=qini_path
    )
    dist_path = output_dir / "uplift_distribution.png"
    plot_uplift_distribution(test_preds, save_path=dist_path)
    deciles_path = output_dir / "deciles.csv"
    deciles.to_csv(deciles_path, index=False)
    decile_plot_path = output_dir / "deciles.png"
    plot_decile_uplift(
        deciles,
        title=f"{model_cfg['name']} on {dataset_cfg['name']} — per-decile uplift",
        save_path=decile_plot_path,
    )
    calibration_path = output_dir / "calibration.png"
    plot_calibration(
        test_preds,
        t_test,
        y_test,
        title=f"{model_cfg['name']} on {dataset_cfg['name']} — calibration",
        save_path=calibration_path,
    )
    config_dump = {
        "dataset": dataset_cfg,
        "model": model_cfg,
        "base_learner": base_learner_cfg,
        "split": split_cfg,
        "bootstrap": bootstrap_cfg,
        "robustness": robustness_cfg,
        "seed": seed,
    }
    config_path = output_dir / "config.json"
    dump_json(config_dump, config_path)

    # Robustness — only compute the parts the config asks for.
    perm_path: Path | None = None
    perm_plot_path: Path | None = None
    if robustness_cfg.get("enable_permutation"):
        perm = permutation_uplift_importance(
            model,
            X_test,
            t_test,
            y_test,
            n_repeats=robustness_cfg.get("permutation_n_repeats", 5),
            seed=seed,
        )
        perm_path = output_dir / "permutation_importance.csv"
        perm.to_csv(perm_path, index=False)
        perm_plot_path = output_dir / "permutation_importance.png"
        plot_permutation_importance(
            perm,
            title=f"{model_cfg['name']} on {dataset_cfg['name']} — permutation importance",
            save_path=perm_plot_path,
        )

    overlap_path: Path | None = None
    overlap_plot_path: Path | None = None
    if robustness_cfg.get("enable_overlap"):
        diag = overlap_diagnostics(X_test, t_test, n_splits=3, seed=seed)
        extra_metrics["overlap_ess_ratio"] = diag.ess_ratio
        extra_metrics["overlap_pct_below_clip"] = diag.pct_below_clip
        extra_metrics["overlap_pct_above_clip"] = diag.pct_above_clip
        # Save the propensity vector for downstream analysis.
        overlap_path = output_dir / "propensity.csv"
        pd.DataFrame({"propensity": diag.propensity}).to_csv(overlap_path, index=False)
        overlap_plot_path = output_dir / "propensity.png"
        plot_propensity_histogram(
            diag.treated_propensity,
            diag.control_propensity,
            title=f"{model_cfg['name']} on {dataset_cfg['name']} — propensity overlap",
            save_path=overlap_plot_path,
        )

    # MLflow.
    run_name = (
        tracking_cfg.get("run_name") or f"{model_cfg['name']}.{dataset_cfg['name']}.seed{seed}"
    )
    with start_run(
        enabled=tracking_cfg["enabled"],
        experiment_name=tracking_cfg["experiment_name"],
        tracking_uri=tracking_cfg["tracking_uri"],
        run_name=run_name,
        tags={
            "dataset": dataset_cfg["name"],
            "model": model_cfg["name"],
            "base_learner": base_learner_cfg["name"],
        },
    ) as run:
        run.log_params(
            {
                "model": model_cfg["name"],
                "dataset": dataset_cfg["name"],
                "seed": seed,
                "base_learner": base_learner_cfg["name"],
                "base_learner_params": base_learner_cfg.get("params", {}),
                "model_extra_params": model_cfg.get("extra_params", {}),
                "split_train_frac": split_cfg["train_frac"],
                "split_val_frac": split_cfg["val_frac"],
                "bootstrap_n_boot": bootstrap_cfg["n_boot"],
                "bootstrap_method": bootstrap_cfg["method"],
                "dataset_source_hash": dataset.source_hash,
                "n_train": len(X_train),
                "n_test": len(X_test),
            }
        )
        run.log_metric_with_ci("qini", qini_ci)
        run.log_metrics(extra_metrics)
        run.log_artifact_path(qini_path)
        run.log_artifact_path(dist_path)
        run.log_artifact_path(deciles_path)
        run.log_artifact_path(config_path)
        if perm_path is not None:
            run.log_artifact_path(perm_path)
        if perm_plot_path is not None:
            run.log_artifact_path(perm_plot_path)
        if overlap_path is not None:
            run.log_artifact_path(overlap_path)
        if overlap_plot_path is not None:
            run.log_artifact_path(overlap_plot_path)
        run.log_artifact_path(decile_plot_path)
        run.log_artifact_path(calibration_path)

    log.info(
        "pipeline_done", model=model_cfg["name"], dataset=dataset_cfg["name"], qini=qini_ci.point
    )

    return TrainResult(
        model_name=model_cfg["name"],
        dataset_name=dataset_cfg["name"],
        seed=seed,
        qini=qini_ci.point,
        qini_ci_lower=qini_ci.lower,
        qini_ci_upper=qini_ci.upper,
        auuc=auuc_res.auuc_normalized,
        metrics=extra_metrics,
        artifacts_dir=output_dir,
    )
