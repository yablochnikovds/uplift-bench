"""End-to-end pipeline for one (model, dataset, seed) combination.

Public entry: `run_one(...)`. Internally split into focused stages:

    _prepare_data        load → validate → split → return train/test folds
    _fit_uplift_model    build model from cfg → fit
    _compute_metrics     Qini + AUUC + uplift@k + decile, with bootstrap CI
    _compute_artifacts   save plots + tables to `output_dir`
    _compute_robustness  permutation + overlap (when enabled)
    _log_run             emit MLflow run

Each stage is independently testable and the orchestration in `run_one`
reads top-to-bottom like a normal script.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from uplift_bench.data.factory import make_loader
from uplift_bench.data.splits import make_splits
from uplift_bench.data.validation import UpliftDataset
from uplift_bench.metrics._common import NDArray1D
from uplift_bench.metrics.auuc import auuc
from uplift_bench.metrics.bootstrap import BootstrapCI, bootstrap_ci
from uplift_bench.metrics.cumulative_gain import cumulative_gain_curve
from uplift_bench.metrics.decile import decile_table
from uplift_bench.metrics.policy_value import policy_value_curve
from uplift_bench.metrics.qini import qini_coefficient, qini_curve
from uplift_bench.metrics.uplift_at_k import uplift_at_k
from uplift_bench.models.base import UpliftModel
from uplift_bench.models.factory import make_model
from uplift_bench.models.utils import build_model_kwargs
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


@dataclass(frozen=True, slots=True)
class _RunCtx:
    """Bundle of every config slice + identity bits for one run.

    Lets the stage helpers (`_compute_artifacts`, `_log_run`) take a
    single `_RunCtx` arg instead of 7 separate kwargs each. Cheap to
    construct (just a snapshot of the dicts that Hydra already built).
    """

    dataset_cfg: dict[str, Any]
    model_cfg: dict[str, Any]
    base_learner_cfg: dict[str, Any]
    split_cfg: dict[str, Any]
    bootstrap_cfg: dict[str, Any]
    robustness_cfg: dict[str, Any]
    tracking_cfg: dict[str, Any]
    seed: int
    output_dir: Path

    @property
    def title_prefix(self) -> str:
        return f"{self.model_cfg['name']} on {self.dataset_cfg['name']}"

    @property
    def run_name(self) -> str:
        return (
            self.tracking_cfg.get("run_name")
            or f"{self.model_cfg['name']}.{self.dataset_cfg['name']}.seed{self.seed}"
        )


# ---------------------------------------------------------------------- #
# stage 1 — data
# ---------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class _DataFolds:
    dataset: UpliftDataset
    X_train: pd.DataFrame
    t_train: NDArray1D
    y_train: NDArray1D
    X_test: pd.DataFrame
    t_test: NDArray1D
    y_test: NDArray1D


def _prepare_data(
    dataset_cfg: dict[str, Any],
    split_cfg: dict[str, Any],
    seed: int,
) -> _DataFolds:
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
    return _DataFolds(
        dataset=dataset,
        X_train=X.iloc[splits.train],
        t_train=t[splits.train],
        y_train=y[splits.train],
        X_test=X.iloc[splits.test],
        t_test=t[splits.test],
        y_test=y[splits.test],
    )


# ---------------------------------------------------------------------- #
# stage 2 — model
# ---------------------------------------------------------------------- #


def _fit_uplift_model(
    folds: _DataFolds,
    model_cfg: dict[str, Any],
    base_learner_cfg: dict[str, Any],
    seed: int,
) -> UpliftModel:
    kwargs = build_model_kwargs(
        model_cfg["name"],
        base_learner_cfg,
        model_cfg.get("extra_params") or {},
        seed=seed,
    )
    model = make_model(model_cfg["name"], **kwargs)
    log.info("fitting_model", model=model_cfg["name"], n_train=len(folds.X_train))
    model.fit(folds.X_train, folds.t_train, folds.y_train)
    return model


# ---------------------------------------------------------------------- #
# stage 3 — metrics
# ---------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class _Metrics:
    qini_ci: BootstrapCI
    auuc_normalized: float
    extras: dict[str, float]


def _compute_metrics(
    test_preds: NDArray1D,
    folds: _DataFolds,
    bootstrap_cfg: dict[str, Any],
) -> _Metrics:
    qini_ci = bootstrap_ci(
        qini_coefficient,
        test_preds,
        folds.t_test,
        folds.y_test,
        n_boot=bootstrap_cfg["n_boot"],
        method=bootstrap_cfg["method"],
        alpha=bootstrap_cfg["alpha"],
        n_jobs=bootstrap_cfg["n_jobs"],
        seed=bootstrap_cfg["seed"],
    )
    auuc_res = auuc(test_preds, folds.t_test, folds.y_test)
    extras: dict[str, float] = {
        "auuc_raw": auuc_res.auuc_raw,
        "auuc_normalized": auuc_res.auuc_normalized,
    }
    for k_frac in UPLIFT_AT_K_FRACTIONS:
        extras[f"uplift_at_{int(k_frac * 100)}"] = uplift_at_k(
            test_preds,
            folds.t_test,
            folds.y_test,
            k=k_frac,
        )

    # Cumulative gain (Radcliffe 2007) — top-k responder rate.
    cg = cumulative_gain_curve(test_preds, folds.t_test, folds.y_test)
    extras["cumulative_gain_auc"] = cg.auc

    # Policy value at standard budget tiers (Manski 2004 / Athey & Wager 2021).
    pv = policy_value_curve(
        test_preds,
        folds.t_test,
        folds.y_test,
        budgets=[0.0, 0.1, 0.2, 0.3, 0.5, 1.0],
    )
    for b, v in zip(pv.budgets, pv.policy_values, strict=True):
        extras[f"policy_value_b{int(b * 100):02d}"] = float(v)
    return _Metrics(
        qini_ci=qini_ci,
        auuc_normalized=auuc_res.auuc_normalized,
        extras=extras,
    )


# ---------------------------------------------------------------------- #
# stage 4 — artifacts
# ---------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class _Artifacts:
    qini_path: Path
    dist_path: Path
    deciles_csv: Path
    deciles_png: Path
    calibration_png: Path
    config_json: Path


def _compute_artifacts(
    test_preds: NDArray1D,
    folds: _DataFolds,
    ctx: _RunCtx,
) -> _Artifacts:
    out = ctx.output_dir
    title = ctx.title_prefix

    deciles = decile_table(test_preds, folds.t_test, folds.y_test, n_buckets=10)
    qcurve = qini_curve(test_preds, folds.t_test, folds.y_test)

    qini_path = out / "qini_curve.png"
    plot_qini_curve(qcurve, title=title, save_path=qini_path)

    dist_path = out / "uplift_distribution.png"
    plot_uplift_distribution(test_preds, save_path=dist_path)

    deciles_csv = out / "deciles.csv"
    deciles.to_csv(deciles_csv, index=False)
    deciles_png = out / "deciles.png"
    plot_decile_uplift(
        deciles,
        title=f"{title} — per-decile uplift",
        save_path=deciles_png,
    )

    calibration_png = out / "calibration.png"
    plot_calibration(
        test_preds,
        folds.t_test,
        folds.y_test,
        title=f"{title} — calibration",
        save_path=calibration_png,
    )

    config_json = out / "config.json"
    dump_json(
        {
            "dataset": ctx.dataset_cfg,
            "model": ctx.model_cfg,
            "base_learner": ctx.base_learner_cfg,
            "split": ctx.split_cfg,
            "bootstrap": ctx.bootstrap_cfg,
            "robustness": ctx.robustness_cfg,
            "seed": ctx.seed,
        },
        config_json,
    )

    return _Artifacts(
        qini_path=qini_path,
        dist_path=dist_path,
        deciles_csv=deciles_csv,
        deciles_png=deciles_png,
        calibration_png=calibration_png,
        config_json=config_json,
    )


# ---------------------------------------------------------------------- #
# stage 5 — robustness
# ---------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class _RobustnessOutputs:
    perm_csv: Path | None
    perm_png: Path | None
    overlap_csv: Path | None
    overlap_png: Path | None
    extras: dict[str, float]


def _compute_robustness(
    model: UpliftModel,
    folds: _DataFolds,
    ctx: _RunCtx,
) -> _RobustnessOutputs:
    rc = ctx.robustness_cfg
    out = ctx.output_dir
    title = ctx.title_prefix
    extras: dict[str, float] = {}
    perm_csv: Path | None = None
    perm_png: Path | None = None
    if rc.get("enable_permutation"):
        perm = permutation_uplift_importance(
            model,
            folds.X_test,
            folds.t_test,
            folds.y_test,
            n_repeats=rc.get("permutation_n_repeats", 5),
            seed=ctx.seed,
        )
        perm_csv = out / "permutation_importance.csv"
        perm.to_csv(perm_csv, index=False)
        perm_png = out / "permutation_importance.png"
        plot_permutation_importance(
            perm,
            title=f"{title} — permutation importance",
            save_path=perm_png,
        )

    overlap_csv: Path | None = None
    overlap_png: Path | None = None
    if rc.get("enable_overlap"):
        diag = overlap_diagnostics(folds.X_test, folds.t_test, n_splits=3, seed=ctx.seed)
        extras["overlap_ess_ratio"] = diag.ess_ratio
        extras["overlap_pct_below_clip"] = diag.pct_below_clip
        extras["overlap_pct_above_clip"] = diag.pct_above_clip
        overlap_csv = out / "propensity.csv"
        pd.DataFrame({"propensity": diag.propensity}).to_csv(overlap_csv, index=False)
        overlap_png = out / "propensity.png"
        plot_propensity_histogram(
            diag.treated_propensity,
            diag.control_propensity,
            title=f"{title} — propensity overlap",
            save_path=overlap_png,
        )

    return _RobustnessOutputs(
        perm_csv=perm_csv,
        perm_png=perm_png,
        overlap_csv=overlap_csv,
        overlap_png=overlap_png,
        extras=extras,
    )


# ---------------------------------------------------------------------- #
# stage 6 — MLflow logging
# ---------------------------------------------------------------------- #


def _log_run(
    *,
    folds: _DataFolds,
    metrics: _Metrics,
    artifacts: _Artifacts,
    robustness: _RobustnessOutputs,
    ctx: _RunCtx,
) -> None:
    tc = ctx.tracking_cfg
    mc = ctx.model_cfg
    dc = ctx.dataset_cfg
    bc = ctx.base_learner_cfg
    with start_run(
        enabled=tc["enabled"],
        experiment_name=tc["experiment_name"],
        tracking_uri=tc["tracking_uri"],
        run_name=ctx.run_name,
        tags={
            "dataset": dc["name"],
            "model": mc["name"],
            "base_learner": bc["name"],
        },
    ) as run:
        run.log_params(
            {
                "model": mc["name"],
                "dataset": dc["name"],
                "seed": ctx.seed,
                "base_learner": bc["name"],
                "base_learner_params": bc.get("params", {}),
                "model_extra_params": mc.get("extra_params", {}),
                "split_train_frac": ctx.split_cfg["train_frac"],
                "split_val_frac": ctx.split_cfg["val_frac"],
                "bootstrap_n_boot": ctx.bootstrap_cfg["n_boot"],
                "bootstrap_method": ctx.bootstrap_cfg["method"],
                "dataset_source_hash": folds.dataset.source_hash,
                "n_train": len(folds.X_train),
                "n_test": len(folds.X_test),
            }
        )
        run.log_metric_with_ci("qini", metrics.qini_ci)
        run.log_metrics({**metrics.extras, **robustness.extras})

        for path in (
            artifacts.qini_path,
            artifacts.dist_path,
            artifacts.deciles_csv,
            artifacts.deciles_png,
            artifacts.calibration_png,
            artifacts.config_json,
        ):
            run.log_artifact_path(path)
        for opt in (
            robustness.perm_csv,
            robustness.perm_png,
            robustness.overlap_csv,
            robustness.overlap_png,
        ):
            if opt is not None:
                run.log_artifact_path(opt)


# ---------------------------------------------------------------------- #
# orchestrator
# ---------------------------------------------------------------------- #


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

    ctx = _RunCtx(
        dataset_cfg=dataset_cfg,
        model_cfg=model_cfg,
        base_learner_cfg=base_learner_cfg,
        split_cfg=split_cfg,
        bootstrap_cfg=bootstrap_cfg,
        robustness_cfg=robustness_cfg,
        tracking_cfg=tracking_cfg,
        seed=seed,
        output_dir=output_dir,
    )

    folds = _prepare_data(dataset_cfg, split_cfg, seed)
    model = _fit_uplift_model(folds, model_cfg, base_learner_cfg, seed)
    test_preds = model.predict_uplift(folds.X_test)

    metrics = _compute_metrics(test_preds, folds, bootstrap_cfg)
    artifacts = _compute_artifacts(test_preds, folds, ctx)
    robustness = _compute_robustness(model, folds, ctx)
    _log_run(folds=folds, metrics=metrics, artifacts=artifacts, robustness=robustness, ctx=ctx)

    log.info(
        "pipeline_done",
        model=model_cfg["name"],
        dataset=dataset_cfg["name"],
        qini=metrics.qini_ci.point,
    )

    return TrainResult(
        model_name=model_cfg["name"],
        dataset_name=dataset_cfg["name"],
        seed=seed,
        qini=metrics.qini_ci.point,
        qini_ci_lower=metrics.qini_ci.lower,
        qini_ci_upper=metrics.qini_ci.upper,
        auuc=metrics.auuc_normalized,
        metrics={**metrics.extras, **robustness.extras},
        artifacts_dir=output_dir,
    )
