"""Hydra entry point for the benchmark.

Wraps `pipelines.train.run_one` so it can be driven by `hydra.main`.
Multirun (`-m model=...,...`) is what gives you the matrix.

We resolve the configs directory at module import time using `__file__`
rather than letting Hydra interpret a relative path. Reason: the package
is also reachable when installed (e.g. inside a Docker image) where the
.py file lives in site-packages and the `../../../configs` shortcut would
escape the install root. Walking up from `__file__` to the repo root is
robust to both cases — we copy `configs/` next to the package on install
via the hatchling include rule below.
"""

from __future__ import annotations

import os
from pathlib import Path

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from uplift_bench.config.schemas import ExperimentConfig
from uplift_bench.pipelines.train import run_one
from uplift_bench.utils.logging import configure, get_logger

cs = ConfigStore.instance()
cs.store(name="base_experiment", node=ExperimentConfig)

log = get_logger(__name__)


def _find_configs_dir() -> Path:
    """Locate the configs/ directory, working both from source and installed.

    Search order:
      1. Env override $UPLIFT_BENCH_CONFIGS — escape hatch for unusual layouts.
      2. <package>/../../../configs (source-tree layout, what we ship from).
      3. <cwd>/configs — when the user is running from a clone.
    """
    if env := os.environ.get("UPLIFT_BENCH_CONFIGS"):
        return Path(env).resolve()
    here = Path(__file__).resolve()
    candidates = [
        here.parents[3] / "configs",  # repo_root/configs from src/uplift_bench/pipelines
        Path.cwd() / "configs",
    ]
    for c in candidates:
        if c.is_dir():
            return c
    raise FileNotFoundError(
        "configs/ directory not found. Set UPLIFT_BENCH_CONFIGS to point at it."
    )


_CONFIG_DIR = str(_find_configs_dir())


@hydra.main(config_path=_CONFIG_DIR, config_name="config", version_base=None)
def hydra_entry(cfg: DictConfig) -> None:
    """Single-config Hydra entry. Multirun composes this across the model axis."""
    configure(level="INFO")

    output_dir = (
        Path(cfg.output_dir).resolve() / f"{cfg.model.name}_{cfg.dataset.name}_seed{cfg.seed}"
    )
    log.info(
        "experiment_started",
        dataset=cfg.dataset.name,
        model=cfg.model.name,
        base_learner=cfg.base_learner.name,
        seed=cfg.seed,
    )

    result = run_one(
        dataset_cfg=OmegaConf.to_container(cfg.dataset, resolve=True),  # type: ignore[arg-type]
        model_cfg=OmegaConf.to_container(cfg.model, resolve=True),  # type: ignore[arg-type]
        base_learner_cfg=OmegaConf.to_container(cfg.base_learner, resolve=True),  # type: ignore[arg-type]
        split_cfg=OmegaConf.to_container(cfg.split, resolve=True),  # type: ignore[arg-type]
        bootstrap_cfg=OmegaConf.to_container(cfg.bootstrap, resolve=True),  # type: ignore[arg-type]
        robustness_cfg=OmegaConf.to_container(cfg.robustness, resolve=True),  # type: ignore[arg-type]
        tracking_cfg=OmegaConf.to_container(cfg.tracking, resolve=True),  # type: ignore[arg-type]
        seed=cfg.seed,
        output_dir=output_dir,
    )
    log.info("experiment_done", qini=result.qini, output=str(output_dir))
