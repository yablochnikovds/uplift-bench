"""Hydra structured configs.

Hydra supports two modes: free-form YAML, or dataclass-typed structured
configs registered with the ConfigStore. We use the latter because:

* mypy and IDE see real types — no more "is `seed` a str or int again?"
* Wrong fields in YAML fail fast with a clear OmegaConf error instead of
  silently being ignored.
* Defaults compose cleanly across the dataset/model/base_learner axes.

Each config below is what the corresponding YAML in `configs/...` validates
against. Adding a new field is a two-step change: dataclass + YAML.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class BaseLearnerConfig:
    name: str = "catboost"  # one of: catboost, lightgbm, logreg
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelConfig:
    name: str = "s_learner"  # registry key in models.factory
    extra_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetConfig:
    name: str = "hillstrom"  # one of: hillstrom, criteo, retailhero, megafon
    data_dir: str = "data/raw"
    # Per-loader knobs end up here. Loaders pull the keys they care about.
    loader_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class SplitConfig:
    train_frac: float = 0.7
    val_frac: float = 0.15
    seed: int = 42


@dataclass
class BootstrapConfig:
    n_boot: int = 1000
    method: str = "bca"  # bca | percentile
    alpha: float = 0.05
    n_jobs: int = 1
    seed: int = 0


@dataclass
class RobustnessConfig:
    enable_permutation: bool = True
    permutation_n_repeats: int = 5
    enable_feature_drop: bool = False
    enable_learning_curve: bool = False
    learning_curve_fractions: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)
    enable_overlap: bool = True


@dataclass
class TrackingConfig:
    enabled: bool = True
    experiment_name: str = "uplift-bench"
    tracking_uri: str = "file://./mlruns"
    run_name: str | None = None  # auto-derived from model+dataset if None


@dataclass
class ExperimentConfig:
    """Top-level Hydra config — what `hydra.main` validates against."""

    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    base_learner: BaseLearnerConfig = field(default_factory=BaseLearnerConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    bootstrap: BootstrapConfig = field(default_factory=BootstrapConfig)
    robustness: RobustnessConfig = field(default_factory=RobustnessConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    seed: int = 42
    output_dir: str = "outputs"
