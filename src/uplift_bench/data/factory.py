"""Build a DatasetLoader from a name + kwargs.

Mirrors uplift_bench.models.factory. Used by the Hydra entry point.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from uplift_bench.data.base import DatasetLoader
from uplift_bench.data.criteo import CriteoLoader
from uplift_bench.data.hillstrom import HillstromLoader
from uplift_bench.data.megafon import MegaFonLoader
from uplift_bench.data.retailhero import RetailHeroLoader

DATASET_REGISTRY: dict[str, type[DatasetLoader]] = {
    "hillstrom": HillstromLoader,
    "criteo": CriteoLoader,
    "retailhero": RetailHeroLoader,
    "megafon": MegaFonLoader,
}


def make_loader(name: str, data_dir: str | Path, **kwargs: Any) -> DatasetLoader:
    if name not in DATASET_REGISTRY:
        raise ValueError(
            f"unknown dataset {name!r}; choose from {sorted(DATASET_REGISTRY)}"
        )
    return DATASET_REGISTRY[name](data_dir=Path(data_dir), **kwargs)
