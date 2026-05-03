"""Abstract dataset loader.

A `DatasetLoader` is responsible for one job: take a path on disk and produce
a validated `UpliftDataset`. Anything fancier (downloading, caching, schema
inference) belongs in subclasses or utility modules — keep the contract
small so swapping a dataset is a one-class change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from uplift_bench.data.validation import DatasetSchema, UpliftDataset, validate_dataframe
from uplift_bench.utils.io import file_sha256
from uplift_bench.utils.logging import get_logger

log = get_logger(__name__)


class DatasetLoader(ABC):
    """Loader contract.

    Subclasses implement `_raw_path`, `_read`, and `schema`. The base class
    handles the common bits — file existence, hashing, validation.
    """

    name: str  # short slug, e.g. "hillstrom"

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = Path(data_dir)

    # ------------------------------------------------------------------ #
    # subclass contract
    # ------------------------------------------------------------------ #

    @property
    @abstractmethod
    def schema(self) -> DatasetSchema: ...

    @abstractmethod
    def _raw_path(self) -> Path:
        """Where the source file lives under `data_dir`."""

    @abstractmethod
    def _read(self, path: Path) -> pd.DataFrame:
        """Read the raw file → DataFrame with the schema's columns present."""

    # Optional override: subclasses that can self-download set this.
    def download(self) -> Path:
        """Fetch the source file; default is no-op (assumes file already there)."""
        path = self._raw_path()
        if not path.exists():
            raise FileNotFoundError(
                f"{self.name}: source file not found at {path}. "
                f"This loader does not support automatic download — see "
                f"docs/datasets.md for instructions."
            )
        return path

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #

    def load(self) -> UpliftDataset:
        """Resolve, read, validate."""
        path = self.download()  # self-download or check existence
        log.info("loading_dataset", name=self.name, path=str(path))
        df = self._read(path)
        df = validate_dataframe(df, self.schema)
        sha = file_sha256(path)
        return UpliftDataset(
            df=df,
            schema=self.schema,
            name=self.name,
            source_hash=sha,
            metadata={"path": str(path)},
        )
