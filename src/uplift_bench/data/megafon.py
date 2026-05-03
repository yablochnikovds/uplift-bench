"""MegaFon Uplift Competition dataset.

Source: https://ods.ai/competitions/megafon-uplift-competition/data
(also requires Ods.ai login). Same shape rationale as RetailHero — manual
placement, sample stand-in for tests.

The dataset has ~600k rows, ~50 anonymised numeric features, binary
treatment_group, and a binary conversion target.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pandas as pd

from uplift_bench.data.base import DatasetLoader
from uplift_bench.data.validation import DatasetSchema
from uplift_bench.utils.logging import get_logger

log = get_logger(__name__)

# The competition publishes 50 anonymised features. Some snapshots have
# fewer; the loader normalises to whatever is on disk and exposes that
# set as `feature_cols`.
_FEATURE_PREFIX: Final[str] = "X_"


class MegaFonLoader(DatasetLoader):
    name = "megafon"

    def __init__(self, data_dir: Path) -> None:
        super().__init__(data_dir)
        self._cached_features: tuple[str, ...] | None = None

    @property
    def schema(self) -> DatasetSchema:
        # `_cached_features` is populated by `_read` on first load. Before
        # that we return the empty tuple — callers should always go through
        # `load()`, not `schema` directly.
        return DatasetSchema(
            treatment_col="treatment",
            outcome_col="outcome",
            feature_cols=self._cached_features or (),
        )

    def _raw_path(self) -> Path:
        return self.data_dir / "megafon" / "train.csv"

    def download(self) -> Path:
        path = self._raw_path()
        if not path.exists():
            raise FileNotFoundError(
                "MegaFon requires manual download (login-walled). Place\n"
                f"  {path}\nfrom "
                "https://ods.ai/competitions/megafon-uplift-competition/data\n"
                "See docs/datasets.md for the exact instructions."
            )
        return path

    def _read(self, path: Path) -> pd.DataFrame:
        log.info("megafon_reading", path=str(path))
        df = pd.read_csv(path)

        # Contest column names vary by snapshot; normalise both seen variants.
        rename = {}
        if "treatment_group" in df.columns:
            rename["treatment_group"] = "treatment"
        if "conversion" in df.columns and "outcome" not in df.columns:
            rename["conversion"] = "outcome"
        df = df.rename(columns=rename)

        # Treatment in the original is "treatment"/"control" string — map.
        if df["treatment"].dtype == object:
            df["treatment"] = (df["treatment"].astype(str) == "treatment").astype("int8")

        feature_cols = tuple(c for c in df.columns if c.startswith(_FEATURE_PREFIX))
        if not feature_cols:
            raise ValueError(
                f"no feature columns found in {path}: expected columns prefixed "
                f"with {_FEATURE_PREFIX!r}"
            )
        self._cached_features = feature_cols

        keep = list(feature_cols) + ["treatment", "outcome"]
        return df[keep].copy()
