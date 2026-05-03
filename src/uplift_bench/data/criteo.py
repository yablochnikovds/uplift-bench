"""Criteo Uplift v2 dataset.

The original dataset is hosted by Criteo Research:
    https://ailab.criteo.com/criteo-uplift-prediction-dataset/

The 12-feature CSV is ~300 MB compressed (~1.5 GB uncompressed) and has
roughly 13.9M rows. We materialise the validated DataFrame as parquet on
first load — pandas takes 90+ seconds to re-parse the CSV, parquet does
it in 4.

For local-machine sanity we also support `subsample`: a fixed-seed random
sample of the full set. The bench reports both "full" and "subsample" runs
in `results/` and labels them clearly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from uplift_bench.data.base import DatasetLoader
from uplift_bench.data.download import download_file
from uplift_bench.data.validation import DatasetSchema
from uplift_bench.utils.io import write_parquet_atomic
from uplift_bench.utils.logging import get_logger

log = get_logger(__name__)

# Criteo's own S3 bucket has been flaky; the HF mirror under criteo's
# official org account is the de-facto source as of 2026-05-03. ~297 MB.
CRITEO_URL: Final[str] = (
    "https://huggingface.co/datasets/criteo/criteo-uplift/resolve/main/"
    "criteo-research-uplift-v2.1.csv.gz"
)
CRITEO_FILENAME: Final[str] = "criteo-research-uplift-v2.1.csv.gz"
CRITEO_PARQUET: Final[str] = "criteo-research-uplift-v2.1.parquet"

_FEATURES: Final[tuple[str, ...]] = tuple(f"f{i}" for i in range(12))


class CriteoLoader(DatasetLoader):
    name = "criteo"

    def __init__(
        self,
        data_dir: Path,
        outcome: str = "visit",
        subsample: int | None = None,
        subsample_seed: int = 42,
    ) -> None:
        super().__init__(data_dir)
        if outcome not in {"visit", "conversion"}:
            raise ValueError(f"outcome must be 'visit' or 'conversion', got {outcome!r}")
        if subsample is not None and subsample <= 0:
            raise ValueError(f"subsample must be a positive int or None, got {subsample}")
        self.outcome = outcome
        self.subsample = subsample
        self.subsample_seed = subsample_seed

    @property
    def schema(self) -> DatasetSchema:
        return DatasetSchema(
            treatment_col="treatment",
            outcome_col="outcome",
            feature_cols=_FEATURES,
        )

    def _raw_path(self) -> Path:
        return self.data_dir / "criteo" / CRITEO_FILENAME

    def _parquet_path(self) -> Path:
        return self.data_dir / "criteo" / CRITEO_PARQUET

    def download(self) -> Path:
        return download_file(CRITEO_URL, self._raw_path())

    def _read(self, path: Path) -> pd.DataFrame:
        parquet = self._parquet_path()
        if parquet.exists():
            log.info("criteo_using_cached_parquet", path=str(parquet))
            df = pd.read_parquet(parquet)
        else:
            log.info("criteo_parsing_csv", path=str(path))
            # Explicit dtypes save ~6 GB of RAM vs default object inference.
            # Mapping[Hashable, str] is the type pandas wants — using a plain
            # dict[str, str] trips mypy invariance.
            from collections.abc import Hashable, Mapping
            dtypes: Mapping[Hashable, str] = {
                **dict.fromkeys(_FEATURES, "float32"),
                "treatment": "int8", "visit": "int8",
                "conversion": "int8", "exposure": "int8",
            }
            df = pd.read_csv(path, dtype=dtypes)
            log.info("criteo_caching_parquet", path=str(parquet))
            write_parquet_atomic(df, parquet)

        df["outcome"] = df[self.outcome].astype("int8")
        keep = list(_FEATURES) + ["treatment", "outcome"]
        df = df[keep]

        if self.subsample is not None and self.subsample < len(df):
            rng = np.random.default_rng(self.subsample_seed)
            idx = rng.choice(len(df), size=self.subsample, replace=False)
            idx.sort()  # keep file-order locality for cache friendliness
            df = df.iloc[idx].reset_index(drop=True)
            log.info("criteo_subsampled", n=len(df), seed=self.subsample_seed)

        return df
