"""Lenta uplift dataset (Russian grocery loyalty campaign).

Public RCT dataset published by Lenta and packaged into the
`scikit-uplift` library. ~687k rows, binary treatment (`group` =
"test"/"control"), binary outcome (`response_att` — store visit).

Dataset is hosted on a stable S3 bucket owned by the scikit-uplift
maintainers — same URL their `sklift.datasets.fetch_lenta` uses
internally, so this loader stays in lock-step with the upstream
package without depending on it at runtime.

Schema reference:
https://www.uplift-modeling.com/en/v0.5.1/api/datasets/fetch_lenta.html

Notes on the raw data
---------------------
* Treatment ratio is ~0.75 (more treated than control), which is normal
  for this campaign — the test group was the broader rollout.
* Outcome `response_att` is the rare-event signal: ~10% positive.
* Several columns have meaningful NaNs (e.g. `gender`); we median-impute
  numerics and one-hot the small categoricals.
* The CSV.gz unzips to ~567 MB so we cache as parquet on first load.
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

LENTA_URL: Final[str] = "https://sklift.s3.eu-west-2.amazonaws.com/lenta_dataset.csv.gz"
LENTA_FILENAME: Final[str] = "lenta_dataset.csv.gz"
LENTA_PARQUET: Final[str] = "lenta_dataset.parquet"

# A handful of the most informative numeric features — enough to make the
# benchmark meaningful while keeping memory reasonable on a laptop.
# Full list reaches ~190 columns; restricting upfront avoids dragging
# every customer-id-like high-cardinality column into the model.
_NUMERIC_FEATURES: Final[tuple[str, ...]] = (
    "age",
    "main_format",
    "cheque_count_3m_g42",
    "cheque_count_6m_g25",
    "sale_count_3m_g32",
    "sale_count_6m_g25",
    "sale_sum_3m_g42",
    "sale_sum_6m_g25",
    "k_var_count_per_cheq_15d_g28",
    "food_share_15d",
    "response_sms",
    "response_viber",
    "months_from_register",
    "months_to_response",
)
_CATEGORICAL_FEATURES: Final[tuple[str, ...]] = ("gender",)
_GENDER_VALUES: Final[tuple[str, ...]] = ("Ж", "М", "U")  # Russian text in source


class LentaLoader(DatasetLoader):
    """Loads Lenta from a fixed-seed cached parquet, downloading once."""

    name = "lenta"

    @property
    def schema(self) -> DatasetSchema:
        feature_cols = list(_NUMERIC_FEATURES) + [f"gender_{g}" for g in _GENDER_VALUES]
        return DatasetSchema(
            treatment_col="treatment",
            outcome_col="outcome",
            feature_cols=tuple(feature_cols),
        )

    def _raw_path(self) -> Path:
        return self.data_dir / "lenta" / LENTA_FILENAME

    def _parquet_path(self) -> Path:
        return self.data_dir / "lenta" / LENTA_PARQUET

    def download(self) -> Path:
        return download_file(LENTA_URL, self._raw_path())

    def _read(self, path: Path) -> pd.DataFrame:
        parquet = self._parquet_path()
        if parquet.exists():
            log.info("lenta_using_cached_parquet", path=str(parquet))
            df = pd.read_parquet(parquet)
        else:
            log.info("lenta_parsing_csv", path=str(path))
            df = pd.read_csv(path, compression="gzip", low_memory=False)
            log.info("lenta_caching_parquet", path=str(parquet))
            write_parquet_atomic(df, parquet)

        # Normalise treatment + outcome into our schema column names.
        df["treatment"] = (df["group"].astype(str) == "test").astype("int8")
        df["outcome"] = df["response_att"].fillna(0).astype("int8")

        # Numeric features: median-impute (real Lenta has missing on most
        # `*_count_*` columns for low-engagement customers).
        feats = pd.DataFrame(index=df.index)
        for col in _NUMERIC_FEATURES:
            if col in df.columns:
                feats[col] = df[col].fillna(df[col].median()).astype(np.float64)
            else:
                feats[col] = 0.0

        # Gender: normalise to {Ж, М, U}; missing → U.
        gender = df.get("gender", pd.Series(["U"] * len(df))).fillna("U").astype(str)
        gender = gender.where(gender.isin(_GENDER_VALUES), "U")
        gender_dummies = pd.get_dummies(gender, prefix="gender", dtype="int8")
        for g in _GENDER_VALUES:
            col = f"gender_{g}"
            if col not in gender_dummies.columns:
                gender_dummies[col] = pd.Series(0, dtype="int8", index=df.index)

        out = pd.concat(
            [
                feats.reset_index(drop=True),
                gender_dummies[[f"gender_{g}" for g in _GENDER_VALUES]].reset_index(drop=True),
                df[["treatment", "outcome"]].reset_index(drop=True),
            ],
            axis=1,
        )
        return out
