"""X5 RetailHero Uplift Modeling Contest dataset.

The data is hosted on Ods.ai behind a free account login:
    https://ods.ai/competitions/x5-retailhero-uplift-modeling/data

That makes a programmatic download impossible without scraping a session
cookie, which we won't do. The loader expects two files placed by the user:

    {data_dir}/retailhero/uplift_train.csv
    {data_dir}/retailhero/clients.csv

`uplift_train.csv` has columns: client_id, treatment_flg, target.
`clients.csv` carries the customer features. We join, drop the id column,
one-hot the small categoricals, and use median-imputation for missing
numeric features (RetailHero ships some genuinely-missing values).

For tests and the smoke-config a tiny synthetic stand-in lives at
`data/sample/retailhero/`. It has the same schema but only a few thousand
rows so CI runs in seconds.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from uplift_bench.data.base import DatasetLoader
from uplift_bench.data.validation import DatasetSchema
from uplift_bench.utils.logging import get_logger

log = get_logger(__name__)

_NUMERIC_FEATURES: Final[tuple[str, ...]] = (
    "age", "average_amount", "purchase_sum_3m", "purchase_count_3m",
    "days_since_first", "days_since_last",
)
_CATEGORICAL_FEATURES: Final[tuple[str, ...]] = ("gender",)
_GENDER_VALUES: Final[tuple[str, ...]] = ("F", "M", "U")  # U = unknown


class RetailHeroLoader(DatasetLoader):
    name = "retailhero"

    @property
    def schema(self) -> DatasetSchema:
        feature_cols = list(_NUMERIC_FEATURES) + [f"gender_{g}" for g in _GENDER_VALUES]
        return DatasetSchema(
            treatment_col="treatment",
            outcome_col="outcome",
            feature_cols=tuple(feature_cols),
        )

    def _raw_path(self) -> Path:
        # Symbolic — the loader actually reads two files, returned by
        # `_files()`. Base class's hash uses this path; we hash the train
        # file because it's the one tied to the contest split.
        return self.data_dir / "retailhero" / "uplift_train.csv"

    def _files(self) -> tuple[Path, Path]:
        base = self.data_dir / "retailhero"
        return base / "uplift_train.csv", base / "clients.csv"

    def download(self) -> Path:
        train, clients = self._files()
        if not (train.exists() and clients.exists()):
            raise FileNotFoundError(
                "RetailHero requires manual download (login-walled). Place\n"
                f"  {train}\nand\n  {clients}\nfrom "
                "https://ods.ai/competitions/x5-retailhero-uplift-modeling/data\n"
                "See docs/datasets.md for the exact instructions."
            )
        return train

    def _read(self, path: Path) -> pd.DataFrame:
        train_path, clients_path = self._files()

        log.info("retailhero_reading", train=str(train_path), clients=str(clients_path))
        train = pd.read_csv(train_path)
        clients = pd.read_csv(clients_path)

        # Standard contest column names. We rename to our schema.
        train = train.rename(columns={"treatment_flg": "treatment", "target": "outcome"})
        df = train.merge(clients, on="client_id", how="left")

        # Median-impute the numeric block in one pass — RetailHero genuinely
        # has missing ages and purchase histories.
        for col in _NUMERIC_FEATURES:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())
            else:
                # Some snapshots don't ship `days_since_*` — fill with 0 and
                # let the model learn from the rest. Better than crashing.
                df[col] = 0.0

        df["gender"] = df.get("gender", pd.Series(["U"] * len(df))).fillna("U")
        df.loc[~df["gender"].isin(_GENDER_VALUES), "gender"] = "U"

        gender_dummies = pd.get_dummies(df["gender"], prefix="gender", dtype="int8")
        for g in _GENDER_VALUES:
            col = f"gender_{g}"
            if col not in gender_dummies.columns:
                gender_dummies[col] = pd.Series(0, dtype="int8", index=df.index)

        out = pd.concat([
            df[list(_NUMERIC_FEATURES)].astype(np.float64).reset_index(drop=True),
            gender_dummies[[f"gender_{g}" for g in _GENDER_VALUES]].reset_index(drop=True),
            df[["treatment", "outcome"]].astype("int8").reset_index(drop=True),
        ], axis=1)

        return out
