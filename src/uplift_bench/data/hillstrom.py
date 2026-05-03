"""Hillstrom (MineThatData) email-marketing dataset.

Kevin Hillstrom's classic 64k-row dataset. Three treatment arms:
"No E-Mail", "Mens E-Mail", "Womens E-Mail". Outcomes: visit, conversion,
spend (we use `visit` because it has decent base rate; conversion is
extremely rare and the bench would need a much bigger n).

We binarize treatment to "Womens E-Mail" vs "No E-Mail" by default — that
contrast has the strongest measured ATE in published analyses, which makes
it a useful sanity-check setup. The other contrasts are reachable via
`treatment_arm`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pandas as pd

from uplift_bench.data.base import DatasetLoader
from uplift_bench.data.download import download_file
from uplift_bench.data.validation import DatasetSchema

# The MineThatData blog post 404s, but Apache on minethatdata.com still
# serves the raw CSV at the original URL (verified 2026-05-03). HTTP-only,
# so we keep an HTTPS fallback for users behind corporate proxies that
# block plain http.
HILLSTROM_URL: Final[str] = (
    "http://www.minethatdata.com/"
    "Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
)
HILLSTROM_URL_FALLBACK: Final[str] = (
    "https://raw.githubusercontent.com/W-Tran/uplift-modelling/master/data/hillstrom/"
    "Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
)
HILLSTROM_FILENAME: Final[str] = "hillstrom.csv"


class HillstromLoader(DatasetLoader):
    name = "hillstrom"

    # Categorical features as listed in the original dataset.
    _CATEGORICAL: Final[tuple[str, ...]] = ("zip_code", "channel", "history_segment")
    _NUMERIC: Final[tuple[str, ...]] = (
        "recency", "history", "mens", "womens", "newbie",
    )

    def __init__(
        self,
        data_dir: Path,
        treatment_arm: str = "Womens E-Mail",
        outcome: str = "visit",
    ) -> None:
        super().__init__(data_dir)
        if treatment_arm not in {"Mens E-Mail", "Womens E-Mail"}:
            raise ValueError(
                f"treatment_arm must be one of {{'Mens E-Mail','Womens E-Mail'}}, "
                f"got {treatment_arm!r}"
            )
        if outcome not in {"visit", "conversion"}:
            raise ValueError(f"outcome must be 'visit' or 'conversion', got {outcome!r}")
        self.treatment_arm = treatment_arm
        self.outcome = outcome

    @property
    def schema(self) -> DatasetSchema:
        # After one-hotting, we end up with these feature columns.
        # Categories are small so dummy explosion is fine.
        feature_cols = list(self._NUMERIC) + [
            f"zip_code_{z}" for z in ("Rural", "Surburban", "Urban")
        ] + [
            f"channel_{c}" for c in ("Multichannel", "Phone", "Web")
        ] + [
            f"history_segment_{s}" for s in (
                "1) $0 - $100", "2) $100 - $200", "3) $200 - $350",
                "4) $350 - $500", "5) $500 - $750", "6) $750 - $1,000",
                "7) $1,000 +",
            )
        ]
        return DatasetSchema(
            treatment_col="treatment",
            outcome_col="outcome",
            feature_cols=tuple(feature_cols),
        )

    def _raw_path(self) -> Path:
        return self.data_dir / "hillstrom" / HILLSTROM_FILENAME

    def download(self) -> Path:
        path = self._raw_path()
        if path.exists():
            return path
        try:
            return download_file(HILLSTROM_URL, path)
        except Exception:  # noqa: BLE001 — falling back to HTTPS mirror is correct
            return download_file(HILLSTROM_URL_FALLBACK, path)

    def _read(self, path: Path) -> pd.DataFrame:
        raw = pd.read_csv(path)
        # Filter to the binary contrast we care about.
        keep_segments = {self.treatment_arm, "No E-Mail"}
        raw = raw[raw["segment"].isin(keep_segments)].copy()

        # Encode treatment.
        raw["treatment"] = (raw["segment"] == self.treatment_arm).astype("int8")

        # Encode outcome. `visit` is already 0/1 in the file; `conversion` too.
        raw["outcome"] = raw[self.outcome].astype("int8")

        # One-hot the categoricals. We use get_dummies (not OHE) because the
        # cardinality is tiny and we want stable column names across runs.
        cat = pd.get_dummies(raw[list(self._CATEGORICAL)], dtype="int8")

        out = pd.concat([raw[list(self._NUMERIC)].reset_index(drop=True),
                         cat.reset_index(drop=True),
                         raw[["treatment", "outcome"]].reset_index(drop=True)], axis=1)

        # The schema lists every dummy we expect; if a category is missing
        # from the file (rare but possible after subsetting), backfill with 0
        # so downstream models always see a stable column set.
        for col in self.schema.feature_cols:
            if col not in out.columns:
                out[col] = pd.Series(0, dtype="int8", index=out.index)

        # Keep only the columns the schema declares — no surprises.
        keep = list(self.schema.feature_cols) + ["treatment", "outcome"]
        return out[keep]
