"""Loader-shim wrapping the synthetic DGP.

Lets you say `make_loader("synthetic", n_samples=4000, seed=0, ...)`
and get a regular `UpliftDataset` you can feed into the same pipeline as
Hillstrom / Criteo / Lenta. The known true tau and propensity are *not*
exposed via this loader — it intentionally pretends to be just another
real dataset so meta-learners are evaluated against observed Y only.

For tests that need ground-truth tau (correlation tests, oracle Qini,
etc.), import `make_uplift_dataset` directly from
`uplift_bench.data.synthetic`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from uplift_bench.data.base import DatasetLoader
from uplift_bench.data.synthetic import OutcomeKind, make_uplift_dataset
from uplift_bench.data.validation import DatasetSchema, UpliftDataset


class SyntheticLoader(DatasetLoader):
    """A `DatasetLoader` that materialises a fresh synthetic DGP each call.

    Doesn't touch the filesystem — the DGP is reproducible from `seed`.
    `data_dir` is accepted only to satisfy the `DatasetLoader` contract.
    """

    name = "synthetic"

    def __init__(
        self,
        data_dir: Path | str = "data",
        n_samples: int = 4000,
        n_features: int = 10,
        n_informative_uplift: int = 3,
        treatment_share: float = 0.5,
        propensity_drift: float = 0.0,
        noise: float = 1.0,
        outcome: OutcomeKind = "binary",
        seed: int = 0,
    ) -> None:
        super().__init__(data_dir if isinstance(data_dir, Path) else Path(data_dir))
        self._n_samples = n_samples
        self._n_features = n_features
        self._n_informative_uplift = n_informative_uplift
        self._treatment_share = treatment_share
        self._propensity_drift = propensity_drift
        self._noise = noise
        self._outcome: OutcomeKind = outcome
        self._seed = seed
        self._params: dict[str, Any] = {
            "n_samples": n_samples,
            "n_features": n_features,
            "n_informative_uplift": n_informative_uplift,
            "treatment_share": treatment_share,
            "propensity_drift": propensity_drift,
            "noise": noise,
            "outcome": outcome,
            "seed": seed,
        }

    @property
    def schema(self) -> DatasetSchema:
        return DatasetSchema(
            treatment_col="treatment",
            outcome_col="outcome",
            feature_cols=tuple(f"f{i}" for i in range(self._n_features)),
            allowed_outcome_values=(None if self._outcome == "continuous" else (0, 1)),
        )

    def _raw_path(self) -> Path:
        # No file is ever read; we still need to satisfy the abstract
        # method. Path is purely informational.
        return self.data_dir / "synthetic" / f"seed_{self._seed}.virtual"

    def download(self) -> Path:
        # Nothing to download — DGP is deterministic from `seed`.
        return self._raw_path()

    def _read(self, _path: Path) -> pd.DataFrame:
        synth = make_uplift_dataset(
            n_samples=self._n_samples,
            n_features=self._n_features,
            n_informative_uplift=self._n_informative_uplift,
            treatment_share=self._treatment_share,
            propensity_drift=self._propensity_drift,
            noise=self._noise,
            outcome=self._outcome,
            seed=self._seed,
        )
        return synth.df

    def load(self) -> UpliftDataset:
        # Override `load` to skip the file-hash step (would crash since the
        # virtual path doesn't exist), but still produce a validated dataset.
        from hashlib import sha256  # noqa: PLC0415

        from uplift_bench.data.validation import validate_dataframe  # noqa: PLC0415

        df = validate_dataframe(self._read(self._raw_path()), self.schema)
        # Stable param-derived fingerprint so MLflow can dedupe configs.
        param_str = ",".join(f"{k}={v}" for k, v in sorted(self._params.items()))
        fingerprint = sha256(param_str.encode()).hexdigest()
        return UpliftDataset(
            df=df,
            schema=self.schema,
            name=self.name,
            source_hash=fingerprint,
            metadata={"params": param_str},
        )


def make_synthetic_loader(**kwargs: Any) -> SyntheticLoader:
    """Tiny helper for callers that don't go through `make_loader`."""
    return SyntheticLoader(**kwargs)
