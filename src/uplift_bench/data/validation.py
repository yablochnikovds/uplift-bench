"""Schema-level validation for uplift datasets.

We validate at the boundary between disk and the rest of the pipeline:
once a `UpliftDataset` is in memory we trust its invariants. This avoids
re-checking inside every metric and model.

The pydantic schema is intentionally narrow — it catches the dumb mistakes
(wrong dtype, missing column, T outside {0, 1}) before they corrupt downstream
computations. Distributional checks (overlap, class balance) live under
robustness/ where they belong.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, model_validator

# Shape/dtype-agnostic 1-D ndarray. mypy --strict insists on type args; we
# don't carry shape info around because numpy ops routinely change it.
NDArray1D = np.ndarray[Any, np.dtype[Any]]


class DatasetSchema(BaseModel):
    """Declarative schema for a binary-treatment uplift dataset.

    Used by every loader (real and synthetic) to declare which columns are
    treatment / outcome / features. Loaders pass the raw DataFrame through
    `validate_dataframe` and get back a normalized one.
    """

    treatment_col: str = Field(min_length=1)
    outcome_col: str = Field(min_length=1)
    feature_cols: tuple[str, ...]
    # Some real datasets (Hillstrom) ship multiple treatment arms — we keep
    # only one vs control for the binary uplift setting. Loaders set this
    # tuple; if non-empty we assert it.
    allowed_treatment_values: tuple[int, ...] = (0, 1)
    allowed_outcome_values: tuple[int, ...] | None = (0, 1)  # None → continuous outcome

    @model_validator(mode="after")
    def _no_overlap(self) -> "DatasetSchema":
        special = {self.treatment_col, self.outcome_col}
        overlap = special & set(self.feature_cols)
        if overlap:
            raise ValueError(f"feature_cols overlaps with treatment/outcome: {overlap}")
        return self


@dataclass(frozen=True, slots=True)
class UpliftDataset:
    """Validated, ready-to-use dataset.

    `df` keeps the original columns; `X`, `t`, `y` are convenience views.
    We avoid copying because for Criteo (~10M rows) the memory matters.
    """

    df: pd.DataFrame
    schema: DatasetSchema
    name: str
    source_hash: str = ""  # SHA-256 of source file when loaded from disk
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def X(self) -> pd.DataFrame:
        return self.df[list(self.schema.feature_cols)]

    @property
    def t(self) -> NDArray1D:
        return self.df[self.schema.treatment_col].to_numpy()

    @property
    def y(self) -> NDArray1D:
        return self.df[self.schema.outcome_col].to_numpy()

    @property
    def n(self) -> int:
        return len(self.df)


def validate_dataframe(df: pd.DataFrame, schema: DatasetSchema) -> pd.DataFrame:
    """Check the DataFrame matches the schema; coerce where safe.

    Coercions performed:
      - treatment column is cast to int8 (memory; downstream code assumes int).
      - outcome column is cast to int8 if `allowed_outcome_values` is set,
        otherwise float64.

    Failures (raised as ValueError):
      - missing required column,
      - treatment values outside `allowed_treatment_values`,
      - outcome values outside `allowed_outcome_values` (when set),
      - any null in treatment, outcome, or feature columns.
    """
    required = {schema.treatment_col, schema.outcome_col, *schema.feature_cols}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")

    # Null check first so downstream casts don't blow up with confusing errors.
    nulls = df[list(required)].isna().sum()
    nulls = nulls[nulls > 0]
    if not nulls.empty:
        raise ValueError(f"nulls in required columns: {nulls.to_dict()}")

    df = df.copy()

    t_raw = df[schema.treatment_col]
    bad_t = set(t_raw.unique()) - set(schema.allowed_treatment_values)
    if bad_t:
        raise ValueError(
            f"treatment column {schema.treatment_col!r} has values "
            f"{sorted(bad_t)} outside allowed {schema.allowed_treatment_values}"
        )
    df[schema.treatment_col] = t_raw.astype(np.int8)

    if schema.allowed_outcome_values is not None:
        y_raw = df[schema.outcome_col]
        bad_y = set(y_raw.unique()) - set(schema.allowed_outcome_values)
        if bad_y:
            raise ValueError(
                f"outcome column {schema.outcome_col!r} has values "
                f"{sorted(bad_y)} outside allowed {schema.allowed_outcome_values}"
            )
        df[schema.outcome_col] = y_raw.astype(np.int8)
    else:
        df[schema.outcome_col] = df[schema.outcome_col].astype(np.float64)

    return df
