from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from uplift_bench.data.validation import (
    DatasetSchema,
    UpliftDataset,
    validate_dataframe,
)


@pytest.fixture
def schema() -> DatasetSchema:
    return DatasetSchema(
        treatment_col="t",
        outcome_col="y",
        feature_cols=("a", "b"),
    )


def _good_df() -> pd.DataFrame:
    return pd.DataFrame({
        "a": [0.1, 0.2, 0.3, 0.4],
        "b": [1.0, 2.0, 3.0, 4.0],
        "t": [0, 1, 0, 1],
        "y": [0, 1, 1, 0],
    })


def test_validate_happy_path(schema: DatasetSchema) -> None:
    out = validate_dataframe(_good_df(), schema)
    assert out["t"].dtype == np.int8
    assert out["y"].dtype == np.int8


def test_validate_continuous_outcome() -> None:
    schema = DatasetSchema(
        treatment_col="t", outcome_col="y", feature_cols=("a",),
        allowed_outcome_values=None,
    )
    df = pd.DataFrame({"a": [0.1, 0.2], "t": [0, 1], "y": [3.14, 2.7]})
    out = validate_dataframe(df, schema)
    assert out["y"].dtype == np.float64


def test_missing_column_raises(schema: DatasetSchema) -> None:
    df = _good_df().drop(columns=["b"])
    with pytest.raises(ValueError, match="missing columns.*'b'"):
        validate_dataframe(df, schema)


def test_null_in_feature_raises(schema: DatasetSchema) -> None:
    df = _good_df()
    df.loc[0, "a"] = np.nan
    with pytest.raises(ValueError, match="nulls"):
        validate_dataframe(df, schema)


def test_unknown_treatment_value_raises(schema: DatasetSchema) -> None:
    df = _good_df()
    df.loc[0, "t"] = 2
    with pytest.raises(ValueError, match="treatment column"):
        validate_dataframe(df, schema)


def test_unknown_outcome_value_raises(schema: DatasetSchema) -> None:
    df = _good_df()
    df.loc[0, "y"] = 7
    with pytest.raises(ValueError, match="outcome column"):
        validate_dataframe(df, schema)


def test_schema_rejects_overlap_in_feature_cols() -> None:
    with pytest.raises(ValueError, match="overlaps"):
        DatasetSchema(treatment_col="t", outcome_col="y", feature_cols=("t", "a"))


def test_uplift_dataset_views(schema: DatasetSchema) -> None:
    df = validate_dataframe(_good_df(), schema)
    ds = UpliftDataset(df=df, schema=schema, name="toy")
    assert ds.n == 4
    assert list(ds.X.columns) == ["a", "b"]
    np.testing.assert_array_equal(ds.t, np.array([0, 1, 0, 1], dtype=np.int8))
    np.testing.assert_array_equal(ds.y, np.array([0, 1, 1, 0], dtype=np.int8))
