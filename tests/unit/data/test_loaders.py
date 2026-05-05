"""Loader tests against the committed sample fixtures.

These tests intentionally do not hit the network. The full-data loaders
(criteo, real hillstrom) are exercised separately under the `needs_data`
marker in `tests/integration/`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from uplift_bench.data.hillstrom import HillstromLoader
from uplift_bench.data.lenta import LentaLoader
from uplift_bench.data.megafon import MegaFonLoader
from uplift_bench.data.retailhero import RetailHeroLoader

SAMPLE_DIR = Path(__file__).resolve().parents[3] / "data" / "sample"


def test_hillstrom_loads_from_sample() -> None:
    # Sample has the same layout as `data/raw/hillstrom/hillstrom.csv`.
    loader = HillstromLoader(data_dir=SAMPLE_DIR.parent / "sample")
    ds = loader.load()
    assert ds.n > 100
    # Treatment binarised to one arm vs control — must be {0, 1} only.
    assert set(ds.df["treatment"].unique()) <= {0, 1}
    assert set(ds.df["outcome"].unique()) <= {0, 1}
    # Every schema feature is present.
    for col in ds.schema.feature_cols:
        assert col in ds.df.columns


def test_hillstrom_treatment_arm_validation() -> None:
    with pytest.raises(ValueError, match="treatment_arm"):
        HillstromLoader(data_dir=SAMPLE_DIR, treatment_arm="something else")


def test_hillstrom_outcome_validation() -> None:
    with pytest.raises(ValueError, match="outcome"):
        HillstromLoader(data_dir=SAMPLE_DIR, outcome="spend")


def test_retailhero_loads_from_sample() -> None:
    loader = RetailHeroLoader(data_dir=SAMPLE_DIR)
    ds = loader.load()
    assert ds.n > 100
    assert set(ds.df["treatment"].unique()) <= {0, 1}
    assert set(ds.df["outcome"].unique()) <= {0, 1}
    # All gender dummies are present.
    for g in ("F", "M", "U"):
        assert f"gender_{g}" in ds.df.columns
    # Numeric features have no nulls (median imputation).
    assert ds.df[["age", "average_amount"]].isna().sum().sum() == 0


def test_retailhero_missing_files_message_is_helpful() -> None:
    loader = RetailHeroLoader(data_dir=Path("/nonexistent/dir"))
    with pytest.raises(FileNotFoundError, match="login-walled"):
        loader.load()


def test_megafon_loads_from_sample() -> None:
    loader = MegaFonLoader(data_dir=SAMPLE_DIR)
    ds = loader.load()
    assert ds.n > 100
    assert set(ds.df["treatment"].unique()) <= {0, 1}
    # MegaFon's feature set is discovered at read time, not declared statically.
    assert all(f.startswith("X_") for f in ds.schema.feature_cols)
    assert len(ds.schema.feature_cols) >= 1


def test_megafon_missing_file_message_is_helpful() -> None:
    loader = MegaFonLoader(data_dir=Path("/nonexistent/dir"))
    with pytest.raises(FileNotFoundError, match="login-walled"):
        loader.load()


def test_lenta_loads_from_sample() -> None:
    loader = LentaLoader(data_dir=SAMPLE_DIR)
    ds = loader.load()
    assert ds.n > 100
    assert set(ds.df["treatment"].unique()) <= {0, 1}
    assert set(ds.df["outcome"].unique()) <= {0, 1}
    # Numeric and dummy features are present.
    for col in ("age", "main_format", "gender_F" if False else "gender_Ж"):
        assert col in ds.df.columns
    # Numeric block has no NaNs (median imputation).
    assert ds.df[["age"]].isna().sum().sum() == 0
