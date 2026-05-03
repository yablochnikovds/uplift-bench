"""Criteo loader tests using a tiny in-memory CSV.gz fixture.

We never download the real ~300 MB Criteo file in the test suite — those
runs live under `needs_data` and only run when explicitly requested.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from uplift_bench.data.criteo import CRITEO_FILENAME, CriteoLoader


def _write_fake_criteo(data_dir: Path, n: int = 800) -> Path:
    """Build a tiny CSV.gz that matches the real Criteo schema."""
    rng = np.random.default_rng(0)
    cols = {f"f{i}": rng.standard_normal(n).astype("float32") for i in range(12)}
    cols["treatment"] = rng.integers(0, 2, n).astype("int8")
    cols["conversion"] = rng.integers(0, 2, n).astype("int8")
    cols["visit"] = rng.integers(0, 2, n).astype("int8")
    cols["exposure"] = rng.integers(0, 2, n).astype("int8")

    df = pd.DataFrame(cols)
    out_dir = data_dir / "criteo"
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / CRITEO_FILENAME
    with gzip.open(target, "wt") as fh:
        df.to_csv(fh, index=False)
    return target


def test_criteo_loads_and_caches_parquet(tmp_path: Path) -> None:
    _write_fake_criteo(tmp_path)
    loader = CriteoLoader(data_dir=tmp_path)

    ds1 = loader.load()
    assert ds1.n == 800
    assert set(ds1.df["treatment"].unique()) <= {0, 1}
    assert set(ds1.df["outcome"].unique()) <= {0, 1}
    # Cached parquet should now exist.
    assert (tmp_path / "criteo" / "criteo-research-uplift-v2.1.parquet").exists()

    # Second load uses the parquet — no exception expected.
    ds2 = loader.load()
    assert ds2.n == 800


def test_criteo_subsample(tmp_path: Path) -> None:
    _write_fake_criteo(tmp_path, n=500)
    loader = CriteoLoader(data_dir=tmp_path, subsample=100, subsample_seed=0)
    ds = loader.load()
    assert ds.n == 100


def test_criteo_subsample_larger_than_dataset_keeps_all(tmp_path: Path) -> None:
    _write_fake_criteo(tmp_path, n=200)
    loader = CriteoLoader(data_dir=tmp_path, subsample=10_000)
    ds = loader.load()
    assert ds.n == 200


def test_criteo_outcome_choice_changes_labels(tmp_path: Path) -> None:
    _write_fake_criteo(tmp_path)
    visit_loader = CriteoLoader(data_dir=tmp_path, outcome="visit")
    conv_loader = CriteoLoader(data_dir=tmp_path, outcome="conversion")
    visit_ds = visit_loader.load()
    conv_ds = conv_loader.load()
    # On a random fixture these will almost surely differ.
    assert not np.array_equal(visit_ds.df["outcome"].to_numpy(),
                              conv_ds.df["outcome"].to_numpy())


def test_criteo_invalid_outcome() -> None:
    with pytest.raises(ValueError, match="outcome"):
        CriteoLoader(data_dir=Path("/x"), outcome="spend")


def test_criteo_invalid_subsample() -> None:
    with pytest.raises(ValueError, match="subsample"):
        CriteoLoader(data_dir=Path("/x"), subsample=0)


def test_criteo_missing_file_attempts_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the file isn't on disk, the loader hits download_file. We stub
    download_file rather than the network so the test is offline."""
    from uplift_bench.data import criteo as criteo_mod

    called: list[str] = []

    def _stub(url: str, dest: Path, **_: object) -> Path:
        called.append(url)
        # Materialise a tiny csv.gz at dest so _read can proceed.
        _write_fake_criteo(tmp_path, n=10)
        return dest

    monkeypatch.setattr(criteo_mod, "download_file", _stub)

    loader = CriteoLoader(data_dir=tmp_path)
    ds = loader.load()
    assert called  # download_file was invoked
    assert ds.n == 10
