from __future__ import annotations

from pathlib import Path

import pandas as pd

from uplift_bench.utils.io import dump_json, ensure_dir, file_sha256, write_parquet_atomic


def test_ensure_dir_is_idempotent(tmp_path: Path) -> None:
    target = tmp_path / "a" / "b" / "c"
    assert ensure_dir(target).exists()
    # Calling twice must not raise.
    assert ensure_dir(target).exists()


def test_file_sha256_stable(tmp_path: Path) -> None:
    p = tmp_path / "x.bin"
    p.write_bytes(b"the quick brown fox" * 1000)
    h1 = file_sha256(p)
    h2 = file_sha256(p)
    assert h1 == h2
    assert len(h1) == 64  # sha256 hex


def test_file_sha256_changes_with_content(tmp_path: Path) -> None:
    p1 = tmp_path / "a.bin"
    p2 = tmp_path / "b.bin"
    p1.write_bytes(b"aaaa")
    p2.write_bytes(b"aaab")
    assert file_sha256(p1) != file_sha256(p2)


def test_write_parquet_atomic_roundtrip(tmp_path: Path) -> None:
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    out = tmp_path / "nested" / "df.parquet"
    write_parquet_atomic(df, out)
    assert out.exists()
    assert not out.with_suffix(".parquet.tmp").exists()
    pd.testing.assert_frame_equal(pd.read_parquet(out), df)


def test_dump_json_sorts_keys(tmp_path: Path) -> None:
    p = tmp_path / "out.json"
    dump_json({"b": 2, "a": 1}, p)
    text = p.read_text()
    assert text.index('"a"') < text.index('"b"')
