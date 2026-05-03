"""Tests for the HTTP downloader's local-side logic.

We don't hit the real network — the focus is the cache-and-verify behaviour
that I keep accidentally regressing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from uplift_bench.data import download as dl
from uplift_bench.utils.io import file_sha256


def test_skip_when_file_exists_and_no_hash_check(tmp_path: Path) -> None:
    target = tmp_path / "file.bin"
    target.write_bytes(b"already-here")
    # No URL is hit; if it were, the test would error with a connection refused.
    out = dl.download_file("http://localhost:1/never-reached", target)
    assert out == target
    assert target.read_bytes() == b"already-here"


def test_skip_when_file_exists_and_hash_matches(tmp_path: Path) -> None:
    target = tmp_path / "file.bin"
    target.write_bytes(b"hello")
    sha = file_sha256(target)
    out = dl.download_file("http://localhost:1/never-reached", target, expected_sha256=sha)
    assert out == target


def test_redownload_when_hash_mismatches_calls_http(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the on-disk hash doesn't match, we re-fetch from the URL."""
    import hashlib

    target = tmp_path / "file.bin"
    target.write_bytes(b"old-content")
    expected_payload = b"hello"
    expected_sha = hashlib.sha256(expected_payload).hexdigest()

    calls: list[str] = []

    class _FakeResp:
        headers = {"content-length": str(len(expected_payload))}

        def raise_for_status(self) -> None: ...
        def iter_content(self, chunk_size: int) -> object:
            del chunk_size
            return iter([expected_payload])
        def __enter__(self) -> "_FakeResp":
            return self
        def __exit__(self, *a: object) -> None: ...

    def _fake_get(url: str, **_: object) -> _FakeResp:
        calls.append(url)
        return _FakeResp()

    monkeypatch.setattr(dl.requests, "get", _fake_get)

    out = dl.download_file(
        "http://example.test/payload",
        target,
        expected_sha256=expected_sha,
    )
    assert calls == ["http://example.test/payload"]
    assert out.read_bytes() == expected_payload


def test_hash_mismatch_after_download_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "file.bin"

    class _FakeResp:
        headers = {"content-length": "3"}

        def raise_for_status(self) -> None: ...
        def iter_content(self, chunk_size: int) -> object:
            del chunk_size
            return iter([b"abc"])
        def __enter__(self) -> "_FakeResp":
            return self
        def __exit__(self, *a: object) -> None: ...

    monkeypatch.setattr(dl.requests, "get", lambda *_a, **_kw: _FakeResp())

    with pytest.raises(ValueError, match="sha256 mismatch"):
        dl.download_file("http://example.test/x", target, expected_sha256="0" * 64)
    # Tmp file must be cleaned up so the next run doesn't re-trip on it.
    assert not target.with_suffix(".bin.part").exists()
    assert not target.exists()


def test_overwrite_forces_redownload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "file.bin"
    target.write_bytes(b"old")
    calls: list[str] = []

    class _FakeResp:
        headers: dict[str, str] = {}

        def raise_for_status(self) -> None: ...
        def iter_content(self, chunk_size: int) -> object:
            del chunk_size
            return iter([b"new-bytes"])
        def __enter__(self) -> "_FakeResp":
            return self
        def __exit__(self, *a: object) -> None: ...

    def _fake_get(url: str, **_: object) -> _FakeResp:
        calls.append(url)
        return _FakeResp()

    monkeypatch.setattr(dl.requests, "get", _fake_get)
    dl.download_file("http://example.test/x", target, overwrite=True)
    assert calls == ["http://example.test/x"]
    assert target.read_bytes() == b"new-bytes"


def test_fetch_unknown_dataset() -> None:
    with pytest.raises(ValueError, match="unknown dataset"):
        dl.fetch("nope", Path("/tmp"))
