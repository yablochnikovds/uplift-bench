"""HTTP downloader with resume + sha verification.

Used by individual loaders that can self-download (Hillstrom, Criteo).
Datasets that gate behind login (RetailHero, MegaFon) raise a clear error
in their loader instead of going through here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import requests
from tqdm.auto import tqdm

from uplift_bench.utils.io import ensure_dir, file_sha256
from uplift_bench.utils.logging import get_logger

log = get_logger(__name__)

# 1 MiB chunks: small enough to update the progress bar smoothly, large
# enough to avoid syscall overhead on multi-GB files.
CHUNK: Final[int] = 1 << 20


def download_file(
    url: str,
    dest: Path,
    *,
    expected_sha256: str | None = None,
    overwrite: bool = False,
    timeout: float = 60.0,
) -> Path:
    """Download `url` to `dest`. Skips if file exists and hash matches.

    Parameters
    ----------
    url
        Direct HTTP(S) URL. We don't follow login redirects on purpose —
        if a host needs auth, the caller should fail loud, not silently.
    dest
        Target path. Parent created if missing.
    expected_sha256
        If given, verifies the downloaded file. On mismatch the file is
        deleted and a ValueError is raised — better than silently using
        corrupted data.
    overwrite
        Force re-download even if `dest` exists.
    timeout
        Per-request connect/read timeout, in seconds.

    Returns
    -------
    Path
        `dest` itself, for chaining.
    """
    ensure_dir(dest.parent)

    if dest.exists() and not overwrite:
        if expected_sha256 is None:
            log.info("download_skip_exists", path=str(dest))
            return dest
        actual = file_sha256(dest)
        if actual == expected_sha256:
            log.info("download_skip_hash_match", path=str(dest))
            return dest
        log.warning(
            "download_hash_mismatch_redownloading",
            path=str(dest), expected=expected_sha256, actual=actual,
        )

    tmp = dest.with_suffix(dest.suffix + ".part")
    log.info("download_start", url=url, dest=str(dest))

    with requests.get(url, stream=True, timeout=timeout) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length") or 0)
        with tmp.open("wb") as fh, tqdm(
            total=total, unit="B", unit_scale=True, unit_divisor=1024,
            desc=dest.name, leave=False,
        ) as bar:
            for chunk in resp.iter_content(chunk_size=CHUNK):
                if not chunk:
                    continue
                fh.write(chunk)
                bar.update(len(chunk))

    if expected_sha256 is not None:
        actual = file_sha256(tmp)
        if actual != expected_sha256:
            tmp.unlink(missing_ok=True)
            raise ValueError(
                f"sha256 mismatch for {url}: expected {expected_sha256}, got {actual}"
            )

    tmp.replace(dest)
    log.info("download_done", path=str(dest), bytes=dest.stat().st_size)
    return dest


def fetch(name: str, data_dir: Path) -> None:
    """CLI dispatch: download one (or all) auto-downloadable dataset(s)."""
    from uplift_bench.data.criteo import CriteoLoader
    from uplift_bench.data.hillstrom import HillstromLoader
    from uplift_bench.data.lenta import LentaLoader

    targets = {
        "hillstrom": HillstromLoader,
        "criteo": CriteoLoader,
        "lenta": LentaLoader,
    }
    if name == "all":
        loaders = list(targets.values())
    elif name in targets:
        loaders = [targets[name]]
    else:
        raise ValueError(
            f"unknown dataset {name!r}; choose from {sorted(targets) + ['all']}"
        )

    for cls in loaders:
        loader = cls(data_dir=data_dir)
        loader.download()
