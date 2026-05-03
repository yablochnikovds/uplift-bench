"""Tiny IO helpers used by data loaders and the report writer.

We deliberately keep this surface area small — anything more elaborate
(parquet partitioning, S3 paths, etc.) gets its own module.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

CHUNK_SIZE = 1 << 20  # 1 MiB — enough that hashing 1 GB CSV stays under ~5s


def ensure_dir(path: Path) -> Path:
    """Create `path` (and parents) if missing. Returns it for chaining."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def file_sha256(path: Path) -> str:
    """SHA-256 of a file, streamed.

    We use this to fingerprint downloaded datasets so MLflow runs carry
    the exact bytes they trained on. md5 would be faster but collides.
    """
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(CHUNK_SIZE):
            h.update(chunk)
    return h.hexdigest()


def write_parquet_atomic(df: pd.DataFrame, path: Path) -> None:
    """Write parquet via tmp+rename so a crash never leaves a half file.

    pandas' to_parquet is not atomic — if the process dies mid-write the
    file looks valid to `os.path.exists` but pyarrow blows up reading it.
    Renames inside the same FS are atomic on POSIX, so write-then-rename
    is enough.
    """
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False, engine="pyarrow", compression="zstd")
    tmp.replace(path)


def dump_json(obj: Any, path: Path) -> None:
    """JSON dump with sort_keys + indent so diffs in git stay sane."""
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str))
