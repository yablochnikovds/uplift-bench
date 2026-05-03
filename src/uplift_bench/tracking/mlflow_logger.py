"""Thin wrapper around mlflow that the pipelines actually call.

Goals:
  * One context manager, no boilerplate at call sites.
  * Logs everything we need (params, metrics with CIs, artifacts, dataset
    fingerprint) without the caller having to remember names.
  * Works in `--no-tracking` mode by short-circuiting to a no-op so unit
    tests don't have to spin up mlflow.
"""

from __future__ import annotations

import contextlib
import json
import os
import platform
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd

from uplift_bench import __version__
from uplift_bench.metrics.bootstrap import BootstrapCI
from uplift_bench.utils.logging import get_logger

log = get_logger(__name__)


def _resolve_file_uri(uri: str) -> str:
    """Make a `file://./...` tracking URI absolute and Hydra-cwd-safe."""
    if not uri.startswith("file://"):
        return uri
    path_part = uri[len("file://") :]
    # Already absolute (file:///abs or file://host/abs)?
    if path_part.startswith("/"):
        return uri
    # Prefer Hydra's original cwd if we're inside a hydra run; falls back to
    # the actual cwd otherwise.
    base_str = os.environ.get("HYDRA_RUN_DIR_OWNER") or _hydra_orig_cwd() or os.getcwd()
    base = Path(base_str)
    return "file://" + str((base / path_part.lstrip("./")).resolve())


def _hydra_orig_cwd() -> str | None:
    # Lazy import — hydra isn't a runtime dep of the tracking module
    # (callers can use MLflow without ever touching Hydra).
    try:
        from hydra.core.hydra_config import HydraConfig  # noqa: PLC0415
    except ImportError:
        return None
    try:
        return str(HydraConfig.get().runtime.cwd)
    except Exception:
        return None


class MLflowRun:
    """Active-run wrapper. Use via the `start_run` context manager below."""

    def __init__(self, *, enabled: bool) -> None:
        self._enabled = enabled

    def log_params(self, params: dict[str, Any]) -> None:
        if not self._enabled:
            return
        # MLflow stringifies values; nested dicts get JSON-encoded so they
        # round-trip cleanly when read back from the UI.
        flat: dict[str, str] = {}
        for k, v in params.items():
            flat[k] = json.dumps(v) if isinstance(v, dict | list | tuple) else str(v)
        mlflow.log_params(flat)

    def log_metric_with_ci(
        self,
        name: str,
        ci: BootstrapCI,
        *,
        step: int | None = None,
    ) -> None:
        if not self._enabled:
            return
        mlflow.log_metric(f"{name}", ci.point, step=step)
        mlflow.log_metric(f"{name}_ci_lower", ci.lower, step=step)
        mlflow.log_metric(f"{name}_ci_upper", ci.upper, step=step)

    def log_metric(self, name: str, value: float, step: int | None = None) -> None:
        if not self._enabled:
            return
        mlflow.log_metric(name, value, step=step)

    def log_metrics(self, metrics: dict[str, float]) -> None:
        if not self._enabled:
            return
        mlflow.log_metrics(metrics)

    def log_dataframe(self, df: pd.DataFrame, filename: str) -> None:
        """Log a DataFrame as a CSV artifact."""
        if not self._enabled:
            return
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / filename
            df.to_csv(path, index=False)
            mlflow.log_artifact(str(path))

    def log_text(self, content: str, filename: str) -> None:
        if not self._enabled:
            return
        mlflow.log_text(content, filename)

    def log_dict(self, data: dict[str, Any], filename: str) -> None:
        if not self._enabled:
            return
        mlflow.log_dict(data, filename)

    def log_artifact_path(self, path: Path) -> None:
        if not self._enabled:
            return
        mlflow.log_artifact(str(path))

    def set_tag(self, name: str, value: str) -> None:
        if not self._enabled:
            return
        mlflow.set_tag(name, value)


@contextlib.contextmanager
def start_run(
    *,
    enabled: bool,
    experiment_name: str,
    tracking_uri: str,
    run_name: str | None = None,
    tags: dict[str, str] | None = None,
) -> Iterator[MLflowRun]:
    """Open an MLflow run (or a no-op equivalent) for the duration of `with`."""
    if not enabled:
        yield MLflowRun(enabled=False)
        return

    # Hydra changes cwd to its run dir, which would turn "file://./mlruns"
    # into "file:///mlruns" once Path resolves it. Resolve relative file
    # URIs against either Hydra's original cwd (if available) or the
    # current cwd at run-start, so mlruns lives at a stable location.
    tracking_uri = _resolve_file_uri(tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    base_tags = {
        "uplift_bench_version": __version__,
        "python": platform.python_version(),
        "host": platform.node(),
        "argv": " ".join(sys.argv),
        "user": os.environ.get("USER", "unknown"),
    }
    if tags:
        base_tags.update(tags)

    with mlflow.start_run(run_name=run_name, tags=base_tags):
        log.info("mlflow_run_started", run_name=run_name, experiment=experiment_name)
        yield MLflowRun(enabled=True)
