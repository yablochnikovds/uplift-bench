"""uplift-bench — reproducible benchmark of uplift modeling approaches.

The top-level package re-exports the high-level API people will actually
import from notebooks. Internal modules (data loaders, metric primitives,
robustness routines) live in submodules and aren't auto-imported here on
purpose — keeping `import uplift_bench` cheap matters when the library
is loaded inside MLflow workers and CI jobs.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("uplift-bench")
except PackageNotFoundError:  # editable install before metadata is generated
    __version__ = "0.0.0+local"

__all__ = ["__version__"]
