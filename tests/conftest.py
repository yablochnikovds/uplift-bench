"""Shared pytest fixtures.

Heavy fixtures (synthetic uplift datasets) live in `tests/fixtures/synthetic.py`
and are exposed here so any test under `tests/` can request them by name.
"""

from __future__ import annotations

import pytest

from uplift_bench.utils.reproducibility import seed_everything


@pytest.fixture(autouse=True)
def _seed_each_test() -> None:
    """Reseed before every test so flaky CI runs aren't a thing."""
    seed_everything(42)
