from __future__ import annotations

from uplift_bench.utils import logging as log_mod


def test_get_logger_lazy_configures() -> None:
    # Reset module state so we exercise the lazy path even if another test ran first.
    log_mod._CONFIGURED = False
    logger = log_mod.get_logger("uplift_bench.test")
    assert logger is not None
    assert log_mod._CONFIGURED is True


def test_configure_is_idempotent() -> None:
    log_mod._CONFIGURED = False
    log_mod.configure(level="INFO")
    first = log_mod._CONFIGURED
    log_mod.configure(level="DEBUG")
    # Second call must be a no-op; otherwise we'd double-handle log records.
    assert first is True
    assert log_mod._CONFIGURED is True
