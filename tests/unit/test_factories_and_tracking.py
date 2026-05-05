"""Coverage-focused tests for the small surface modules.

Each thing here is tiny enough that adding a separate test file per
module would be more noise than signal — keep them grouped.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from uplift_bench.data.factory import DATASET_REGISTRY, make_loader
from uplift_bench.data.hillstrom import HillstromLoader
from uplift_bench.metrics.bootstrap import BootstrapCI
from uplift_bench.pipelines.report import aggregate_results, write_markdown_table
from uplift_bench.pipelines.train import TrainResult
from uplift_bench.tracking.mlflow_logger import MLflowRun, _resolve_file_uri, start_run


def test_dataset_factory_returns_correct_class() -> None:
    loader = make_loader("hillstrom", data_dir="data/sample")
    assert isinstance(loader, HillstromLoader)


def test_dataset_factory_unknown_raises() -> None:
    with pytest.raises(ValueError, match="unknown dataset"):
        make_loader("nope", data_dir="/tmp")


def test_dataset_registry_covers_all_six() -> None:
    expected = {
        "hillstrom",
        "criteo",
        "lenta",
        "retailhero",
        "megafon",
        "synthetic",
    }
    assert expected == set(DATASET_REGISTRY)


def test_resolve_file_uri_passes_non_file_through() -> None:
    assert _resolve_file_uri("http://x") == "http://x"
    assert _resolve_file_uri("sqlite:///x.db") == "sqlite:///x.db"


def test_resolve_file_uri_keeps_absolute_paths() -> None:
    assert _resolve_file_uri("file:///abs/path") == "file:///abs/path"


def test_resolve_file_uri_makes_relative_absolute(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    out = _resolve_file_uri("file://./mlruns")
    assert out.startswith("file://")
    assert out.endswith("mlruns")
    assert "/./" not in out
    # Path component is absolute.
    assert Path(out[len("file://") :]).is_absolute()


def test_disabled_run_is_a_noop_for_all_methods(tmp_path: Path) -> None:
    bogus = tmp_path / "bogus"  # never created — noop must not touch fs
    with start_run(
        enabled=False,
        experiment_name="x",
        tracking_uri=f"file://{bogus}",
    ) as run:
        run.log_params({"a": 1, "b": [1, 2]})
        run.log_metric("m", 0.5)
        run.log_metrics({"m1": 1.0, "m2": 2.0})
        run.log_metric_with_ci(
            "q",
            BootstrapCI(
                point=0.1,
                lower=0.0,
                upper=0.2,
                method="bca",
                alpha=0.05,
                n_boot=100,
            ),
        )
        run.log_text("hello", "hi.txt")
        run.log_dict({"a": 1}, "d.json")
        run.set_tag("tag", "value")
    assert isinstance(run, MLflowRun)
    assert not bogus.exists()


def test_aggregate_results_and_markdown(tmp_path: Path) -> None:
    results = [
        TrainResult(
            model_name="s_learner",
            dataset_name="hillstrom",
            seed=1,
            qini=0.10,
            qini_ci_lower=0.05,
            qini_ci_upper=0.15,
            auuc=0.4,
            metrics={"auuc_normalized": 0.4, "uplift_at_10": 0.02},
        ),
        TrainResult(
            model_name="dr_learner",
            dataset_name="hillstrom",
            seed=1,
            qini=0.14,
            qini_ci_lower=0.09,
            qini_ci_upper=0.19,
            auuc=0.5,
            metrics={"auuc_normalized": 0.5, "uplift_at_10": 0.03},
        ),
    ]
    df = aggregate_results(results)
    assert {"model", "dataset", "qini", "qini_ci_lower", "qini_ci_upper", "auuc_normalized"} <= set(
        df.columns
    )
    assert len(df) == 2

    out = tmp_path / "results.md"
    write_markdown_table(df, out)
    text = out.read_text()
    assert "dr_learner" in text
    assert "hillstrom" in text
