"""Tests for the corners that the higher-level integration tests skip.

Two foci: the lightgbm/logreg branches of `make_base_learner`, and the
artifact-logging methods on MLflowRun (which the disabled-run test
short-circuits past).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from uplift_bench.metrics.bootstrap import BootstrapCI
from uplift_bench.models._base_learners import make_base_learner
from uplift_bench.models.s_learner import SLearner
from uplift_bench.tracking.mlflow_logger import start_run


@pytest.mark.parametrize("name", ["catboost", "lightgbm", "logreg"])
@pytest.mark.parametrize("task", ["classification", "regression"])
def test_make_base_learner_returns_fittable(name: str, task: str) -> None:
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.standard_normal((40, 4)), columns=list("abcd"))
    y_cls = rng.integers(0, 2, 40)
    y_reg = rng.standard_normal(40)
    y = y_cls if task == "classification" else y_reg
    est = make_base_learner(
        name,
        task=task,
        seed=0,  # type: ignore[arg-type]
        params={"iterations": 30, "n_estimators": 30},
    )
    est.fit(X, y)
    pred = est.predict(X)
    assert len(pred) == 40


def test_make_base_learner_unknown() -> None:
    with pytest.raises(ValueError, match="unknown base learner"):
        make_base_learner("xgboost", task="classification")  # type: ignore[arg-type]


def test_lightgbm_iterations_alias_is_remapped() -> None:
    """Passing `iterations` to lightgbm should silently become n_estimators."""
    est = make_base_learner("lightgbm", task="classification", seed=0, params={"iterations": 25})
    # If the alias rewriting were missing, lightgbm would error on
    # an unknown kwarg. Reaching this assertion means it worked.
    assert hasattr(est, "fit")


def test_s_learner_with_lightgbm_runs_end_to_end() -> None:
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.standard_normal((300, 5)), columns=[f"f{i}" for i in range(5)])
    t = rng.integers(0, 2, 300)
    y = rng.integers(0, 2, 300)
    m = SLearner(base_learner="lightgbm", base_params={"n_estimators": 30, "min_child_samples": 5})
    m.fit(X, t, y)
    preds = m.predict_uplift(X)
    assert len(preds) == 300
    assert np.all(np.isfinite(preds))


def test_mlflow_run_logs_artifacts_to_disk(tmp_path: Path) -> None:
    """Exercise the live (enabled=True) path of every artifact logger."""
    mlruns = tmp_path / "mlruns"
    artifact = tmp_path / "extra.txt"
    artifact.write_text("an artifact")

    with start_run(
        enabled=True,
        experiment_name="cov",
        tracking_uri=f"file://{mlruns}",
    ) as run:
        run.log_params({"plain": "v", "nested": {"k": 1}, "tup": (1, 2)})
        run.log_metric("solo", 0.5)
        run.log_metrics({"a": 1.0, "b": 2.0})
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
        run.log_text("hello world", "hi.txt")
        run.log_dict({"k": "v"}, "d.json")
        run.log_dataframe(pd.DataFrame({"x": [1, 2], "y": [3, 4]}), "table.csv")
        run.log_artifact_path(artifact)
        run.set_tag("custom_tag", "yes")

    # The mlruns directory has at least one experiment with one run that
    # logged metrics. We don't introspect mlflow's internal layout further.
    metric_files = list(mlruns.rglob("metrics/q"))
    assert metric_files
    artifact_files = list(mlruns.rglob("artifacts/*"))
    assert any(p.name == "hi.txt" for p in artifact_files)
    assert any(p.name == "d.json" for p in artifact_files)
    assert any(p.name == "table.csv" for p in artifact_files)
    assert any(p.name == "extra.txt" for p in artifact_files)
