"""End-to-end test: data load → split → fit → metrics → MLflow log.

Uses the sample Hillstrom fixture so it runs offline. Tracking goes to a
tmp dir so we can inspect the resulting MLflow store.
"""

from __future__ import annotations

from pathlib import Path

from uplift_bench.pipelines.train import run_one


def test_end_to_end_hillstrom_smoke(tmp_path: Path) -> None:
    out = tmp_path / "out"
    mlruns = tmp_path / "mlruns"
    result = run_one(
        dataset_cfg={
            "name": "hillstrom",
            "data_dir": "data/sample",
            "loader_params": {
                "treatment_arm": "Womens E-Mail",
                "outcome": "visit",
            },
        },
        model_cfg={"name": "s_learner", "extra_params": {}},
        base_learner_cfg={"name": "logreg", "params": {"max_iter": 200}},
        split_cfg={"train_frac": 0.7, "val_frac": 0.15, "seed": 0},
        bootstrap_cfg={
            "n_boot": 80,
            "method": "percentile",
            "alpha": 0.05,
            "n_jobs": 1,
            "seed": 0,
        },
        robustness_cfg={
            "enable_permutation": True,
            "permutation_n_repeats": 2,
            "enable_overlap": True,
        },
        tracking_cfg={
            "enabled": True,
            "experiment_name": "test",
            "tracking_uri": f"file://{mlruns}",
        },
        seed=0,
        output_dir=out,
    )
    assert result.qini_ci_lower <= result.qini <= result.qini_ci_upper
    # Artefacts on disk.
    assert (out / "qini_curve.png").exists()
    assert (out / "deciles.csv").exists()
    assert (out / "config.json").exists()
    assert (out / "permutation_importance.csv").exists()
    assert (out / "propensity.csv").exists()
    # MLflow store has at least one run.
    assert mlruns.exists()
    runs = list(mlruns.rglob("metrics/qini"))
    assert runs, "no qini metric was logged in MLflow store"


def test_end_to_end_with_tracking_disabled(tmp_path: Path) -> None:
    """Pipeline must still run when MLflow tracking is off."""
    out = tmp_path / "out"
    result = run_one(
        dataset_cfg={
            "name": "hillstrom",
            "data_dir": "data/sample",
            "loader_params": {"treatment_arm": "Womens E-Mail", "outcome": "visit"},
        },
        model_cfg={"name": "s_learner", "extra_params": {}},
        base_learner_cfg={"name": "logreg", "params": {"max_iter": 100}},
        split_cfg={"train_frac": 0.7, "val_frac": 0.15, "seed": 0},
        bootstrap_cfg={"n_boot": 60, "method": "percentile", "alpha": 0.05, "n_jobs": 1, "seed": 0},
        robustness_cfg={"enable_permutation": False, "enable_overlap": False},
        tracking_cfg={
            "enabled": False,
            "experiment_name": "test",
            "tracking_uri": "file:///tmp/never-touched",
        },
        seed=0,
        output_dir=out,
    )
    assert (out / "qini_curve.png").exists()
    assert result.qini_ci_lower <= result.qini <= result.qini_ci_upper
