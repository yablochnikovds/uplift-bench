"""Click-based CLI entry point.

The CLI is a thin shim — it dispatches to functions in `pipelines/`. Keeping
business logic out of click commands means everything is also reachable from
notebooks and tests without monkey-patching click.

Subcommands are added in their respective stages: `download` lives in the
data layer, `benchmark` in the pipelines layer. We avoid declaring placeholder
commands here so `--help` always reflects what actually works.
"""

from __future__ import annotations

import sys
from pathlib import Path

import click

from uplift_bench import __version__
from uplift_bench.utils.logging import configure


@click.group(
    help="uplift-bench — benchmark uplift modeling approaches.",
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.version_option(__version__, prog_name="uplift-bench")
@click.option(
    "--log-level",
    default="INFO",
    show_default=True,
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
)
@click.option(
    "--json-logs/--text-logs",
    default=False,
    show_default=True,
    help="Emit line-delimited JSON logs (for Docker / CI).",
)
def main(log_level: str, json_logs: bool) -> None:
    configure(level=log_level, json_logs=json_logs)


@main.command("info")
def info() -> None:
    """Print version + environment fingerprint."""
    import platform

    import numpy as np
    import pandas as pd

    click.echo(f"uplift-bench {__version__}")
    click.echo(f"python      {platform.python_version()}")
    click.echo(f"numpy       {np.__version__}")
    click.echo(f"pandas      {pd.__version__}")


@main.command("download")
@click.argument("dataset", type=click.Choice(["hillstrom", "criteo", "lenta", "all"]))
@click.option(
    "--data-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("data/raw"),
    show_default=True,
)
def download(dataset: str, data_dir: Path) -> None:
    """Download an auto-fetchable dataset to `--data-dir` (skips if cached).

    Datasets behind login walls (RetailHero, MegaFon) are not handled here —
    see docs/datasets.md.
    """
    from uplift_bench.data import download as _download

    _download.fetch(dataset, data_dir)


@main.command(
    "benchmark",
    help="Run the benchmark via Hydra. All `key=value` overrides after the "
    "command pass through to Hydra.",
    context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
)
@click.pass_context
def benchmark(ctx: click.Context) -> None:
    # Hydra owns sys.argv parsing — strip our "benchmark" subcommand off and
    # delegate. Click's allow_extra_args lets users do
    #   uplift-bench benchmark dataset=criteo model=dr_learner seed=7
    # without click eating the keys.
    from uplift_bench.pipelines.benchmark import hydra_entry

    sys.argv = [sys.argv[0], *ctx.args]
    hydra_entry()


if __name__ == "__main__":  # pragma: no cover — exercised by `python -m`
    sys.exit(main())
