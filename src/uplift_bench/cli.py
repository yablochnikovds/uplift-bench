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


if __name__ == "__main__":  # pragma: no cover — exercised by `python -m`
    sys.exit(main())
