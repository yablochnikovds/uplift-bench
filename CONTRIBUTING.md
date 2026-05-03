# Contributing

Thanks for considering a contribution. The bar is high but the rules are simple.

## Setup

```bash
uv sync --extra bench --extra dev
uv run pre-commit install
```

## Before opening a PR

```bash
make lint        # ruff
make typecheck   # mypy --strict
make test        # pytest with 90% coverage gate
```

CI runs the same three on every push and is expected to be green.

## Style

- Type hints everywhere, including return types. `mypy --strict` clean.
- NumPy-style docstrings on public functions. Skip the docstring on trivial
  private helpers — a clear name beats a docstring.
- Comments explain *why*, not *what*. If the next reader will obviously see
  what the code does, don't restate it.
- No emojis in code or commit messages.
- Conventional Commits (`feat:`, `fix:`, `refactor:`, `docs:`, `test:`,
  `ci:`, `chore:`) but loose — clarity over ceremony.

## Adding a new uplift model

1. Subclass `uplift_bench.models.base.UpliftModel`.
2. Add a Hydra config under `configs/model/<name>.yaml`.
3. Register in `uplift_bench.models.factory.MODEL_REGISTRY`.
4. Add a unit test that fits on the synthetic fixture and verifies the
   model's uplift estimates correlate with the *known* true uplift above
   a sensible threshold (≥ 0.4 for non-trivial methods).

## Adding a new dataset

1. Subclass `uplift_bench.data.base.DatasetLoader`.
2. Provide either an automatic download or a documented manual-download path
   in `docs/datasets.md`.
3. Add a pydantic schema in `uplift_bench.data.validation`.
4. Add a tiny sample under `data/sample/` so the test suite can run offline.
