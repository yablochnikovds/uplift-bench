# Sample fixtures

Tiny stand-ins for the real datasets. They:

- Match the production schema column-for-column.
- Are small enough to commit (a few KB each).
- Let the test suite and `quick_smoke` config run with zero network.

These are **not** the real data — numbers here are synthetic and any model
results on them are meaningless. For real benchmark numbers, place the
real files under `data/raw/` (see `docs/datasets.md`).
