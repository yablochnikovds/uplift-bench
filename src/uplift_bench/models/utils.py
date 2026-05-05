"""Cross-cutting helpers for the model layer.

Tiny module on purpose — it exists so the same `build_model_kwargs`
canonical form is used by `pipelines/train.py`, the matrix benchmark
script, and any one-off scripts that need to reconstruct `make_model`
calls from a config blob without re-deriving the rules.
"""

from __future__ import annotations

from typing import Any


def build_model_kwargs(
    model_name: str,
    base_learner_cfg: dict[str, Any] | None = None,
    extra_params: dict[str, Any] | None = None,
    *,
    seed: int = 0,
) -> dict[str, Any]:
    """Translate a config layout into kwargs for `make_model`.

    Causal forest takes its own knobs (no base learner); every other
    meta-learner accepts `base_learner` + `base_params`. We fold those
    plus the per-model `extra_params` into a flat kwarg dict.
    """
    extra = dict(extra_params or {})
    if model_name == "causal_forest":
        return {"seed": seed, **extra}
    bl = base_learner_cfg or {}
    return {
        "base_learner": bl.get("name", "catboost"),
        "base_params": bl.get("params") or None,
        "seed": seed,
        **extra,
    }
