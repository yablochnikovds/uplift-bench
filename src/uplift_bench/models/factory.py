"""Build an UpliftModel from a string name + kwargs.

Used by the Hydra config loader and the CLI. The registry is the single
place to add a new meta-learner — every other module imports from here.
"""

from __future__ import annotations

from typing import Any

from uplift_bench.models.base import UpliftModel
from uplift_bench.models.causal_forest import CausalForestModel
from uplift_bench.models.class_transformation import ClassTransformationLearner
from uplift_bench.models.dr_learner import DRLearner
from uplift_bench.models.r_learner import RLearner
from uplift_bench.models.s_learner import SLearner
from uplift_bench.models.t_learner import TLearner
from uplift_bench.models.x_learner import XLearner

MODEL_REGISTRY: dict[str, type[UpliftModel]] = {
    "s_learner": SLearner,
    "t_learner": TLearner,
    "x_learner": XLearner,
    "r_learner": RLearner,
    "dr_learner": DRLearner,
    "class_transformation": ClassTransformationLearner,
    "causal_forest": CausalForestModel,
}


def make_model(name: str, **kwargs: Any) -> UpliftModel:
    """Instantiate a meta-learner by short name."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"unknown model {name!r}; choose from {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](**kwargs)
