"""Seed plumbing.

The motivation is boring but important: between numpy, python's random,
sklearn (which uses np), CatBoost, LightGBM, and joblib's worker forks,
it is *very* easy to leave one source of randomness un-seeded and lose
bit-for-bit reproducibility. This module is the single place that knows
about all of them.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class SeedBundle:
    """Snapshot of every seed we control. Logged to MLflow per run."""

    python: int
    numpy: int
    sklearn: int
    catboost: int
    lightgbm: int
    pythonhashseed: int


def seed_everything(seed: int = 42) -> SeedBundle:
    """Seed every source of randomness we touch.

    We deliberately use the *same* int for everything. Splitting the seed
    across libraries gives slightly more entropy but makes "rerun this exact
    experiment" much harder to explain in a paper or a README.

    Parameters
    ----------
    seed
        Master seed. Must be in [0, 2**32 - 1] because numpy enforces it.

    Returns
    -------
    SeedBundle
        Echoes back every value we set, so the caller can log it.
    """
    if not 0 <= seed < 2**32:
        raise ValueError(f"seed {seed} out of range for numpy (0..2**32-1)")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    # CatBoost and LightGBM read these env vars at fit time (in addition to
    # accepting `random_state` kwargs). Setting them belt-and-braces.
    os.environ["CATBOOST_SEED"] = str(seed)
    os.environ["LIGHTGBM_SEED"] = str(seed)

    return SeedBundle(
        python=seed,
        numpy=seed,
        sklearn=seed,
        catboost=seed,
        lightgbm=seed,
        pythonhashseed=seed,
    )
