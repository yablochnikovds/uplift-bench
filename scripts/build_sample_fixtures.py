"""Generate the tiny sample CSVs under `data/sample/`.

Re-run this script when the schemas in any loader change. The output files
are committed so tests stay hermetic.

Each sample is a few thousand rows, large enough that train/val/test split
gives non-degenerate folds, small enough that it fits on disk in a few KB.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent / "data" / "sample"


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def build_hillstrom(n: int = 2_000) -> None:
    rng = _rng(101)
    out = ROOT / "hillstrom"
    out.mkdir(parents=True, exist_ok=True)

    segments = rng.choice(
        ["Womens E-Mail", "Mens E-Mail", "No E-Mail"],
        size=n,
        p=[0.35, 0.35, 0.30],
    )
    df = pd.DataFrame(
        {
            "recency": rng.integers(1, 12, n),
            "history_segment": rng.choice(
                [
                    "1) $0 - $100",
                    "2) $100 - $200",
                    "3) $200 - $350",
                    "4) $350 - $500",
                    "5) $500 - $750",
                    "6) $750 - $1,000",
                    "7) $1,000 +",
                ],
                size=n,
            ),
            "history": rng.uniform(20, 1500, n).round(2),
            "mens": rng.integers(0, 2, n),
            "womens": rng.integers(0, 2, n),
            "zip_code": rng.choice(["Surburban", "Urban", "Rural"], size=n),
            "newbie": rng.integers(0, 2, n),
            "channel": rng.choice(["Multichannel", "Phone", "Web"], size=n),
            "segment": segments,
        }
    )

    # Visits: roughly 14% control, 30% treated. Conversion: ~0.5%/0.9%.
    base_visit = 0.14 + 0.16 * (df["segment"] != "No E-Mail").astype(float)
    df["visit"] = (rng.uniform(size=n) < base_visit).astype(int)
    df["conversion"] = (rng.uniform(size=n) < base_visit * 0.04).astype(int)
    df["spend"] = np.where(df["conversion"] == 1, rng.uniform(20, 200, n).round(2), 0.0)

    df.to_csv(out / "hillstrom.csv", index=False)
    print(f"wrote {out / 'hillstrom.csv'}  ({n} rows)")


def build_retailhero(n: int = 3_000) -> None:
    rng = _rng(202)
    out = ROOT / "retailhero"
    out.mkdir(parents=True, exist_ok=True)

    client_ids = [f"c_{i:06d}" for i in range(n)]
    treatment = rng.integers(0, 2, n)
    target = (rng.uniform(size=n) < (0.10 + 0.08 * treatment)).astype(int)
    pd.DataFrame(
        {
            "client_id": client_ids,
            "treatment_flg": treatment,
            "target": target,
        }
    ).to_csv(out / "uplift_train.csv", index=False)

    age = rng.integers(18, 80, n).astype(float)
    age[rng.uniform(size=n) < 0.05] = np.nan
    pd.DataFrame(
        {
            "client_id": client_ids,
            "gender": rng.choice(["F", "M", "U"], size=n, p=[0.5, 0.45, 0.05]),
            "age": age,
            "average_amount": rng.uniform(50, 5000, n).round(2),
            "purchase_sum_3m": rng.uniform(0, 50_000, n).round(2),
            "purchase_count_3m": rng.integers(0, 50, n),
            "days_since_first": rng.integers(30, 2000, n),
            "days_since_last": rng.integers(0, 365, n),
        }
    ).to_csv(out / "clients.csv", index=False)
    print(f"wrote {out / 'uplift_train.csv'} + clients.csv  ({n} rows)")


def build_megafon(n: int = 3_000, n_features: int = 12) -> None:
    rng = _rng(303)
    out = ROOT / "megafon"
    out.mkdir(parents=True, exist_ok=True)

    data = {f"X_{i}": rng.standard_normal(n).round(4) for i in range(n_features)}
    data["treatment_group"] = rng.choice(["treatment", "control"], size=n)
    base = 0.08 + 0.06 * (np.array(data["treatment_group"]) == "treatment").astype(float)
    data["conversion"] = (rng.uniform(size=n) < base).astype(int)
    pd.DataFrame(data).to_csv(out / "train.csv", index=False)
    print(f"wrote {out / 'train.csv'}  ({n} rows, {n_features} features)")


if __name__ == "__main__":
    ROOT.mkdir(parents=True, exist_ok=True)
    build_hillstrom()
    build_retailhero()
    build_megafon()
