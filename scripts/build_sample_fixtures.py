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


def build_lenta(n: int = 4_000) -> None:
    rng = _rng(404)
    out = ROOT / "lenta"
    out.mkdir(parents=True, exist_ok=True)

    group = rng.choice(["test", "control"], size=n, p=[0.75, 0.25])
    base = 0.07 + 0.04 * (group == "test").astype(float)
    response = (rng.uniform(size=n) < base).astype(int)

    age = rng.integers(18, 80, n).astype(float)
    age[rng.uniform(size=n) < 0.04] = np.nan

    df = pd.DataFrame(
        {
            "group": group,
            "response_att": response,
            # Cyrillic letters intentional — matches Lenta's source-of-truth values.
            "gender": rng.choice(["Ж", "М", "U"], size=n, p=[0.55, 0.40, 0.05]),  # noqa: RUF001
            "age": age,
            "main_format": rng.integers(0, 4, n),
            "cheque_count_3m_g42": rng.integers(0, 30, n),
            "cheque_count_6m_g25": rng.integers(0, 60, n),
            "sale_count_3m_g32": rng.integers(0, 100, n),
            "sale_count_6m_g25": rng.integers(0, 200, n),
            "sale_sum_3m_g42": rng.uniform(0, 50_000, n).round(2),
            "sale_sum_6m_g25": rng.uniform(0, 100_000, n).round(2),
            "k_var_count_per_cheq_15d_g28": rng.uniform(0, 3, n).round(3),
            "food_share_15d": rng.uniform(0, 1, n).round(3),
            "response_sms": rng.integers(0, 2, n),
            "response_viber": rng.integers(0, 2, n),
            "months_from_register": rng.integers(0, 60, n),
            "months_to_response": rng.integers(0, 6, n),
        }
    )
    df.to_csv(out / "lenta_dataset.csv.gz", index=False, compression="gzip")
    print(f"wrote {out / 'lenta_dataset.csv.gz'}  ({n} rows)")


if __name__ == "__main__":
    ROOT.mkdir(parents=True, exist_ok=True)
    build_hillstrom()
    build_retailhero()
    build_megafon()
    build_lenta()
