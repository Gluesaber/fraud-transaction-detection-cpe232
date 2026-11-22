"""Generate a small SYNTHETIC dataset with the same schema as the competition data.

Nothing here comes from the real (course-restricted) dataset. It exists only so the
notebook can be run end to end without access to the competition files.

Usage:  python scripts/make_sample_data.py
"""
import numpy as np
import pandas as pd
from pathlib import Path

rng = np.random.default_rng(0)
OUT = Path(__file__).resolve().parent.parent / "data" / "sample"
OUT.mkdir(parents=True, exist_ok=True)


def make(n, start_id, with_label):
    types = rng.choice(["CASH_OUT", "TRANSFER", "PAYMENT", "CASH_IN", "DEBIT"], n,
                       p=[0.35, 0.10, 0.35, 0.18, 0.02])
    amount = np.round(rng.lognormal(10, 1.5, n), 2)
    src_bal = np.round(rng.lognormal(10.5, 1.8, n), 2)
    dst_bal = np.round(rng.lognormal(10.5, 1.8, n), 2)
    fraud = ((np.isin(types, ["CASH_OUT", "TRANSFER"])) & (rng.random(n) < 0.04)).astype(int)
    # simple, learnable fraud pattern: drain the source account
    amount = np.where(fraud == 1, src_bal, np.minimum(amount, src_bal))
    src_new = np.where(types == "CASH_IN", src_bal + amount, np.maximum(src_bal - amount, 0))
    dst_new = np.where(types == "PAYMENT", 0.0, dst_bal + amount)
    dst_prefix = np.where(types == "PAYMENT", "M", "C")
    df = pd.DataFrame({
        "id": np.arange(start_id, start_id + n),
        "time_ind": rng.integers(1, 744, n),
        "transac_type": types,
        "amount": amount,
        "src_acc": ["C" + str(x) for x in rng.integers(10**8, 10**9, n)],
        "src_bal": src_bal,
        "src_new_bal": np.round(src_new, 2),
        "dst_acc": [p + str(x) for p, x in zip(dst_prefix, rng.integers(10**8, 10**9, n))],
        "dst_bal": dst_bal,
        "dst_new_bal": np.round(dst_new, 2),
    })
    if with_label:
        df["is_fraud"] = fraud
    df["is_flagged_fraud"] = 0
    # inject missing values like the real data (about 10-13% in some columns)
    for col, p in [("transac_type", 0.12), ("amount", 0.13), ("src_bal", 0.05), ("dst_bal", 0.05)]:
        df.loc[rng.random(n) < p, col] = np.nan
    return df


train = make(5000, 0, True)
test = make(1000, 5000, False)
train.to_csv(OUT / "train.csv", index=False)
test.to_csv(OUT / "test.csv", index=False)
print("train", train.shape, "fraud rate", train.is_fraud.mean().round(4))
print("test ", test.shape)
