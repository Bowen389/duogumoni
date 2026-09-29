"""
第三优先：行业 + 市值中性化 —— 把"小盘/行业 beta"从打分里剥离，看剩下的是不是真 alpha
每天截面上：score 先减去行业均值，再对（行业内去均值后的）对数流通市值做回归取残差。
  python experiments/neutralize.py --pred retail_h1 --mode ind+mv
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.common import DATA, EXTRA, KEY, read_parts  # noqa

ap = argparse.ArgumentParser()
ap.add_argument("--pred", required=True)
ap.add_argument("--mode", choices=["ind", "mv", "ind+mv"], default="ind+mv")
a = ap.parse_args()

p = pd.read_parquet(f"output/preds/{a.pred}.parquet")
ind = pd.read_parquet(os.path.join(EXTRA, "industry.parquet"))[["instrument", "industry"]]
df = p.merge(ind, on="instrument", how="left")
if "mv" in a.mode or os.path.isdir(os.path.join(DATA, "feat_extra")):   # 只做行业中性时不强制需要市值数据
    mv = read_parts(os.path.join(DATA, "feat_extra"), columns=KEY + ["LOGMV"], start=p["datetime"].min())
    df = df.merge(mv, on=KEY, how="left")
else:
    df["LOGMV"] = 0.0
df["industry"] = df["industry"].replace("", np.nan).fillna("UNK")
df["LOGMV"] = df["LOGMV"].fillna(df.groupby("datetime")["LOGMV"].transform("median"))
df["s"] = df.groupby("datetime")["score"].rank(pct=True) - 0.5   # 先转排名，抗极值

if "ind" in a.mode:
    df["s"] = df["s"] - df.groupby(["datetime", "industry"])["s"].transform("mean")
    df["m"] = df["LOGMV"] - df.groupby(["datetime", "industry"])["LOGMV"].transform("mean")
else:
    df["m"] = df["LOGMV"] - df.groupby("datetime")["LOGMV"].transform("mean")
if "mv" in a.mode:
    cov = (df["s"] * df["m"]).groupby(df["datetime"]).transform("mean")
    var = (df["m"] ** 2).groupby(df["datetime"]).transform("mean")
    df["s"] = df["s"] - cov / (var + 1e-12) * df["m"]

# 诊断：原始打分与市值的相关性
has_mv = df["LOGMV"].std() > 0
corr0 = np.nan if not has_mv else df.groupby("datetime").apply(lambda x: x["score"].rank().corr(x["LOGMV"].rank())).mean()
corr1 = np.nan if not has_mv else df.groupby("datetime").apply(lambda x: x["s"].rank().corr(x["LOGMV"].rank())).mean()
print(f"打分与流通市值的截面相关：中性化前 {corr0:+.3f} → 中性化后 {corr1:+.3f}")
tag = f"{a.pred}_neu_{a.mode.replace('+', '_')}"
df[KEY + ["s"]].rename(columns={"s": "score"}).to_parquet(f"output/preds/{tag}.parquet", index=False)
print("saved", tag)
