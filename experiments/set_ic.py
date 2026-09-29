"""
新因子检验（通用版）：python experiments/set_ic.py em   → output/newdata/em_ic.csv
  默认检验 feat_{set} 中全部因子
  1) 单因子 RankIC（vs T+1 收盘买入持有 5 日收益），ICIR，逐年 IC
  2) 行业中性后的 RankIC（策略实际用的是行业中性打分）
  3) 与原 16 个散户因子的最大截面秩相关（重复度）
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.common import DATA, EXTRA, KEY, read_parts  # noqa

import pyarrow.dataset as ds  # noqa: E402

START = "2016-01-01"
# 2GB 内存：每 3 个交易日抽 1 天（约 800 天，统计上足够）
_d = pd.read_parquet(os.path.join(DATA, "bench.parquet"))["datetime"]
DAYS = _d[_d >= START].iloc[::3]
flt = ds.field("datetime").isin(pd.to_datetime(DAYS).values)
SET = sys.argv[1] if len(sys.argv) > 1 else "em"
beh = read_parts(os.path.join(DATA, f"feat_{SET}"), start=START, filters_extra=flt)
old = read_parts(os.path.join(DATA, "feat_retail"), start=START, filters_extra=flt)
lab = read_parts(os.path.join(DATA, "panel"), columns=KEY + ["in_pool", "ret5", "ret1", "nb"], start=START, filters_extra=flt)
lab = lab[lab["in_pool"] & ~lab["nb"]]            # 次日买不进的剔除（与可成交标签一致）
ind = pd.read_parquet(os.path.join(EXTRA, "industry.parquet"))[["instrument", "industry"]]
df = beh.merge(lab, on=KEY).merge(ind, on="instrument", how="left")
df["industry"] = df["industry"].fillna("UNK")
bcols = [c for c in beh.columns if c not in KEY]
ocols = [c for c in old.columns if c not in KEY]

# 截面排名（一次算好）
g = df.groupby("datetime")
R = pd.DataFrame({c: g[c].rank(pct=True) for c in bcols + ["ret5", "ret1"]})
R["datetime"] = df["datetime"].values
Rn = R.copy()
gi = [df["datetime"], df["industry"]]
for c in bcols:   # 行业中性：排名减去行业均值
    Rn[c] = R[c] - R[c].groupby(gi).transform("mean")


def daily_ic(frame, c, y="ret5"):
    x = frame[["datetime", c, y]].dropna()
    return x.groupby("datetime").apply(lambda z: z[c].corr(z[y]) if len(z) > 100 else np.nan).dropna()


# 与原因子的相关性：每 10 个交易日抽一天
days = sorted(df["datetime"].unique())[::4]
mo = beh[beh["datetime"].isin(days)].merge(old[old["datetime"].isin(days)], on=KEY)
corr = {c: {} for c in bcols}
for d, x in mo.groupby("datetime"):
    rk = x[bcols + ocols].rank()
    cm = rk.corr().loc[bcols, ocols]
    for c in bcols:
        corr[c][d] = cm.loc[c]
rows = []
for c in bcols:
    ic, icn, ic1 = daily_ic(R, c), daily_ic(Rn, c), daily_ic(Rn, c, "ret1")
    if len(ic) < 20:
        continue
    cm = pd.DataFrame(corr[c]).T.mean()
    top = cm.abs().idxmax()
    yr = ic.groupby(ic.index.year).mean()
    rows.append({"因子": c, "RankIC": ic.mean(), "ICIR": ic.mean() / ic.std(), "行业中性IC": icn.mean(),
                 "行业中性ICIR": icn.mean() / icn.std(), "中性ICIR_1日": ic1.mean() / ic1.std(), "覆盖": df[c].notna().mean(), "IC同号年份": f"{int((np.sign(yr) == np.sign(ic.mean())).sum())}/{len(yr)}",
                 "最像的原因子": top, "相关": cm[top], **{f"IC{y}": v for y, v in yr.items()}})
    print(f"{c:10s} IC={ic.mean():+.4f} ICIR={ic.mean() / ic.std():+.2f}  中性ICIR={icn.mean() / icn.std():+.2f}  最像 {top} {cm[top]:+.2f}", flush=True)
res = pd.DataFrame(rows).set_index("因子").sort_values("行业中性ICIR", key=abs, ascending=False)
os.makedirs("output/newdata", exist_ok=True)
res.to_csv(f"output/newdata/{SET}_ic.csv", encoding="utf-8-sig", float_format="%.4f")
pd.set_option("display.width", 300)
print(res.round(3).to_string())
