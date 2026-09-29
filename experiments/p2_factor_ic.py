"""第二优先：新因子单因子检验（RankIC vs 未来5日收益，按年分解）"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.common import DATA, KEY, read_parts  # noqa

f = read_parts(os.path.join(DATA, "feat_extra"), start="2016-01-01")
lab = read_parts(os.path.join(DATA, "panel"), columns=KEY + ["in_pool", "ret5", "up_lim"], start="2016-01-01")
lab = lab[lab["in_pool"]]
df = f.merge(lab, on=KEY)
df = df[~df["up_lim"]]            # 当日涨停买不进，剔除（与回测口径一致）
cols = [c for c in f.columns if c not in KEY + ["IS_ST"]]
rows = []
for c in cols:
    sub = df[["datetime", c, "ret5"]].dropna()
    ic = sub.groupby("datetime").apply(lambda x: x[c].rank().corr(x["ret5"].rank()) if x[c].nunique() > 1 else np.nan).dropna()
    if len(ic) < 100:
        continue
    yr = ic.groupby(ic.index.year).mean()
    rows.append({"因子": c, "RankIC": ic.mean(), "ICIR": ic.mean() / ic.std(), "天数": len(ic),
                 **{f"IC_{y}": v for y, v in yr.items()}})
    print(f"{c:12s} RankIC={ic.mean():+.4f} ICIR={ic.mean() / ic.std():+.2f}", flush=True)
res = pd.DataFrame(rows).set_index("因子").sort_values("ICIR", key=abs, ascending=False)
os.makedirs("output/p2", exist_ok=True)
res.to_csv("output/p2/extra_factor_ic.csv", encoding="utf-8-sig", float_format="%.4f")
pd.set_option("display.width", 250)
print(res.round(3).to_string())
