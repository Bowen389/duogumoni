"""
集中策略候选的稳健性：滑点、分年、交易统计、持仓流动性
  python experiments/conc_check.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.common import DATA, KEY, read_parts  # noqa
from experiments.conc_grid import P, timings  # noqa
from experiments.conc_lib import bench_series, load_panel, simulate, stats  # noqa

C = {"C1 k2 M300 MA60 开盘": dict(k=2, M=300, t="指数>MA60", ex="open"),
     "C2 k3 M150 不择时 尾盘": dict(k=3, M=150, t="不择时", ex="close"),
     "C3 k1 M300 MA60 开盘": dict(k=1, M=300, t="指数>MA60", ex="open"),
     "C4 k2 M150 MA60 开盘": dict(k=2, M=150, t="指数>MA60", ex="open"),
     "C5 k3 M300 MA60 开盘": dict(k=3, M=300, t="指数>MA60", ex="open"),
     "C6 k2 M300 MA60 尾盘": dict(k=2, M=300, t="指数>MA60", ex="close")}
tag = sys.argv[1] if len(sys.argv) > 1 else "rb_h5"
rows, yrs = [], []
for per, (s, e, suf) in P.items():
    panel = load_panel(s, e)
    amt = panel.pivot(index="datetime", columns="instrument", values="raw_close")
    bench_full = bench_series(pd.Timestamp("2019-01-01"), pd.Timestamp(e))
    T = timings(bench_full)
    bench = bench_full[bench_full.index >= s]
    sc = pd.read_parquet(f"output/preds/{tag}{suf}.parquet")
    pa = read_parts(os.path.join(DATA, "panel"), columns=KEY + ["amount", "raw_close"], start=s, end=e)
    for name, c in C.items():
        for slip in [0.001, 0.002, 0.003]:
            res = simulate(panel, sc, k=c["k"], M=c["M"], exec_at=c["ex"], timing=T[c["t"]], slip=slip)
            d = stats(res, bench)
            row = dict(区间=per, 方案=name, 滑点=slip, **{k: v for k, v in d.items() if not k.startswith("Y")})
            if slip == 0.001:
                # 交易统计：平均持有天数、年交易次数、持仓股票的成交额与股价
                h = res[["datetime", "hold"]].copy()
                h["hold"] = h["hold"].str.split(",")
                h = h.explode("hold")
                h = h[h["hold"] != ""].rename(columns={"hold": "instrument"})
                h = h.merge(pa, on=KEY, how="left")
                row["持仓日均成交额(亿)"] = h["amount"].median() / 1e5
                row["持仓股价中位"] = h["raw_close"].median()
                row["年换股次数"] = res["turnover"].sum() * 2 * c["k"] / (len(res) / 244) / 2
                yrs.append(dict(区间=per, 方案=name, **{k: v for k, v in d.items() if k.startswith("Y")}))
            rows.append(row)
    del panel, pa
res = pd.DataFrame(rows)
pd.set_option("display.width", 300)
print(res.round(3).to_string())
print(pd.DataFrame(yrs).round(3).to_string())
res.to_csv(f"output/conc/check_{tag}.csv", index=False, encoding="utf-8-sig")
