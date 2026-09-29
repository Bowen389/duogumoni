"""
持股数量敏感性：Top 50 / 30 / 20 / 10 / 5
两种换仓方式：
  - 同换手：日均约换 4% 仓位（50只每天换2只的水平）→ 20只每天1只(5%)、10只每2天1只、5只每4天1只
  - 每天换1只：持仓越少，换手越高
候选 A（散户·行业中性）和 H（A 与 Alpha158·行业中性打分平均），全部带 R3，100 万账户
  python experiments/topk.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.backtest import Backtester, N_YEAR  # noqa
from experiments.candidates_lib import r3  # noqa

bt = Backtester("2023-01-01", "2026-09-18")
PRED = {"A 散户·行业中性": "retail_h1_neu_ind", "H 散户+A158打分平均": "blend_retail_a158_neu"}
# (topk, n_drop, every)
GRID = {"同换手": [(50, 2, 1), (30, 1, 1), (20, 1, 1), (10, 1, 2), (5, 1, 4)],
        "每天换1只": [(50, 1, 1), (30, 1, 1), (20, 1, 1), (10, 1, 1), (5, 1, 1)]}

rows, curves = [], {}
for name, tag in PRED.items():
    p = pd.read_parquet(f"output/preds/{tag}.parquet")
    done = {}
    for how, grid in GRID.items():
        for k, nd, ev in grid:
            key = (k, nd, ev)
            if key not in done:
                r0 = bt.run(p, mode="dropout", topk=k, n_drop=nd, every=ev)
                r1 = bt.run(p, mode="dropout", topk=k, n_drop=nd, every=ev, slip=0.001)
                done[key] = (r3(bt, r0), r3(bt, r1), r0)
            r, rs, raw = done[key]
            m, ms = bt.metrics(r), bt.metrics(rs)
            ex = r["excess"]
            yearly = [m.get(f"超额{y}", np.nan) for y in (2023, 2024, 2025, 2026)]
            rows.append({"模型": name, "换仓方式": how, "持股": k, "规则": f"每{ev}天换{nd}只",
                         "超额年化": m["超额年化"], "信息比率": m["信息比率"], "超额回撤": m["超额回撤"],
                         "策略年化": m["策略年化"], "策略回撤": m["策略回撤"], "日均换手": m["日均换手"],
                         "+千1滑点": ms["超额年化"], "最差年": min(yearly),
                         "2023": yearly[0], "2024": yearly[1], "2025": yearly[2], "2026": yearly[3],
                         "单票最大日亏损": np.nan})
            if how == "同换手":
                curves[f"{name[:1]} Top{k}"] = r
    print(name, "done", flush=True)

res = pd.DataFrame(rows).drop(columns="单票最大日亏损")
os.makedirs("output/topk", exist_ok=True)
res.to_csv("output/topk/summary.csv", index=False, encoding="utf-8-sig", float_format="%.4f")
pd.set_option("display.width", 250)
print(res.round(3).to_string(index=False))
bt.plot({k: curves[k] for k in ["H Top50", "H Top20", "H Top10", "H Top5"]},
        "output/topk/topk_H.png", "H：不同持股数的累计超额（同换手，含 R3）")
bt.plot({k: curves[k] for k in ["A Top50", "A Top20", "A Top10", "A Top5"]},
        "output/topk/topk_A.png", "A：不同持股数的累计超额（同换手，含 R3）")
