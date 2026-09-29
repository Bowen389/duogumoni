"""第一优先 ①：降换手 —— 对比不同调仓方式（预测来自 retail_h1 / retail_h5）"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.backtest import Backtester, N_YEAR  # noqa

TEST = ("2023-01-01", "2026-09-18")
bt = Backtester(*TEST)
P = {t: pd.read_parquet(f"output/preds/{t}.parquet") for t in ["retail_h1", "retail_h5"]}

CONFIGS = [
    ("A 基线: 日频 Top50 换5 (h1)", "retail_h1", dict(mode="dropout", topk=50, n_drop=5)),
    ("日频 Top30 换2 (h1)", "retail_h1", dict(mode="dropout", topk=30, n_drop=2)),
    ("日频 Top50 换2 (h1)", "retail_h1", dict(mode="dropout", topk=50, n_drop=2)),
    ("周频 Top50 缓冲1.0 (h1, 标签不匹配)", "retail_h1", dict(mode="periodic", topk=50, rebalance=5, buffer=1.0)),
    ("周频 Top50 缓冲1.0 (h5)", "retail_h5", dict(mode="periodic", topk=50, rebalance=5, buffer=1.0)),
    ("周频 Top50 缓冲1.5 (h5)", "retail_h5", dict(mode="periodic", topk=50, rebalance=5, buffer=1.5)),
    ("周频 Top50 缓冲2.0 (h5)", "retail_h5", dict(mode="periodic", topk=50, rebalance=5, buffer=2.0)),
    ("双周 Top50 缓冲1.5 (h5)", "retail_h5", dict(mode="periodic", topk=50, rebalance=10, buffer=1.5)),
]

rows, curves = [], {}
for name, tag, cfg in CONFIGS:
    for slip in (0.0, 0.001):
        r = bt.run(P[tag], slip=slip, **cfg)
        m = bt.metrics(r)
        if slip == 0.0:
            curves[name] = r
            se = r["excess"].std() * np.sqrt(N_YEAR) / np.sqrt(len(r) / N_YEAR)
            row = {"方案": name, **{k: m[k] for k in ["超额年化", "信息比率", "超额回撤", "日均换手", "年化成本"]},
                   "±1σ误差": se}
            row.update({k: v for k, v in m.items() if k.startswith("超额20")})
        else:
            row["超额年化(+千1滑点)"] = m["超额年化"]
            rows.append(row)
    print(name, "done", flush=True)

res = pd.DataFrame(rows).set_index("方案")
os.makedirs("output/p1", exist_ok=True)
res.to_csv("output/p1/turnover.csv", encoding="utf-8-sig", float_format="%.4f")
pd.set_option("display.width", 250)
print(res.round(3).to_string())
bt.plot({k: curves[k] for k in [CONFIGS[0][0], CONFIGS[1][0], CONFIGS[5][0]]}, "output/p1/turnover.png",
        "P1 turnover: cumulative excess (after cost)")
