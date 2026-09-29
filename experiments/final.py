"""
汇总：把三轮优化里被验证有效的改动叠加，得到 v2 策略，并和 v1 对比
  v1 = 散户16因子 + LightGBM + 日频 Top50 每日换5（第一版）
  改动①（P1）Top50 每日只换2 → 降换手
  改动②（P3）行业（+市值）中性化
  改动③（P1）R3 超额回撤止损（回撤>8% 降到半仓超额，回到 -4% 以内恢复）
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.backtest import Backtester, N_YEAR  # noqa

TEST = ("2023-01-01", "2026-09-18")
bt = Backtester(*TEST)


def r3(res, dd_in=-0.08, dd_out=-0.04):
    cum = (1 + res["excess"]).cumprod()
    dd = (cum / cum.cummax() - 1).shift(1).fillna(0)
    st, e = 1.0, []
    for v in dd.values:
        st = 0.5 if (st == 1.0 and v < dd_in) else (1.0 if (st == 0.5 and v > dd_out) else st)
        e.append(st)
    return bt.overlay(res, pd.Series(e, index=res.index))


P = lambda t: pd.read_parquet(f"output/preds/{t}.parquet")  # noqa
runs = {}
runs["v1 第一版 (Top50换5)"] = bt.run(P("retail_h1"), topk=50, n_drop=5)
runs["① +降换手 (换2)"] = bt.run(P("retail_h1"), topk=50, n_drop=2)
runs["①② +行业中性"] = bt.run(P("retail_h1_neu_ind"), topk=50, n_drop=2)
runs["①② +行业+市值中性"] = bt.run(P("retail_h1_neu_ind_mv"), topk=50, n_drop=2)
runs["v2 = ①② 行业中性 + ③ 回撤止损"] = r3(runs["①② +行业中性"])
runs["v2保守版 = 行业市值中性 + 回撤止损"] = r3(runs["①② +行业+市值中性"])

rows = []
for k, r in runs.items():
    m = bt.metrics(r)
    r_s = None
    rows.append({"方案": k, **{c: m[c] for c in ["超额年化", "信息比率", "超额回撤", "策略年化", "策略回撤", "日均换手"]},
                 **{c: v for c, v in m.items() if c.startswith("超额20")}})
res = pd.DataFrame(rows).set_index("方案")

# 滑点敏感性（千1 双边）
slip = {}
for k, (tag, nd) in {"v1 第一版 (Top50换5)": ("retail_h1", 5), "v2 = ①② 行业中性 + ③ 回撤止损": ("retail_h1_neu_ind", 2),
                     "v2保守版 = 行业市值中性 + 回撤止损": ("retail_h1_neu_ind_mv", 2)}.items():
    r = bt.run(P(tag), topk=50, n_drop=nd, slip=0.001)
    if "v2" in k:
        r = r3(r)
    slip[k] = bt.metrics(r)["超额年化"]
res["超额年化(+千1滑点)"] = pd.Series(slip)
os.makedirs("output/final", exist_ok=True)
res.to_csv("output/final/summary.csv", encoding="utf-8-sig", float_format="%.4f")
pd.set_option("display.width", 250)
print(res.round(3).to_string())
bt.plot({k: runs[k] for k in ["v2 = ①② 行业中性 + ③ 回撤止损", "v1 第一版 (Top50换5)", "① +降换手 (换2)", "①② +行业中性"]},
        "output/final/v1_vs_v2.png", "v1 → v2：中证1000 累计超额（扣费后，测试期 2023-01 ~ 2026-09）")
runs["v2 = ①② 行业中性 + ③ 回撤止损"].to_csv("output/final/v2_daily.csv")
