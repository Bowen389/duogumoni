"""
第一优先 ③：拥挤度监控 + 控仓
三个事先设定好的规则（参数不在测试期调，避免过拟合），信号全部只用 T-1 及以前的信息：
  R1 因子失效：模型近 60 日 RankIC 均值 < 0            → 超额暴露降到 50%
  R2 小盘过热：中证1000/沪深300 成交额比的 250 日 z 分数 > 1.5 → 降到 50%
  R3 策略止损：影子策略超额从高点回撤超过 8%，回到 -4% 以内才恢复 → 降到 50%
"降仓"的部分换成中证1000 指数（ETF/期货），所以控制的是**超额风险**，不改变整体仓位。
用法：python experiments/p1_crowding.py --pred both_h1   （默认用第一优先选出的最佳预测）
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.backtest import Backtester  # noqa
from engine.common import DATA, KEY, init_qlib, read_parts  # noqa

ap = argparse.ArgumentParser()
ap.add_argument("--pred", default="retail_h1")
ap.add_argument("--topk", type=int, default=50)
ap.add_argument("--n_drop", type=int, default=2)
a = ap.parse_args()
TEST = ("2023-01-01", "2026-09-18")

bt = Backtester(*TEST)
pred = pd.read_parquet(f"output/preds/{a.pred}.parquet")
base = bt.run(pred, mode="dropout", topk=a.topk, n_drop=a.n_drop)
dates = base.index

# ---------- R1：模型滚动 RankIC（ret1 在 t+2 收盘才知道 → 决策日 T 只能用到 T-3 的 IC）----------
lab = read_parts(os.path.join(DATA, "panel"), columns=KEY + ["ret1"], start="2022-12-01", end=TEST[1])
m = pred.merge(lab, on=KEY)
ic = m.groupby("datetime").apply(lambda x: x["score"].rank().corr(x["ret1"].rank()))
ic60 = ic.rolling(60, min_periods=40).mean().shift(3).reindex(dates)
e1 = pd.Series(np.where(ic60 < 0, 0.5, 1.0), index=dates)

# ---------- R2：小盘成交额占比过热 ----------
init_qlib()
from qlib.data import D  # noqa
amt = {}
for pool in ["csi1000", "csi300"]:
    x = D.features(D.instruments(pool), ["$amount"], "2021-01-01", TEST[1])
    amt[pool] = x.groupby(level="datetime")["$amount"].sum()
share = (amt["csi1000"] / amt["csi300"]).rename("share")
z = ((share - share.rolling(250, min_periods=120).mean()) / share.rolling(250, min_periods=120).std()).shift(1)
e2 = pd.Series(np.where(z.reindex(dates) > 1.5, 0.5, 1.0), index=dates)

# ---------- R3：影子策略超额回撤止损（滞后 1 天）----------
cum = (1 + base["excess"]).cumprod()
dd = (cum / cum.cummax() - 1).shift(1).fillna(0)
state, e3 = 1.0, []
for v in dd.values:
    if state == 1.0 and v < -0.08:
        state = 0.5
    elif state == 0.5 and v > -0.04:
        state = 1.0
    e3.append(state)
e3 = pd.Series(e3, index=dates)

rules = {"不控仓": None, "R1 因子失效": e1, "R2 小盘过热": e2, "R3 回撤止损": e3,
         "R1+R2": np.minimum(e1, e2), "R1+R2+R3": np.minimum(np.minimum(e1, e2), e3)}
rows, curves = [], {}
for k, e in rules.items():
    r = base if e is None else bt.overlay(base, e)
    mt = bt.metrics(r)
    row = {"规则": k, **{c: mt[c] for c in ["超额年化", "信息比率", "超额回撤"]},
           "平均暴露": 1.0 if e is None else float(e.mean())}
    row.update({c: v for c, v in mt.items() if c.startswith("超额20")})
    rows.append(row)
    curves[k] = r
res = pd.DataFrame(rows).set_index("规则")
os.makedirs("output/p1", exist_ok=True)
res.to_csv(f"output/p1/crowding_{a.pred}.csv", encoding="utf-8-sig", float_format="%.4f")
pd.DataFrame({"ic60": ic60, "share_z": z.reindex(dates), "excess_dd": dd}).to_csv(f"output/p1/crowding_signals_{a.pred}.csv")
pd.set_option("display.width", 200)
print(res.round(3).to_string())
bt.plot({k: curves[k] for k in ["不控仓", "R1 因子失效", "R3 回撤止损", "R1+R2+R3"]},
        f"output/p1/crowding_{a.pred}.png", f"P1 拥挤度控仓（{a.pred}，Top{a.topk} 换{a.n_drop}）")
