"""
提高收益的方案对比：每个方案都在 研究期 2023-01~2026-09 和 样本外 2020-01~2022-12 两段上回测
（样本外的模型只用 2015-2018 训练，这 3 年没有参与任何选择）

方案
  0  基线 A_top50：中证1000 / 散户16因子 / 行业中性 / Top50 每天换2 / 尾盘成交 / R3
  1  去掉 R3
  2  多周期融合：1 日标签模型 + 5 日标签模型打分平均
  3a 开盘成交（同一模型，T 收盘出信号 → T+1 开盘成交）
  3b 开盘成交 + 用"开盘到开盘"标签重训的模型
  5  融资加杠杆 1.3 倍（融资利率按 6%/年）
  6  加入 12 个"散户长期犯错"因子（处置效应/锚定/长期过度反应/注意力/恐慌，retail_factors.BEHAVIOR_FACTORS）
  以及上述方案的组合、Top20 版本
  python experiments/boost.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import engine.backtest as EB  # noqa
from experiments.candidates_lib import r3, rank_blend  # noqa

N_YEAR = 238
PER = {"研究期": ("2023-01-01", "2026-09-18", ""), "样本外": ("2020-01-01", "2022-12-31", "_oos")}
P = lambda t: pd.read_parquet(f"output/preds/{t}.parquet")  # noqa
has = lambda t: os.path.exists(f"output/preds/{t}.parquet")  # noqa

# 方案定义：name -> dict(pred=函数(sfx)->DataFrame 或 None, pool, exec, topk, n_drop, r3, lev)
S = {}
base = dict(pool="csi1000", exec="close", topk=50, n_drop=2, r3=True, lev=1.0)
S["0 基线 A_top50"] = dict(base, pred=lambda x: P(f"retail_h1{x}_neu_ind"))
S["1 去掉 R3"] = dict(base, pred=S["0 基线 A_top50"]["pred"], r3=False)
S["2 多周期融合 h1+h5"] = dict(base, pred=lambda x: rank_blend([P(f"retail_h1{x}_neu_ind"), P(f"retail_h5{x}_neu_ind")], [1, 1])
                           if has(f"retail_h5{x}_neu_ind") else None)
S["3a 开盘成交（原模型）"] = dict(base, pred=S["0 基线 A_top50"]["pred"], exec="open")
S["3b 开盘成交 + 开盘标签模型"] = dict(base, pred=lambda x: P(f"retail_h1o{x}_neu_ind") if has(f"retail_h1o{x}_neu_ind") else None,
                                exec="open")
S["5 杠杆 1.3 倍"] = dict(base, pred=S["0 基线 A_top50"]["pred"], lev=1.3)
S["6 +12个长期犯错因子"] = dict(base, pred=lambda x: P(f"rb_h1{x}_neu_ind") if has(f"rb_h1{x}_neu_ind") else None)
S["6+3a 新因子 + 开盘成交"] = dict(S["6 +12个长期犯错因子"], exec="open")
S["0' 基线 A_top20"] = dict(base, pred=S["0 基线 A_top50"]["pred"], topk=20, n_drop=1)
S["6' 新因子 Top20"] = dict(S["6 +12个长期犯错因子"], topk=20, n_drop=1)
S["3b' 开盘标签 Top20"] = dict(S["3b 开盘成交 + 开盘标签模型"], topk=20, n_drop=1)

only = sys.argv[1:]  # 可只跑指定编号，如 python experiments/boost.py 0 3a
BT = {}


def get_bt(pool, per):
    k = (pool, per)
    if k not in BT:
        s, e, _ = PER[per]
        BT[k] = EB.Backtester(s, e)
    return BT[k]


def lever(res, L, rate=0.06):
    out = res.copy()
    out["return"] = L * res["return"] - (L - 1) * rate / N_YEAR
    out["excess"] = out["return"] - out["bench"]
    return out


rows = []
for name, c in S.items():
    if only and name.split()[0] not in only:
        continue
    row = {"方案": name}
    for per, (s, e, sfx) in PER.items():
        try:
            pred = c["pred"](sfx)
        except FileNotFoundError:
            pred = None
        if pred is None:
            continue
        bt = get_bt(c["pool"], per)
        out = []
        for slip in (0.0, 0.001):
            r = bt.run(pred, mode="dropout", topk=c["topk"], n_drop=c["n_drop"], slip=slip, exec_at=c["exec"])
            if c["r3"]:
                r = r3(bt, r)
            if c["lev"] != 1:
                r = lever(r, c["lev"])
            out.append(r)
        m, ms = bt.metrics(out[0]), bt.metrics(out[1])
        row.update({f"{per}超额": m["超额年化"], f"{per}IR": m["信息比率"], f"{per}超额回撤": m["超额回撤"],
                    f"{per}策略年化": m["策略年化"], f"{per}策略回撤": m["策略回撤"], f"{per}换手": m["日均换手"],
                    f"{per}+千1滑点": ms["超额年化"]})
        out[0].to_csv(f"output/boost/daily_{name.split()[0]}_{'is' if per == '研究期' else 'oos'}.csv", float_format="%.6f")
    rows.append(row)
    print(name, {k: round(v, 3) for k, v in row.items() if k.endswith("超额") or k.endswith("IR")}, flush=True)

os.makedirs("output/boost", exist_ok=True)
res = pd.DataFrame(rows).set_index("方案")
fn = "output/boost/summary.csv"
if only and os.path.exists(fn):
    old = pd.read_csv(fn, index_col=0)
    res = pd.concat([old.drop(index=[i for i in res.index if i in old.index]), res])
    res = res.reindex([k for k in S if k in res.index])
res.to_csv(fn, encoding="utf-8-sig", float_format="%.4f")
pd.set_option("display.width", 300)
cols = [c for c in ["研究期超额", "研究期IR", "研究期超额回撤", "研究期策略年化", "研究期策略回撤", "研究期+千1滑点",
                    "样本外超额", "样本外IR", "样本外超额回撤", "样本外策略年化", "样本外+千1滑点"] if c in res]
print(res[cols].round(3).to_string())
