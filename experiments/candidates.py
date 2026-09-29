"""
候选策略横向对比（统一口径：Top50 每日换2、真实涨跌停、一手100股、最低5元、R3 回撤止损）
另外把测试期拆成前后两半，检查稳定性：前半 2023-01~2024-12，后半 2025-01~2026-09
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.backtest import Backtester, N_YEAR  # noqa

TEST = ("2023-01-01", "2026-09-18")
bt = Backtester(*TEST)
P = lambda t: pd.read_parquet(f"output/preds/{t}.parquet")  # noqa


def r3(res, dd_in=-0.08, dd_out=-0.04):
    cum = (1 + res["excess"]).cumprod()
    dd = (cum / cum.cummax() - 1).shift(1).fillna(0)
    st, e = 1.0, []
    for v in dd.values:
        st = 0.5 if (st == 1.0 and v < dd_in) else (1.0 if (st == 0.5 and v > dd_out) else st)
        e.append(st)
    return bt.overlay(res, pd.Series(e, index=res.index))


def mix(a, b, w=0.5):
    m = a.copy()
    for c in ["return", "turnover", "cost"]:
        m[c] = w * a[c] + (1 - w) * b[c]
    m["excess"] = m["return"] - m["bench"]
    return m


def rank_blend(tags, name):
    ps = []
    for t in tags:
        d = P(t)
        d["score"] = d.groupby("datetime")["score"].rank(pct=True)
        ps.append(d.set_index(["datetime", "instrument"])["score"])
    out = pd.concat(ps, axis=1).mean(axis=1).rename("score").reset_index()
    out.to_parquet(f"output/preds/{name}.parquet", index=False)
    return out


run = lambda p, acct=1e6, slip=0.0: bt.run(p, mode="dropout", topk=50, n_drop=2, account=acct, slip=slip)  # noqa

C = {}
C["A 散户·行业中性 (v2)"] = lambda s: r3(run(P("retail_h1_neu_ind"), slip=s))
C["B 散户·行业+市值中性 (v2保守)"] = lambda s: r3(run(P("retail_h1_neu_ind_mv"), slip=s))
C["C 散户·行业中性·可成交标签"] = lambda s: r3(run(P("retail_h1_tl_neu_ind"), slip=s))
C["D 散户+换手涨停龙虎榜·行业中性"] = lambda s: r3(run(P("retail_extra_lhb_tl_neu_ind"), slip=s))
C["E Alpha158 (官方基线)"] = lambda s: r3(run(P("a158_h1"), slip=s))
C["F Alpha158·行业中性"] = lambda s: r3(run(P("a158_h1_neu_ind"), slip=s))
C["G 双账户: A + F 各50万"] = lambda s: mix(r3(run(P("retail_h1_neu_ind"), 5e5, s)), r3(run(P("a158_h1_neu_ind"), 5e5, s)))
blend = rank_blend(["retail_h1_neu_ind", "a158_h1_neu_ind"], "blend_retail_a158_neu")
C["H 单账户: A 与 F 打分平均"] = lambda s: r3(run(blend, slip=s))

rows, curves = [], {}
for k, fn in C.items():
    r = fn(0.0)
    m = bt.metrics(r)
    h1, h2 = r[r.index < "2025-01-01"]["excess"], r[r.index >= "2025-01-01"]["excess"]
    ir = lambda x: x.mean() / x.std() * np.sqrt(N_YEAR)  # noqa
    row = {"候选": k, **{c: m[c] for c in ["超额年化", "信息比率", "超额回撤", "策略年化", "策略回撤"]},
           "前半IR": ir(h1), "后半IR": ir(h2), "超额年化(+千1滑点)": bt.metrics(fn(0.001))["超额年化"]}
    row.update({c: v for c, v in m.items() if c.startswith("超额20")})
    rows.append(row)
    curves[k] = r
    print(k, "done", flush=True)

res = pd.DataFrame(rows).set_index("候选")
# 两策略日超额相关
ex = pd.DataFrame({k: v["excess"] for k, v in curves.items()})
os.makedirs("output/candidates", exist_ok=True)
res.to_csv("output/candidates/summary.csv", encoding="utf-8-sig", float_format="%.4f")
ex.corr().to_csv("output/candidates/excess_corr.csv", encoding="utf-8-sig", float_format="%.3f")
pd.set_option("display.width", 260)
print(res.round(3).to_string())
print("\nA 与 F 日超额相关:", round(ex.iloc[:, 0].corr(ex.iloc[:, 5]), 3))
bt.plot({k: curves[k] for k in ["G 双账户: A + F 各50万", "A 散户·行业中性 (v2)", "F Alpha158·行业中性", "B 散户·行业+市值中性 (v2保守)"]},
        "output/candidates/candidates.png", "候选策略：累计超额（扣费后，均含 R3 回撤止损）")
