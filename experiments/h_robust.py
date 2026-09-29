"""
H 的稳健性检验
  1) 权重扫描：散户 : Alpha158 = 1:0 … 0:1（2023-01 ~ 2026-09）
  2) 样本外：用 2015-2018 训练、2019 验证的新模型，在从未参与任何选择的 2020-2022 上重跑整套方案
     需要先训练：
       python -m engine.model --sets retail   --horizon 1 --train 2015-01-01 2018-12-31 --valid 2019-01-01 2019-12-31 --test 2020-01-01 2022-12-31 --tag retail_h1_oos
       python -m engine.model --sets alpha158 --horizon 1 --sample 3 --train ... 同上 --tag a158_h1_oos
       python experiments/neutralize.py --pred retail_h1_oos --mode ind ; 同理 a158_h1_oos
  python experiments/h_robust.py [weights|oos]
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.backtest import Backtester  # noqa
from experiments.candidates_lib import ir, mix, r3, rank_blend  # noqa

P = lambda t: pd.read_parquet(f"output/preds/{t}.parquet")  # noqa
os.makedirs("output/h_robust", exist_ok=True)
pd.set_option("display.width", 250)
what = sys.argv[1] if len(sys.argv) > 1 else "weights"


def row(bt, name, r, rs, split):
    m, ex = bt.metrics(r), r["excess"]
    d = {"方案": name, "超额年化": m["超额年化"], "信息比率": m["信息比率"], "超额回撤": m["超额回撤"],
         "策略年化": m["策略年化"], "策略回撤": m["策略回撤"],
         "前半IR": ir(ex[ex.index < split]), "后半IR": ir(ex[ex.index >= split]), "+千1滑点": bt.metrics(rs)["超额年化"]}
    d.update({k: v for k, v in m.items() if k.startswith("超额20")})
    return d


if what == "weights":
    bt = Backtester("2023-01-01", "2026-09-18")
    a, f = P("retail_h1_neu_ind"), P("a158_h1_neu_ind")
    rows = []
    for w in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.0]:
        p = a if w == 1 else (f if w == 0 else rank_blend([a, f], [w, 1 - w]))
        r = r3(bt, bt.run(p, topk=50, n_drop=2))
        rs = r3(bt, bt.run(p, topk=50, n_drop=2, slip=0.001))
        rows.append(row(bt, f"散户{w:.1f} : A158 {1 - w:.1f}", r, rs, "2025-01-01"))
        print(rows[-1]["方案"], round(rows[-1]["超额年化"], 4), flush=True)
    res = pd.DataFrame(rows).set_index("方案")
    res.to_csv("output/h_robust/weights.csv", encoding="utf-8-sig", float_format="%.4f")
    print(res.round(3).to_string())

if what == "oos":
    bt = Backtester("2020-01-01", "2022-12-31")
    a, f = P("retail_h1_oos_neu_ind"), P("a158_h1_oos_neu_ind")
    a_raw, f_raw = P("retail_h1_oos"), P("a158_h1_oos")
    h = rank_blend([a, f], [0.5, 0.5])
    cands = {"v1 散户·不中性": (a_raw, None), "A 散户·行业中性 (v2)": (a, None),
             "E Alpha158 官方基线": (f_raw, None), "F Alpha158·行业中性": (f, None),
             "G 双账户 A+F": (a, f), "H 打分平均 A+F": (h, None)}
    rows, curves = [], {}
    for k, (p1, p2) in cands.items():
        go = lambda s, p, acct=1e6: r3(bt, bt.run(p, topk=50, n_drop=2, slip=s, account=acct))  # noqa
        if p2 is None:
            r, rs = go(0, p1), go(0.001, p1)
        else:
            r, rs = mix(go(0, p1, 5e5), go(0, p2, 5e5)), mix(go(0.001, p1, 5e5), go(0.001, p2, 5e5))
        rows.append(row(bt, k, r, rs, "2021-07-01"))
        curves[k] = r
        print(k, round(rows[-1]["超额年化"], 4), flush=True)
    for k in [20, 10]:   # 顺带看集中持股在样本外的表现
        ev = 1 if k == 20 else 2
        r = r3(bt, bt.run(h, topk=k, n_drop=1, every=ev))
        rs = r3(bt, bt.run(h, topk=k, n_drop=1, every=ev, slip=0.001))
        rows.append(row(bt, f"H Top{k}", r, rs, "2021-07-01"))
    res = pd.DataFrame(rows).set_index("方案")
    ex = pd.DataFrame({k: v["excess"] for k, v in curves.items()})
    print("\nA 与 F 日超额相关:", round(ex["A 散户·行业中性 (v2)"].corr(ex["F Alpha158·行业中性"]), 3))
    res.to_csv("output/h_robust/oos_2020_2022.csv", encoding="utf-8-sig", float_format="%.4f")
    print(res.round(3).to_string())
    bt.plot({k: curves[k] for k in ["H 打分平均 A+F", "A 散户·行业中性 (v2)", "F Alpha158·行业中性", "v1 散户·不中性"]},
            "output/h_robust/oos_2020_2022.png", "样本外 2020-2022：累计超额（2015-18训练，含 R3）")
