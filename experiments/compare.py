"""通用对比：多个预测文件 × 同一回测设置 → 表格 + 图
  python experiments/compare.py --preds a158_h1,retail_h1,both_h1 --out output/p1/features
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.backtest import Backtester, N_YEAR  # noqa
from engine.common import DATA, KEY, read_parts  # noqa

ap = argparse.ArgumentParser()
ap.add_argument("--preds", required=True)
ap.add_argument("--labels", default="", help="逗号分隔的显示名（可选）")
ap.add_argument("--topk", type=int, default=50)
ap.add_argument("--n_drop", type=int, default=2)
ap.add_argument("--test", nargs=2, default=["2023-01-01", "2026-09-18"])
ap.add_argument("--out", required=True)
ap.add_argument("--ensemble", default="", help="把若干预测按截面排名平均，生成新预测，格式 name=a+b+c")
a = ap.parse_args()

tags = a.preds.split(",")
labels = a.labels.split(",") if a.labels else tags
preds = {t: pd.read_parquet(f"output/preds/{t}.parquet") for t in tags}
if a.ensemble:
    name, parts = a.ensemble.split("=")
    ps = []
    for p in parts.split("+"):
        d = pd.read_parquet(f"output/preds/{p}.parquet")
        d["score"] = d.groupby("datetime")["score"].rank(pct=True)
        ps.append(d.set_index(KEY)["score"])
    ens = pd.concat(ps, axis=1).mean(axis=1).rename("score").reset_index()
    ens.to_parquet(f"output/preds/{name}.parquet", index=False)
    preds[name] = ens
    tags.append(name)
    labels.append(name)

bt = Backtester(*a.test)
lab = read_parts(os.path.join(DATA, "panel"), columns=KEY + ["ret1", "ret5"], start=a.test[0], end=a.test[1])
rows, curves = [], {}
for t, l in zip(tags, labels):
    p = preds[t]
    m = p.merge(lab, on=KEY)
    ric = m.groupby("datetime").apply(lambda x: x["score"].rank().corr(x["ret1"].rank()))
    r = bt.run(p, mode="dropout", topk=a.topk, n_drop=a.n_drop)
    r2 = bt.run(p, mode="dropout", topk=a.topk, n_drop=a.n_drop, slip=0.001)
    mt = bt.metrics(r)
    row = {"模型": l, "RankIC": ric.mean(), "RankICIR": ric.mean() / ric.std(),
           **{k: mt[k] for k in ["超额年化", "信息比率", "超额回撤", "日均换手"]},
           "超额年化(+千1滑点)": bt.metrics(r2)["超额年化"]}
    row.update({k: v for k, v in mt.items() if k.startswith("超额20")})
    rows.append(row)
    curves[l] = r
    print(l, "done", flush=True)
res = pd.DataFrame(rows).set_index("模型")
os.makedirs(os.path.dirname(a.out), exist_ok=True)
res.to_csv(a.out + ".csv", encoding="utf-8-sig", float_format="%.4f")
pd.set_option("display.width", 250)
print(res.round(3).to_string())
bt.plot(curves, a.out + ".png", f"累计超额（扣费后，Top{a.topk} 每日换{a.n_drop}）")
