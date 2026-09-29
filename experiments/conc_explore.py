"""
集中持股摸底：每天只买打分最高的 k 只（T+1 收盘买、持有 1 天），看绝对收益
  python experiments/conc_explore.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.common import DATA, KEY, read_parts  # noqa
from experiments.candidates_lib import rank_blend  # noqa

P = {"研究期": ("2023-01-01", "2026-09-18", ""), "样本外": ("2020-01-01", "2022-12-31", "_oos")}
rows = []
for per, (s, e, suf) in P.items():
    lab = read_parts(os.path.join(DATA, "panel"), columns=KEY + ["in_pool", "nb", "ret1", "reto1", "nbo"], start=s, end=e)
    lab = lab[lab.in_pool]
    fr = read_parts(os.path.join(DATA, "feat_retail"), columns=KEY + ["VOL20", "MAX20", "REV5", "AMIHUD20"], start=s, end=e)
    pr = pd.read_parquet(f"output/preds/retail_h1{suf}_neu_ind.parquet")
    pb = pd.read_parquet(f"output/preds/rb_h1{suf}_neu_ind.parquet")
    raw = pd.read_parquet(f"output/preds/rb_h1{suf}.parquet").rename(columns={"score": "raw"})
    ae = rank_blend([pr, pb], [1, 1]).rename(columns={"score": "ae"})
    df = lab.merge(ae, on=KEY).merge(raw, on=KEY, how="left").merge(fr, on=KEY, how="left")
    df = df[~df.nb]
    g = df.groupby("datetime")
    df["ae_r"] = g["ae"].rank(ascending=False)
    df["raw_r"] = g["raw"].rank(ascending=False)
    df["vol_r"] = g["VOL20"].rank(pct=True)
    # 高分里挑低波：AE 前 10% 中 VOL20 最低
    df["ae_pct"] = g["ae"].rank(pct=True)
    df["lv"] = np.where(df.ae_pct >= 0.9, -df["VOL20"], -np.inf)
    df["lv_r"] = df.groupby("datetime")["lv"].rank(ascending=False)
    pool = g["ret1"].mean()
    for name, col in [("AE 排名", "ae_r"), ("rb 原始分(未中性)", "raw_r"), ("AE前10%里最低波", "lv_r")]:
        for k in [1, 3, 5, 10]:
            x = df[df[col] <= k].groupby("datetime")["ret1"].mean().reindex(pool.index).fillna(0)
            net = x - 0.0017   # 每天全换：买万5+卖千1+少量滑点
            rows.append(dict(区间=per, 规则=name, k=k, 日均毛收益=x.mean(), 股票池日均=pool.mean(),
                             胜率=(x > 0).mean(), 日波动=x.std(),
                             年化_日换全仓=(1 + net).prod() ** (244 / len(net)) - 1,
                             最大回撤=((1 + net).cumprod() / (1 + net).cumprod().cummax() - 1).min()))
    del df, lab, fr
res = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print(res.round(4).to_string())
os.makedirs("output/conc", exist_ok=True)
res.to_csv("output/conc/explore.csv", index=False, encoding="utf-8-sig")
