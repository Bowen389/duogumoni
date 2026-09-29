"""
散户犯错"事件"研究（中证1000）：事件日 t 收盘后发出信号，t+1 成交，看持有 h 天的收益
  入场：t+1 收盘（close）或 t+1 开盘（open）；剔除入场时涨停/一字涨停买不进的
  超额 = 事件收益 - 同期股票池平均
  分三段：2016-19 / 2020-22 / 2023-26
  python experiments/conc_events.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.common import DATA, KEY, read_parts  # noqa

cols = KEY + ["open", "high", "low", "close", "volume", "amount", "in_pool", "susp", "up_lim", "dn_lim", "up_open",
              "dn_open", "change", "raw_close"]
H = [1, 3, 5, 10]


def part_events(p):
    p = p.sort_values(["instrument", "datetime"]).reset_index(drop=True)
    g = p.groupby("instrument", sort=False)
    c, o, h, lo, v = p["close"], p["open"], p["high"], p["low"], p["volume"]
    sh = lambda s, n: s.groupby(p["instrument"], sort=False).shift(n)  # noqa: E731
    roll = lambda s, n, fn="mean": s.groupby(p["instrument"], sort=False).transform(  # noqa: E731
        lambda x: getattr(x.rolling(n, min_periods=max(2, n // 2)), fn)())
    ret = c / sh(c, 1) - 1
    p["ret"] = ret
    # ---------- 前向收益 ----------
    c1, o1 = sh(c, -1), sh(o, -1)
    for hh in H:
        p[f"fc{hh}"] = sh(c, -1 - hh) / c1 - 1           # t+1 收盘买，持 h 天
        p[f"fo{hh}"] = sh(o, -1 - hh) / o1 - 1           # t+1 开盘买，t+1+h 开盘卖
    p["nb_c"] = sh(p["up_lim"] | p["susp"], -1).fillna(True).astype(bool)
    p["nb_o"] = sh(p["up_open"] | p["susp"], -1).fillna(True).astype(bool)
    # ---------- 辅助量 ----------
    vma20 = roll(v, 20)
    r5 = c / sh(c, 5) - 1
    r20 = c / sh(c, 20) - 1
    r60 = c / sh(c, 60) - 1
    dn3 = roll(p["dn_lim"].astype(float), 3, "sum") * 3 / 3
    up_cnt60 = roll(p["up_lim"].astype(float), 60, "sum")
    dn_cnt5 = p["dn_lim"].astype(float).groupby(p["instrument"], sort=False).transform(lambda x: x.rolling(5, 1).sum())
    up_cnt5 = p["up_lim"].astype(float).groupby(p["instrument"], sort=False).transform(lambda x: x.rolling(5, 1).sum())
    prev_up = sh(p["up_lim"], 1).fillna(False).astype(bool)
    prev_dn = sh(p["dn_lim"], 1).fillna(False).astype(bool)
    rng = (h - lo) / sh(c, 1)
    lshadow = (np.minimum(o, c) - lo) / sh(c, 1)
    gap = o / sh(c, 1) - 1
    hi250 = roll(c, 250, "max")
    vol20 = roll(ret, 20, "std")

    E = {
        # 恐慌类：散户恐慌抛售后的反弹
        "连续跌停后打开（前2天跌停，今天不跌停）": (dn_cnt5 >= 2) & prev_dn & ~p["dn_lim"],
        "跌停当日（首个）": p["dn_lim"] & ~prev_dn,
        "5日暴跌>25%": r5 < -0.25,
        "5日暴跌>25%且今日收长下影": (r5 < -0.25) & (lshadow > 0.04),
        "低开>5%后收红（恐慌开盘被接）": (gap < -0.05) & (c > o) & (ret > 0),
        "放量(3倍)大跌>7%非跌停": (ret < -0.07) & ~p["dn_lim"] & (v > 3 * vma20),
        "20日跌>30%": r20 < -0.30,
        # 追涨类：散户追涨后的回落/延续
        "首板（60日首个涨停）": p["up_lim"] & (up_cnt60 == 1),
        "二连板": p["up_lim"] & prev_up & (up_cnt5 == 2),
        "炸板（触及涨停未封住）": (h >= sh(p["raw_close"], 1) * 0 + h) & False,   # 占位，下面单独定义
        "涨停后次日低开>3%": prev_up & (gap < -0.03),
        "放量(3倍)大涨>7%非涨停": (ret > 0.07) & ~p["up_lim"] & (v > 3 * vma20),
        "创250日新高": c >= hi250,
        "创250日新高且低波动": (c >= hi250) & (vol20 < vol20.groupby(p["datetime"]).transform("median")),
        "60日涨>60%": r60 > 0.6,
        # 无人问津：散户忽视
        "缩量(0.4倍)横盘": (v < 0.4 * vma20) & (r20.abs() < 0.05),
    }
    # 炸板：盘中触及涨停但收盘没封住
    raw_h = h / p["raw_close"] * p["raw_close"]  # 复权与否不影响比例
    lim = np.where(p["instrument"].str[2:5].isin(["300", "301", "688"]), 0.2, 0.1)
    prev_close = sh(c, 1)
    touched = (h / prev_close - 1) >= (lim - 0.002)
    E["炸板（触及涨停未封住）"] = touched & ~p["up_lim"]
    E["炸板且收跌"] = touched & ~p["up_lim"] & (ret < 0)

    return p, E


PER = [pd.Timestamp("2015-12-31"), pd.Timestamp("2019-12-31"), pd.Timestamp("2022-12-31"), pd.Timestamp("2026-12-31")]
ev_rows, pool_parts = [], []
pdir = os.path.join(DATA, "panel")
for fn in sorted(os.listdir(pdir)):
    p = pd.read_parquet(os.path.join(pdir, fn), columns=cols)
    p = p[p["datetime"] >= "2015-06-01"]
    p, E = part_events(p)
    m = p["in_pool"] & (p["datetime"] >= "2016-01-01") & (p["datetime"] <= "2026-09-10")
    fwd = [f"fc{hh}" for hh in H] + [f"fo{hh}" for hh in H]
    pool_parts.append(p.loc[m].groupby("datetime")[fwd].agg(["sum", "count"]))
    for name, ev in E.items():
        ev = ev.fillna(False).astype(bool) & m
        x = p.loc[ev, ["datetime", "instrument", "nb_c", "nb_o"] + fwd].copy()
        x["事件"] = name
        ev_rows.append(x)
    print(fn, flush=True)
    del p, E
PL = pd.concat(pool_parts).groupby(level=0).sum()
EV = pd.concat(ev_rows, ignore_index=True)
EV.to_parquet("/tmp/conc_events.parquet")
rows = []
for name, x in EV.groupby("事件", sort=False):
    per = pd.cut(x["datetime"], PER, labels=["16-19", "20-22", "23-26"])
    for mode, nbc in [("close", "nb_c"), ("open", "nb_o")]:
        sel = ~x[nbc]
        if sel.sum() < 30:
            continue
        d = dict(事件=name, 入场=mode, 次数=int(sel.sum()), 年均次数=round(sel.sum() / 10.7, 0),
                 买不进比例=float(x[nbc].mean()))
        for hh in H:
            col = ("fc" if mode == "close" else "fo") + str(hh)
            pm = (PL[(col, "sum")] / PL[(col, "count")]).reindex(x.loc[sel, "datetime"]).values
            r = x.loc[sel, col]
            ex = r - pm
            d[f"h{hh}超额"] = ex.mean()
            if hh in (1, 5):
                d[f"h{hh}胜率"] = (r > 0).mean()
                for pp in ["16-19", "20-22", "23-26"]:
                    d[f"h{hh}_{pp}"] = ex[(per[sel] == pp).values].mean()
        rows.append(d)
res = pd.DataFrame(rows)
os.makedirs("output/conc", exist_ok=True)
res.to_csv("output/conc/events.csv", index=False, encoding="utf-8-sig", float_format="%.4f")
pd.set_option("display.width", 300)
print(res.round(4).to_string())
