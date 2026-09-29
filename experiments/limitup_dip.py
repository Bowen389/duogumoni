"""
检验网上流传的“首板低开”策略（中证1000 股票池，日线）
  原版规则：股价处于近 60 日低位（涨停前收盘 < 60 日中位数），昨日首板（前日未涨停），今日低开 3%~4% → 开盘买入 → 次日卖出
  用法：python experiments/limitup_dip.py
成本：佣金万2.5×2 + 印花税（2023-08-28 前千1，之后万5）+ 滑点单边万10（涨停次日的开盘竞价波动大）
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from engine.common import DATA, KEY  # noqa: E402


def load():
    pdir = os.path.join(DATA, "panel")
    parts = []
    for fn in sorted(x for x in os.listdir(pdir) if x.endswith(".parquet")):
        p = pd.read_parquet(os.path.join(pdir, fn), columns=KEY + ["open", "close", "up_lim", "susp", "in_pool", "up_open", "ret_on"],
                            filters=[("datetime", ">=", pd.Timestamp("2015-06-01"))]).sort_values(KEY)
        g = p.groupby("instrument", sort=False)
        up1 = g["up_lim"].shift(1).fillna(False).astype(bool)
        up2 = g["up_lim"].shift(2).fillna(False).astype(bool)
        q = p[KEY].copy()
        for L in (20, 60, 120):   # 涨停前收盘 / 过去 L 日收盘中位数
            q[f"pos{L}"] = (g["close"].shift(2) / g["close"].transform(lambda x: x.shift(2).rolling(L, L // 3).median())).values
        q["low"] = q["pos60"] < 1
        q["gap"] = p["ret_on"]                                        # 今日开盘相对昨收
        q["r_o1"] = g["open"].shift(-1) / p["open"] - 1               # 次日开盘卖
        q["r_c1"] = g["close"].shift(-1) / p["open"] - 1              # 次日收盘卖
        m = up1 & ~up2 & ~p["susp"] & ~p["up_open"] & p["in_pool"]
        parts.append(q[m.values])
    X = pd.concat(parts, ignore_index=True)
    X["年"] = X["datetime"].dt.year
    return X


def cost(d):
    return 0.0005 + np.where(d < pd.Timestamp("2023-08-28"), 0.001, 0.0005) + 0.002


def main():
    pd.set_option("display.width", 250)
    X = load()
    nyr = X["datetime"].dt.year.nunique()
    rows = []
    cases = {"原版：低位 + 首板低开3-4%": X["low"] & X["gap"].between(-0.04, -0.03),
             "首板低开3-4%（不限低位）": X["gap"].between(-0.04, -0.03),
             "低位 + 首板低开2-7%": X["low"] & X["gap"].between(-0.07, -0.02),
             "首板任意低开": X["gap"] < 0,
             "首板高开0-5%": X["gap"].between(0, 0.05),
             "全部首板（次日开盘买）": X["gap"] > -1}
    for name, m in cases.items():
        s = X[m].dropna(subset=["r_c1"])
        net = s["r_c1"] - cost(s["datetime"])
        rows.append(dict(规则=name, 笔数=len(s), 每年笔数=round(len(s) / nyr), 次日开盘卖毛=s["r_o1"].mean(), 次日收盘卖毛=s["r_c1"].mean(),
                         中位数=s["r_c1"].median(), 胜率=(s["r_c1"] > 0).mean(), 扣成本=net.mean()))
    d = pd.DataFrame(rows)
    os.makedirs("output/overnight", exist_ok=True)
    d.to_csv("output/overnight/limitup_dip.csv", index=False, encoding="utf-8-sig", float_format="%.4f")
    for c in ["次日开盘卖毛", "次日收盘卖毛", "中位数", "胜率", "扣成本"]:
        d[c] = d[c].map(lambda v: f"{v:.2%}")
    print(d.to_string(index=False))
    s = X[X["low"] & X["gap"].between(-0.04, -0.03)].dropna(subset=["r_c1"])
    print("\n原版规则分年（次日收盘卖，毛收益均值 / 笔数）")
    print(s.groupby("年")["r_c1"].agg(["mean", "count"]).round(4).T.to_string())
    X = X.dropna(subset=["r_c1"]).copy()
    X["net"] = X["r_c1"] - cost(X["datetime"])
    rows = []
    for L in (20, 60, 120):
        for thr in (0.9, 1.0, 1.1):
            for lo, hi in [(-0.02, -0.01), (-0.03, -0.02), (-0.04, -0.03), (-0.05, -0.04), (-0.07, -0.05), (-0.10, -0.07),
                           (-0.05, -0.02), (-0.07, -0.02)]:
                t = X[(X[f"pos{L}"] < thr) & X["gap"].between(lo, hi)]
                rows.append(dict(低位=f"{L}日中位×{thr}", 低开=f"{-hi:.0%}-{-lo:.0%}", 每笔扣成本=t["net"].mean(), 每年笔数=len(t) / nyr))
    d = pd.DataFrame(rows)
    d.to_csv("output/overnight/limitup_dip_grid.csv", index=False, encoding="utf-8-sig", float_format="%.4f")
    print("\n== 邻域：扣成本后每笔平均收益（%）==")
    print(d.pivot_table(index="低位", columns="低开", values="每笔扣成本").mul(100).round(2).to_string())
    print("正收益格子占比", round((d["每笔扣成本"] > 0).mean(), 2), " 中位数", round(d["每笔扣成本"].median() * 100, 2), "%")
    print("\n== 资金曲线：K 个槽位，T 日开盘买入、T+1 收盘卖出，空闲资金收益为 0 ==")
    for name, sel in {"原版 60日低位 低开3-4%": X[(X.pos60 < 1) & X.gap.between(-0.04, -0.03)],
                      "60日低位 低开2-7%": X[(X.pos60 < 1) & X.gap.between(-0.07, -0.02)],
                      "120日低位 低开2-5%": X[(X.pos120 < 1) & X.gap.between(-0.05, -0.02)]}.items():
        for K in (1, 3):
            ann, mdd, yr, expo = sim(sel, K)
            print(f"{name} 槽位{K}: 年化 {ann:.1%} 回撤 {mdd:.0%} 平均仓位 {expo:.0%} | " +
                  " ".join(f"{y}:{v:+.0%}" for y, v in yr.items()))


def sim(sel, K):
    days = pd.Series(sorted(pd.read_parquet(os.path.join(DATA, "bench.parquet"))["datetime"]))
    days = days[days >= "2015-06-01"].reset_index(drop=True)
    di = {d: i for i, d in enumerate(days)}
    pnl, used = np.zeros(len(days)), np.zeros(len(days))
    ends = []
    for d, grp in sel.sort_values(["datetime", "gap"]).groupby("datetime"):
        i = di.get(d)
        if i is None or i + 1 >= len(days):
            continue
        free = K - sum(1 for e in ends if e >= i)
        for _, r in grp.head(max(free, 0)).iterrows():
            pnl[i + 1] += r["net"] / K
            used[i:i + 2] += 1 / K
            ends.append(i + 1)
    r = pd.Series(pnl, index=days)
    nav = (1 + r).cumprod()
    return (nav.iloc[-1] ** (243 / len(r)) - 1, (nav / nav.cummax() - 1).min(),
            r.groupby(r.index.year).apply(lambda x: (1 + x).prod() - 1), used.mean())


if __name__ == "__main__":
    main()
