"""
集中持股模拟器：持 k 只等权；持仓排名掉出前 M 名才卖（缓冲，降换手）；可空仓（择时）
  信号日 t-1 收盘后计算 → 第 t 日成交（close: 尾盘集合竞价；open: 开盘集合竞价）
  close 成交：涨停/停牌买不进，跌停/停牌卖不出
  open  成交：开盘一字涨停买不进，开盘跌停卖不出；当日买入的只赚日内，当日卖出的只赚隔夜
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.common import DATA, KEY, read_parts  # noqa

PANEL_COLS = KEY + ["in_pool", "susp", "ret", "ret_on", "ret_id", "up_lim", "dn_lim", "up_open", "dn_open", "raw_close"]


def load_panel(start, end):
    p = read_parts(os.path.join(DATA, "panel"), columns=PANEL_COLS, start=start, end=end)
    return p


def bench_series(start, end):
    b = pd.read_parquet(os.path.join(DATA, "bench.parquet")).set_index("datetime")
    return b[(b.index >= start) & (b.index <= end)].iloc[:, 0]


def simulate(panel, score, k=3, M=30, exec_at="close", timing=None, buy_cost=0.0005, sell_cost=0.0010, slip=0.001,
             max_hold=0, min_score=None):
    """score: DataFrame(datetime, instrument, score) —— 越大越好，datetime = 信号日
    timing: pd.Series(bool, index=信号日)；False → 下一交易日清仓空仓
    max_hold>0: 持有满 max_hold 天强制换
    min_score: 分数低于此值的股票不买（可让策略在没有好机会时部分空仓）
    返回 DataFrame(datetime, ret, n_hold, turnover)"""
    days = np.array(sorted(panel["datetime"].unique()))
    insts = np.array(sorted(panel["instrument"].unique()))
    di = pd.Series(np.arange(len(days)), index=days)
    ii = pd.Series(np.arange(len(insts)), index=insts)
    r_ = di.reindex(panel["datetime"].values).values
    c_ = ii.reindex(panel["instrument"].values).values
    mats = {}
    for c in ["ret", "ret_on", "ret_id"]:
        m = np.zeros((len(days), len(insts)), dtype="float32")
        m[r_, c_] = panel[c].fillna(0).values
        mats[c] = m
    for c, dflt in [("up_lim", False), ("dn_lim", False), ("up_open", False), ("dn_open", False), ("susp", True),
                    ("in_pool", False)]:
        m = np.full((len(days), len(insts)), dflt, dtype=bool)
        m[r_, c_] = panel[c].fillna(dflt).values.astype(bool)
        mats[c] = m
    have = np.zeros((len(days), len(insts)), dtype=bool)
    have[r_, c_] = True
    # 排名（信号日）
    sc = score[score["datetime"].isin(days) & score["instrument"].isin(insts)].copy()
    sc["rk"] = sc.groupby("datetime")["score"].rank(ascending=False, method="first")
    top = sc[sc["rk"] <= max(M, k + 20)].sort_values(["datetime", "rk"])
    rank_by_day, score_by_day, order_by_day = {}, {}, {}
    for d, gdf in top.groupby("datetime"):
        idx = ii.reindex(gdf["instrument"].values).values
        rank_by_day[d] = dict(zip(idx, gdf["rk"].values))
        score_by_day[d] = dict(zip(idx, gdf["score"].values))
        order_by_day[d] = list(idx)
    hold = {}   # 股票序号 -> 持有天数
    out = []
    for i in range(1, len(days)):
        t, s = days[i], days[i - 1]          # 在 t 日成交，用 s 日信号
        rk = rank_by_day.get(s, {})
        on = True if timing is None else bool(timing.get(s, True))
        sells, keeps = [], []
        for x, h in hold.items():
            want_sell = (not on) or rk.get(x, 1e9) > M or (max_hold and h >= max_hold) or not mats["in_pool"][i, x]
            blocked = mats["susp"][i, x] or (mats["dn_lim"][i, x] if exec_at == "close" else mats["dn_open"][i, x])
            (sells if want_sell and not blocked else keeps).append(x)
        n_free = k - len(keeps)
        buys = []
        if on and n_free > 0:
            for x in order_by_day.get(s, []):
                if len(buys) >= n_free:
                    break
                if x in hold:
                    continue
                if min_score is not None and score_by_day[s][x] < min_score:
                    break
                blocked = mats["susp"][i, x] or (mats["up_lim"][i, x] if exec_at == "close" else mats["up_open"][i, x])
                if not blocked and have[i, x]:
                    buys.append(x)
        w = 1.0 / k
        r = 0.0
        if exec_at == "close":
            for x in keeps + sells:
                r += w * mats["ret"][i, x]
        else:
            for x in keeps:
                r += w * mats["ret"][i, x]
            for x in sells:
                r += w * mats["ret_on"][i, x]
            for x in buys:
                r += w * mats["ret_id"][i, x]
        cost = w * (len(sells) * (sell_cost + slip) + len(buys) * (buy_cost + slip))
        r -= cost
        hold = {x: hold[x] + 1 for x in keeps}
        for x in buys:
            hold[x] = 1
        out.append((t, r, len(hold), w * (len(sells) + len(buys)) / 2, ",".join(insts[list(hold)])))
    return pd.DataFrame(out, columns=["datetime", "ret", "n_hold", "turnover", "hold"])


def stats(res, bench=None):
    r = res.set_index("datetime")["ret"]
    nav = (1 + r).cumprod()
    yrs = len(r) / 244
    d = dict(年化=nav.iloc[-1] ** (1 / yrs) - 1, 最大回撤=(nav / nav.cummax() - 1).min(),
             夏普=r.mean() / r.std() * np.sqrt(244) if r.std() > 0 else np.nan,
             日换手=res["turnover"].mean(), 持仓天数占比=(res["n_hold"] > 0).mean())
    d["卡玛"] = d["年化"] / abs(d["最大回撤"]) if d["最大回撤"] < 0 else np.nan
    if bench is not None:
        b = bench.reindex(r.index).fillna(0)
        d["基准年化"] = (1 + b).prod() ** (1 / yrs) - 1
    yr = r.groupby(r.index.year).apply(lambda x: (1 + x).prod() - 1)
    d.update({f"Y{y}": v for y, v in yr.items()})
    return d
