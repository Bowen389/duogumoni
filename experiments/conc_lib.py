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

PANEL_COLS = KEY + ["in_pool", "susp", "ret", "ret_on", "ret_id", "up_lim", "dn_lim", "up_open", "dn_open", "raw_close", "raw_open", "factor", "buy_ok", "is_st", "is_delisted", "status_known"]


def load_panel(start, end):
    p = read_parts(os.path.join(DATA, "panel"), columns=PANEL_COLS, start=start, end=end)
    return p


def bench_series(start, end):
    b = pd.read_parquet(os.path.join(DATA, "bench.parquet")).set_index("datetime")
    return b[(b.index >= start) & (b.index <= end)].iloc[:, 0]


def simulate(panel, score, k=3, M=30, exec_at="close", timing=None, buy_cost=0.0005, sell_cost=0.0010, slip=0.001,
             max_hold=0, min_score=None, account=1_000_000, min_cost=5.0):
    """Cash/share replay through the same executor as K_top3 paper trading."""
    from engine.execution import Ledger, concentrated_orders, signal_score, execute_conc
    from engine.market import PanelMarket
    active = panel[panel["in_pool"]]
    if not active["status_known"].fillna(False).all():
        raise ValueError("Historical ST status missing; backfill fetch_status.py and rebuild panel")
    mk = PanelMarket(panel)
    days = sorted(pd.to_datetime(panel["datetime"].unique()))
    scores = {pd.Timestamp(d): g.set_index("instrument")["score"] for d, g in score.groupby("datetime")}
    acc = Ledger(account)
    cfg = dict(k=k, exec_at=exec_at, open_cost=buy_cost, close_cost=sell_cost, min_cost=min_cost, slip=slip)
    ages, out = {}, []
    for i in range(1, len(days)):
        d, signal_day = days[i], days[i - 1]
        prev = acc.total()
        acc.mark(mk, d, at=exec_at)
        day = mk.day(signal_day)
        sc = signal_score(scores.get(signal_day, pd.Series(dtype=float)), day)
        on = True if timing is None else bool(timing.get(signal_day, False))
        pend = concentrated_orders(sc, list(acc.s["positions"]), k, M, on, day["buy_ok"].fillna(False))
        if max_hold:
            pend["sell"] = list(dict.fromkeys(pend["sell"] + [j for j in ages if ages[j] >= max_hold]))
            if on:
                n = max(k - len([j for j in acc.s["positions"] if j not in pend["sell"]]), 0)
                cand = [j for j in sc.sort_values(ascending=False, kind="stable").index
                        if j not in acc.s["positions"] and day["buy_ok"].get(j, False)]
                pend["buy"] = [dict(inst=j) for j in cand[:n]]
                pend["backup"] = [dict(inst=j) for j in cand[n:n + 5]]
        if min_score is not None:
            for c in ("buy", "backup"):
                pend[c] = [x for x in pend[c] if sc.get(x["inst"], -np.inf) >= min_score]
        n0 = len(acc.trades)
        execute_conc(acc, mk, d, pend, {}, cfg)
        acc.mark(mk, d)
        ages = {j: ages.get(j, 0) + 1 for j in acc.s["positions"]}
        trades = [t for t in acc.trades[n0:] if t["操作"] in ("买入", "卖出")]
        traded = sum(t["金额"] for t in trades)
        out.append(dict(datetime=d, ret=acc.total() / prev - 1, value=acc.total(), cash=acc.s["cash"],
                        n_hold=len(acc.s["positions"]), turnover=traded / prev / 2,
                        hold=",".join(acc.s["positions"])))
    res = pd.DataFrame(out, columns=["datetime", "ret", "value", "cash", "n_hold", "turnover", "hold"])
    res.attrs["trades"] = pd.DataFrame(acc.trades)
    return res


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
