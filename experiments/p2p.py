"""
逐只精选（“点对点”）方案：最多持 k 只（k≤10），每只固定 1/k 仓位；没有合格股票的仓位留现金（允许空仓）
  python experiments/p2p.py
输出 output/p2p/grid.csv

每天、每只股票单独判断：
  买入：按打分 S 从高到低，逐只检查 ——
        共识过滤（可选）：该股必须同时在 1 日模型集成 C 与 5 日模型集成 H5 中都排进前 q 名（两个周期都看好）
        可成交（非停牌、非涨停/开盘一字）
  卖出：打分 S 排名跌出前 M；或（可选）从买入起累计亏损超过 stop；或 指数择时转空；或 调出中证1000
信号日 T 收盘后计算 → T+1 成交（open：开盘集合竞价 / close：尾盘集合竞价）；佣金 + 印花税 + 千1 滑点
打分：C = retail/rb/rbh 三个 1 日模型（行业中性）排名平均；H5 = 6 个 5 日模型（原始打分）排名平均；C+H5 = 两者排名平均
两段：研究期 2023-01~2026-09（模型 2015-2020 训练）；样本外 2020-2022（模型 2015-2018 训练）
"""
import itertools
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from experiments.conc_lib import load_panel  # noqa: E402
from engine.common import DATA  # noqa: E402
from strategies.live import rank_blend  # noqa: E402

N_YEAR = 244
P = {"研究期": ("2023-01-01", "2026-09-18", 0), "样本外": ("2020-01-01", "2022-12-31", 1)}
H1 = {"retail": ("retail_h1_neu_ind", "retail_h1_oos_neu_ind"), "rb": ("rb_h1_neu_ind", "rb_h1_oos_neu_ind"),
      "rbh": ("rbh_h1_neu_ind", "rbh_h1_oos_neu_ind")}
H5 = ("h5ens6", "h5ens6_oos")
MAS = [20, 40, 60, 80, 120]


class Market:
    def __init__(self, start, end):
        p = load_panel(start, end)
        self.days = np.array(sorted(p["datetime"].unique()))
        self.insts = np.array(sorted(p["instrument"].unique()))
        self.di = pd.Series(np.arange(len(self.days)), index=self.days)
        self.ii = pd.Series(np.arange(len(self.insts)), index=self.insts)
        r_, c_ = self.di.reindex(p["datetime"].values).values, self.ii.reindex(p["instrument"].values).values
        shp = (len(self.days), len(self.insts))
        self.m = {}
        for c in ["ret", "ret_on", "ret_id"]:
            a = np.zeros(shp, dtype="float32")
            a[r_, c_] = p[c].fillna(0).values
            self.m[c] = a
        for c, dflt in [("up_lim", False), ("dn_lim", False), ("up_open", False), ("dn_open", False), ("susp", True),
                        ("in_pool", False)]:
            a = np.full(shp, dflt, dtype=bool)
            a[r_, c_] = p[c].fillna(dflt).values.astype(bool)
            self.m[c] = a
        del p

    def ranks(self, score, depth=200):
        """每个信号日：{股票序号: 名次}（只保留前 depth 名）和按名次排序的列表"""
        sc = score[score["datetime"].isin(self.days) & score["instrument"].isin(self.insts)].copy()
        sc["rk"] = sc.groupby("datetime")["score"].rank(ascending=False, method="first")
        sc = sc[sc["rk"] <= depth].sort_values(["datetime", "rk"])
        rk, order = {}, {}
        for d, g in sc.groupby("datetime"):
            idx = self.ii.reindex(g["instrument"].values).values
            rk[d] = dict(zip(idx, g["rk"].values))
            order[d] = list(idx)
        return rk, order


def simulate(mk, rkS, ordS, k=10, M=60, exec_at="open", timing=None, stop=None, cons=None, q=None,
             buy_cost=0.0005, sell_cost=0.0010, slip=0.001):
    """cons: (rkC, rkH) 两个周期的名次字典；q: 共识门槛"""
    m, days = mk.m, mk.days
    hold = {}   # 股票 -> 买入以来净值
    out = []
    w = 1.0 / k
    for i in range(1, len(days)):
        t, s = days[i], days[i - 1]
        rk = rkS.get(s, {})
        on = True if timing is None else bool(timing.get(s, True))
        sells, keeps = [], []
        for x, v in hold.items():
            want = (not on) or rk.get(x, 1e9) > M or not m["in_pool"][i, x] or (stop is not None and v - 1 <= -stop)
            blocked = m["susp"][i, x] or (m["dn_lim"][i, x] if exec_at == "close" else m["dn_open"][i, x])
            (sells if want and not blocked else keeps).append(x)
        buys = []
        n_free = k - len(keeps)
        if on and n_free > 0:
            if cons is not None:
                rc, rh = cons[0].get(s, {}), cons[1].get(s, {})
            for x in ordS.get(s, []):
                if len(buys) >= n_free:
                    break
                if x in hold:
                    continue
                if cons is not None and not (rc.get(x, 1e9) <= q and rh.get(x, 1e9) <= q):
                    continue
                if not m["in_pool"][i, x]:
                    continue
                blocked = m["susp"][i, x] or (m["up_lim"][i, x] if exec_at == "close" else m["up_open"][i, x])
                if not blocked:
                    buys.append(x)
        r = 0.0
        newv = {}
        if exec_at == "close":
            for x in keeps + sells:
                g = m["ret"][i, x]
                r += w * g
                newv[x] = hold[x] * (1 + g)
            for x in buys:
                newv[x] = 1.0
        else:
            for x in keeps:
                g = m["ret"][i, x]
                r += w * g
                newv[x] = hold[x] * (1 + g)
            for x in sells:
                r += w * m["ret_on"][i, x]
            for x in buys:
                g = m["ret_id"][i, x]
                r += w * g
                newv[x] = 1 + g
        r -= w * (len(sells) * (sell_cost + slip) + len(buys) * (buy_cost + slip))
        hold = {x: newv[x] for x in keeps + buys}
        out.append((t, r, len(hold), w * (len(sells) + len(buys)) / 2))
    return pd.DataFrame(out, columns=["datetime", "ret", "n_hold", "turnover"]).set_index("datetime")


def stats(res, k):
    r = res["ret"]
    nav = (1 + r).cumprod()
    ann = nav.iloc[-1] ** (N_YEAR / len(r)) - 1
    dd = (nav / nav.cummax() - 1).min()
    d = dict(年化=ann, 最大回撤=dd, 卡玛=ann / abs(dd) if dd < 0 else np.nan,
             夏普=r.mean() / r.std() * np.sqrt(N_YEAR) if r.std() > 0 else np.nan,
             平均仓位=(res["n_hold"] / k).mean(), 空仓天数占比=(res["n_hold"] == 0).mean(), 日换手=res["turnover"].mean())
    for y, v in r.groupby(r.index.year):
        d[f"Y{y}"] = (1 + v).prod() - 1
    return d


def main():
    out = "output/p2p"
    os.makedirs(out, exist_ok=True)
    b = pd.read_parquet(os.path.join(DATA, "bench.parquet")).set_index("datetime")["bench"]
    lvl = (1 + b).cumprod()
    vote = sum((lvl > lvl.rolling(mm).mean()).astype(int) for mm in MAS)
    TIM = {"不择时": None, "投票≥3": vote >= 3}
    rows = []
    for per, (s, e, kk) in P.items():
        mk = Market(s, e)
        C = rank_blend([pd.read_parquet(f"output/preds/{H1[n][kk]}.parquet") for n in H1])
        H = pd.read_parquet(f"output/preds/{H5[kk]}.parquet")[["datetime", "instrument", "score"]]
        CH = rank_blend([C, H])
        R = {n: mk.ranks(x) for n, x in [("C", C), ("H5", H), ("C+H5", CH)]}
        del C, H, CH
        cons = (R["C"][0], R["H5"][0])
        grid = itertools.product(["C", "H5", "C+H5"], [None, 30, 100], [5, 10], [20, 60], list(TIM), ["open", "close"],
                                 [None, 0.10])
        for sname, q, k, M, tn, ex, stop in grid:
            res = simulate(mk, *R[sname], k=k, M=M, exec_at=ex, timing=TIM[tn], stop=stop,
                           cons=cons if q else None, q=q)
            rows.append(dict(区间=per, 打分=sname, 共识前q=q or "无", 最多持股=k, 缓冲M=M, 择时=tn, 成交=ex,
                             止损="无" if stop is None else f"-{stop:.0%}", **stats(res, k)))
        print(per, "done", len(rows), flush=True)
        del mk
    df = pd.DataFrame(rows)
    df.to_csv(f"{out}/grid.csv", index=False, encoding="utf-8-sig", float_format="%.4f")


if __name__ == "__main__":
    main()
