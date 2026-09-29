"""
分散策略（A / AE / C × 持股 50/20/10/5）加“指数均线投票择时”的效果
  python experiments/timing_overlay.py
输出 output/timing/summary.csv、yearly.csv

问题：持仓 50 只一起大跌时，dropout 规则每天仍只换 2 只 —— 这是市场（beta）下跌，换股解决不了，
只能调整整体仓位。这里测试两种做法：
  空仓：中证1000 站上 20/40/60/80/120 日均线不足 need 条 → 股票全部卖出持有现金
  对冲：同样的信号 → 股票不动，做空等额中证1000股指期货（只赚超额，扣年化 hedge_cost 的贴水成本；粗略估计）
时点：T 日收盘算投票 → T+1 尾盘执行（与策略成交时点一致）→ 影响 T+2 起的收益（lag=2，默认）
      lag=1 为“T 日 14:55 用近似收盘价判断、当天尾盘执行”，实盘可以做到但更紧张，仅作参考
底层回测 = strategies/backtest.run（真实涨跌停、一手取整、佣金印花税、R3），择时只在其日收益上叠加仓位，
空仓期间虚拟组合照常按 dropout 规则演变；重新入场时买回当时的虚拟组合（换仓成本已扣）。
"""
import itertools
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from engine.backtest import Backtester  # noqa: E402
from engine.common import DATA  # noqa: E402
from strategies.backtest import run  # noqa: E402
from strategies.config import COMMON  # noqa: E402
from strategies.live import rank_blend  # noqa: E402

N_YEAR = 238
MAS = [20, 40, 60, 80, 120]
PREDS = {"retail": ("retail_h1_neu_ind", "retail_h1_oos_neu_ind"),
         "rb": ("rb_h1_neu_ind", "rb_h1_oos_neu_ind"),
         "rbh": ("rbh_h1_neu_ind", "rbh_h1_oos_neu_ind")}
MODELS = {"A": ["retail"], "AE": ["retail", "rb"], "C": ["retail", "rb", "rbh"]}
TOPK = [(50, 2), (20, 1), (10, 1), (5, 1)]
PERIODS = {"研究期2023-26": ("2023-01-01", "2026-09-18", 0), "样本外2020-22": ("2020-01-01", "2022-12-31", 1)}
HEDGE_COST = 0.10    # 做空中证1000期货的年化贴水成本（2023-25 年 IM 贴水大致 8-15%），粗略假设


def votes():
    b = pd.read_parquet(os.path.join(DATA, "bench.parquet")).set_index("datetime")["bench"]
    lvl = (1 + b).cumprod()
    return sum((lvl > lvl.rolling(m).mean()).astype(int) for m in MAS)


def overlay(r, on, mode, slip, lag=2):
    """r: 策略日收益表；on: 布尔序列（T 日收盘的择时信号）"""
    e = on.astype(float).shift(lag).reindex(r.index).fillna(1.0) if on is not None else pd.Series(1.0, index=r.index)
    d = e.diff().fillna(0)
    if mode == "cash":
        cost = d.clip(lower=0) * (COMMON["open_cost"] + slip) + (-d).clip(lower=0) * (COMMON["close_cost"] + slip)
        ret = e * r["return"] - cost
    else:   # hedge
        ret = r["return"] - (1 - e) * (r["bench"] + HEDGE_COST / N_YEAR) - d.abs() * 0.0002
    return ret, e


def stats(ret, bench, e):
    nav = (1 + ret).cumprod()
    ex = ret - bench
    n = len(ret)
    ann = nav.iloc[-1] ** (N_YEAR / n) - 1
    dd = (nav / nav.cummax() - 1).min()
    return {"年化": ann, "最大回撤": dd, "收益回撤比": ann / abs(dd) if dd < 0 else np.nan,
            "超额年化": ex.mean() * N_YEAR, "持股天数占比": e.mean(),
            "年切换次数": e.diff().abs().gt(0).sum() / n * N_YEAR}


def main():
    out = "output/timing"
    os.makedirs(out, exist_ok=True)
    V = votes()
    timings = {"不择时": None, "投票≥2": V >= 2, "投票≥3": V >= 3, "投票≥4": V >= 4}
    rows, yrows = [], []
    for per, (s, e, k) in PERIODS.items():
        bt = Backtester(s, e)
        for (mname, names), (topk, nd) in itertools.product(MODELS.items(), TOPK):
            pred = rank_blend([pd.read_parquet(f"output/preds/{PREDS[n][k]}.parquet") for n in names])
            cfg = dict(COMMON, topk=topk, n_drop=nd, every=1, account=1_000_000)
            for slip in [0.0, 0.001]:
                r = run(bt, pred, cfg, slip=slip)
                for (tn, on), mode, lag in itertools.product(timings.items(), ["cash", "hedge"], [2, 1]):
                    if on is None and (mode == "hedge" or lag == 1):
                        continue
                    ret, ex_ = overlay(r, on, mode, slip, lag)
                    st = stats(ret, r["bench"], ex_)
                    row = {"区间": per, "模型": mname, "持股": topk, "滑点": slip, "择时": tn,
                           "方式": "空仓" if mode == "cash" else "对冲", "lag": lag, **st}
                    rows.append(row)
                    if slip == 0.001 and lag == 2:
                        for y, v in ret.groupby(ret.index.year):
                            yrows.append({"区间": per, "模型": mname, "持股": topk, "择时": tn, "方式": row["方式"],
                                          "年份": y, "收益": (1 + v).prod() - 1,
                                          "基准": (1 + r["bench"].loc[v.index]).prod() - 1})
            print(per, mname, topk, "done", flush=True)
            del pred
        del bt
    df = pd.DataFrame(rows)
    df.to_csv(f"{out}/summary.csv", index=False, encoding="utf-8-sig", float_format="%.4f")
    pd.DataFrame(yrows).to_csv(f"{out}/yearly.csv", index=False, encoding="utf-8-sig", float_format="%.4f")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    m = df[(df["滑点"] == 0.001) & (df["lag"] == 2)]
    for per in PERIODS:
        x = m[m["区间"] == per]
        print("\n==", per, "（千1滑点，T+1尾盘执行择时）年化 / 最大回撤 ==")
        x = x.assign(v=[f"{a:.1%}/{b:.0%}" for a, b in zip(x["年化"], x["最大回撤"])])
        print(x.pivot_table(index=["模型", "持股"], columns=["方式", "择时"], values="v", aggfunc="first").to_string())


if __name__ == "__main__":
    main()
