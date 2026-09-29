"""
复现 C_top50 / K_top3 的两段回测（研究期 2023-01~2026-09；样本外 2020-01~2022-12）
  python strategies2/backtest.py                      # 两个策略
  python strategies2/backtest.py --strategies K_top3
输出 strategies2/results/：summary.csv、yearly.csv、daily_*.csv、nav_*.png

C_top50：与 A 系列相同的回测引擎（engine.backtest：100 万账户、一手取整、涨跌停、最低 5 元佣金、R3）
K_top3 ：集中持仓模拟器（experiments/conc_lib.py：3 个等权槽位、开盘集合竞价成交、一字涨停买不进、跌停卖不出）
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from engine.backtest import Backtester  # noqa: E402
from engine.common import DATA  # noqa: E402
from experiments.conc_lib import load_panel, simulate, stats  # noqa: E402
from strategies.backtest import run as run_dropout  # noqa: E402
from strategies.live import rank_blend  # noqa: E402
from strategies2.config import RESEARCH_PREDS, STRATEGIES  # noqa: E402

PERIODS = {"研究期2023-26": ("2023-01-01", "2026-09-18", 0, "insample_2023_26"),
           "样本外2020-22": ("2020-01-01", "2022-12-31", 1, "oos_2020_22")}
OUT = "strategies2/results"


def bench_series(end):
    b = pd.read_parquet(os.path.join(DATA, "bench.parquet")).set_index("datetime").iloc[:, 0]
    return b[b.index <= end]


def vote_timing(b, mas, need):
    lvl = (1 + b).cumprod()
    return sum((lvl > lvl.rolling(m).mean()).astype(int) for m in mas) >= need


def plot(curves, bench, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for name, r in curves.items():
        ax.plot((1 + r).cumprod(), label=name)
    ax.plot((1 + bench.reindex(next(iter(curves.values())).index).fillna(0)).cumprod(), label="中证1000", color="gray", ls="--")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategies", help="逗号分隔，如 K_top3")
    a = ap.parse_args()
    names = a.strategies.split(",") if a.strategies else list(STRATEGIES)
    os.makedirs(OUT, exist_ok=True)
    rows, yearly = [], []
    for per, (s, e, k, ft) in PERIODS.items():
        b_all = bench_series(pd.Timestamp(e))
        bench = b_all[b_all.index >= s]
        curves = {}
        for name in names:
            cfg = STRATEGIES[name]
            pred = rank_blend([pd.read_parquet(f"output/preds/{p}.parquet") for p in RESEARCH_PREDS[name][k]])
            if cfg["kind"] == "dropout":
                bt = Backtester(s, e)
                r = run_dropout(bt, pred, cfg)
                m, ms = bt.metrics(r), bt.metrics(run_dropout(bt, pred, cfg, slip=0.001))
                ret, ex = r["return"], r["excess"]
                row = dict(区间=per, 策略=name, 年化=m["策略年化"], 最大回撤=m["策略回撤"], 超额年化=m["超额年化"],
                           信息比率=m["信息比率"], 超额回撤=m["超额回撤"], 千1滑点后超额=ms["超额年化"], 日均换手=m["日均换手"],
                           持股天数占比=1.0, R3触发天数占比=(r["exposure"] < 1).mean())
                r.to_csv(f"{OUT}/daily_{name}_{ft}.csv", encoding="utf-8-sig", float_format="%.6f")
            else:
                panel = load_panel(s, e)
                tm = vote_timing(b_all, cfg["timing_mas"], cfg["timing_need"])
                kw = dict(k=cfg["k"], M=cfg["keep_rank"], exec_at=cfg["exec_at"], timing=tm,
                          buy_cost=cfg["open_cost"], sell_cost=cfg["close_cost"])
                res = simulate(panel, pred, slip=cfg["slip"], **kw)
                d = stats(res, bench)
                d3 = stats(simulate(panel, pred, slip=0.003, **kw), bench)
                ret = res.set_index("datetime")["ret"]
                ex = ret - bench.reindex(ret.index).fillna(0)
                row = dict(区间=per, 策略=name, 年化=d["年化"], 最大回撤=d["最大回撤"], 夏普=d["夏普"],
                           超额年化=ex.mean() * 244, 滑点千3时年化=d3["年化"], 日均换手=d["日换手"],
                           持股天数占比=d["持仓天数占比"], 年换股次数=res["turnover"].sum() * cfg["k"] / (len(res) / 244))
                res.to_csv(f"{OUT}/daily_{name}_{ft}.csv", index=False, encoding="utf-8-sig", float_format="%.6f")
                del panel
            row["基准年化"] = (1 + bench.reindex(ret.index).fillna(0)).prod() ** (244 / len(ret)) - 1
            rows.append(row)
            for y, v in ret.groupby(ret.index.year):
                yearly.append(dict(区间=per, 策略=name, 年份=y, 策略收益=(1 + v).prod() - 1,
                                   基准收益=(1 + bench.reindex(v.index).fillna(0)).prod() - 1))
            curves[name] = ret
            print(per, name, f"年化 {row['年化']:.1%}  最大回撤 {row['最大回撤']:.1%}  超额 {row['超额年化']:.1%}", flush=True)
        plot(curves, bench, f"{OUT}/nav_{ft}.png", f"{per}：净值（扣费）")
    res = pd.DataFrame(rows)
    y = pd.DataFrame(yearly)
    if not a.strategies:
        res.to_csv(f"{OUT}/summary.csv", index=False, encoding="utf-8-sig", float_format="%.4f")
        y.to_csv(f"{OUT}/yearly.csv", index=False, encoding="utf-8-sig", float_format="%.4f")
    pd.set_option("display.width", 250)
    print(res.round(3).to_string(index=False))
    print(y.pivot_table(index=["区间", "年份"], columns="策略", values="策略收益").round(3).to_string())


if __name__ == "__main__":
    main()
