"""
复现两个策略的回测（真实涨跌停、停牌、一手取整、最低 5 元佣金、R3 风控）

  python strategies/backtest.py                    # 用仓库自带的预测文件（与 README 数字一致）
  python strategies/backtest.py --pred my_preds    # 用自己训练的预测（output/preds/my_preds.parquet，会自动做行业中性）
  python strategies/backtest.py --strategies AE_top20

两段检验：
  研究期  2023-01 ~ 2026-09  模型用 2015-2020 训练、2021-2022 早停
  样本外  2020-01 ~ 2022-12  模型用 2015-2018 训练、2019 早停；这 3 年从未参与任何方案选择
  预测文件见 config.RESEARCH_PREDS（多模型策略会把各模型的预测按排名等权平均）
输出 strategies/results/：summary.csv、yearly.csv、monthly_excess_{策略}_{区间}.csv、daily_{策略}_{区间}.csv、nav_{区间}.png
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
from strategies.config import RESEARCH_PREDS, STRATEGIES  # noqa: E402
from strategies.live import industry_neutral, rank_blend  # noqa: E402

N_YEAR = 238


def r3(bt, res, cfg):
    cum = (1 + res["excess"]).cumprod()
    dd = (cum / cum.cummax() - 1).shift(1).fillna(0)
    st, e = 1.0, []
    for v in dd.values:
        if st == 1.0 and v < cfg["r3_in"]:
            st = cfg["r3_exposure"]
        elif st < 1.0 and v > cfg["r3_out"]:
            st = 1.0
        e.append(st)
    return bt.overlay(res, pd.Series(e, index=res.index))


def run(bt, pred, cfg, slip=0.0, account=None):
    r = bt.run(pred, mode="dropout", topk=cfg["topk"], n_drop=cfg["n_drop"], every=cfg["every"],
               open_cost=cfg["open_cost"], close_cost=cfg["close_cost"], min_cost=cfg["min_cost"],
               risk_degree=cfg["risk_degree"], account=account or cfg["account"], slip=slip)
    return r3(bt, r, cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", help="自定义预测（output/preds/ 下的文件名，不带扩展名）")
    ap.add_argument("--start", default="2023-01-01")
    ap.add_argument("--end", default="2026-09-18")
    ap.add_argument("--strategies", help="只跑部分策略，逗号分隔，如 A_top50,AE_top20")
    a = ap.parse_args()
    out = "strategies/results"
    os.makedirs(out, exist_ok=True)

    cache = {}

    def research_pred(names, k):   # k=0 研究期 / 1 样本外；多模型 → 排名等权平均
        key = (tuple(names), k)
        if key not in cache:
            cache[key] = rank_blend([pd.read_parquet(f"output/preds/{RESEARCH_PREDS[n][k]}.parquet") for n in names])
        return cache[key]

    if a.pred:
        p = industry_neutral(pd.read_parquet(f"output/preds/{a.pred}.parquet"))
        periods = {"自定义": (a.start, a.end, lambda cfg: p)}
    else:
        periods = {"研究期2023-26": ("2023-01-01", "2026-09-18", lambda cfg: research_pred(cfg["models"], 0)),
                   "样本外2020-22": ("2020-01-01", "2022-12-31", lambda cfg: research_pred(cfg["models"], 1))}

    FTAG = {"研究期2023-26": "insample_2023_26", "样本外2020-22": "oos_2020_22", "自定义": "custom"}
    rows, yearly = [], []
    for per, (s, e, pred_of) in periods.items():
        ft = FTAG[per]
        bt = Backtester(s, e)
        curves = {}
        for name, cfg in STRATEGIES.items():
            if a.strategies and name not in a.strategies.split(","):
                continue
            pred = pred_of(cfg)
            r = run(bt, pred, cfg)
            rs = run(bt, pred, cfg, slip=0.001)
            m, ms = bt.metrics(r), bt.metrics(rs)
            ex = r["excess"]
            rows.append({"区间": per, "策略": name, "超额年化": m["超额年化"], "信息比率": m["信息比率"],
                         "超额回撤": m["超额回撤"], "策略年化": m["策略年化"], "策略回撤": m["策略回撤"],
                         "基准年化": (1 + r["bench"]).prod() ** (N_YEAR / len(r)) - 1,
                         "日均换手": m["日均换手"], "年化成本": m["年化成本"], "+千1滑点超额": ms["超额年化"],
                         "日胜率": (ex > 0).mean(), "月胜率": (ex.groupby(ex.index.to_period("M")).sum() > 0).mean(),
                         "R3触发天数占比": (r["exposure"] < 1).mean()})
            for y, v in ex.groupby(ex.index.year):
                yearly.append({"区间": per, "策略": name, "年份": y, "超额": (1 + v).prod() - 1,
                               "策略收益": (1 + r.loc[v.index, "return"]).prod() - 1,
                               "基准收益": (1 + r.loc[v.index, "bench"]).prod() - 1})
            r.to_csv(f"{out}/daily_{name}_{ft}.csv", encoding="utf-8-sig", float_format="%.6f")
            mon = ex.groupby([ex.index.year, ex.index.month]).apply(lambda x: (1 + x).prod() - 1).unstack()
            mon.to_csv(f"{out}/monthly_excess_{name}_{ft}.csv", encoding="utf-8-sig", float_format="%.4f")
            curves[name] = r
            print(per, name, f"超额 {m['超额年化']:.1%}  IR {m['信息比率']:.2f}", flush=True)
        bt.plot(curves, f"{out}/nav_{ft}.png", f"{per}：累计超额（扣费、含 R3；下图为 {list(curves)[0]} 与中证1000）")

    res = pd.DataFrame(rows)
    res.to_csv(f"{out}/summary.csv", index=False, encoding="utf-8-sig", float_format="%.4f")
    pd.DataFrame(yearly).to_csv(f"{out}/yearly.csv", index=False, encoding="utf-8-sig", float_format="%.4f")
    pd.set_option("display.width", 250)
    print(res.round(3).to_string(index=False))
    print(pd.DataFrame(yearly).pivot_table(index=["区间", "年份"], columns="策略", values="超额").round(3))


if __name__ == "__main__":
    main()
