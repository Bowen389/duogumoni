"""
C_top50（主仓）+ 低位首板低开（机会仓，fb_dip 最终版）合并回测
  先：python experiments/senti.py build && python experiments/fb_dip.py build
  再：python experiments/fb_combo.py
主仓日收益取自 strategies2/results/daily_C_top50_*.csv（回测时的基础成本，未含滑点）；
  另给一个近似扣千1滑点的版本：每日减去 换手率 × 2 × 0.001（换手率为该文件的 turnover 列）
机会仓：最终版，每笔占机会仓资金的 50%，最多 2 只，单边滑点千1（见 fb_dip.simulate）；机会仓闲置资金收益按 0 计
组合：每日按固定比例 (1-w) : w 再平衡（实际操作中每月再平衡一次即可，差别很小）
输出 output/fb_dip/combo.csv
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from experiments import fb_dip as F  # noqa: E402

PERIODS = {"样本外 2020-22": "strategies2/results/daily_C_top50_oos_2020_22.csv",
           "研究期 2023-26": "strategies2/results/daily_C_top50_insample_2023_26.csv"}


def perf(r):
    nav = (1 + r).cumprod()
    ann = nav.iloc[-1] ** (243 / len(r)) - 1
    return ann, (nav / nav.cummax() - 1).min(), ann / (r.std() * np.sqrt(243)) if r.std() > 0 else np.nan


def main():
    pd.set_option("display.width", 250)
    X = F.load()
    fb, used = F.simulate(X[F.final_mask(X)], K=2, exit_="C1", order="gap")
    rows = []
    for per, fn in PERIODS.items():
        c = pd.read_csv(fn, parse_dates=["datetime"]).set_index("datetime")
        for slip_name, main in {"C_top50 无滑点": c["return"], "C_top50 千1滑点(近似)": c["return"] - 2 * 0.001 * c["turnover"]}.items():
            f = fb.reindex(main.index).fillna(0.0)
            corr = main.corr(f)
            for w in (0.0, 0.2, 0.3, 0.4, 0.5, 1.0):
                r = (1 - w) * main + w * f
                ann, mdd, sh = perf(r)
                yr = r.groupby(r.index.year).apply(lambda x: (1 + x).prod() - 1)
                rows.append(dict(区间=per, 主仓=slip_name, 机会仓比例=w, 年化=ann, 最大回撤=mdd, 收益回撤比=ann / -mdd,
                                 相关系数=corr, **{f"Y{y}": v for y, v in yr.items()}))
    d = pd.DataFrame(rows)
    os.makedirs("output/fb_dip", exist_ok=True)
    d.to_csv("output/fb_dip/combo.csv", index=False, encoding="utf-8-sig", float_format="%.4f")
    show = d.copy()
    for col in ["年化", "最大回撤"] + [x for x in d.columns if x.startswith("Y")]:
        show[col] = show[col].map(lambda v: "" if pd.isna(v) else f"{v:.1%}")
    show["收益回撤比"] = show["收益回撤比"].round(2)
    show["相关系数"] = show["相关系数"].round(3)
    show["机会仓比例"] = show["机会仓比例"].map(lambda v: f"{v:.0%}")
    print(show.to_string(index=False))


if __name__ == "__main__":
    main()
