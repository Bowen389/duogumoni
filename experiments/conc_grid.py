"""
集中持股网格：打分 × k × M(缓冲) × 择时 × 成交时点，两段区间
  python experiments/conc_grid.py [tag]
"""
import itertools
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.candidates_lib import rank_blend  # noqa
from experiments.conc_lib import bench_series, load_panel, simulate, stats  # noqa

P = {"研究期": ("2023-01-01", "2026-09-18", ""), "样本外": ("2020-01-01", "2022-12-31", "_oos")}


def timings(bench):
    lvl = (1 + bench).cumprod()
    return {"不择时": None,
            "指数>MA20": lvl > lvl.rolling(20).mean(),
            "指数>MA60": lvl > lvl.rolling(60).mean(),
            "MA20>MA60": lvl.rolling(20).mean() > lvl.rolling(60).mean()}


def scores(suf):
    pr = pd.read_parquet(f"output/preds/retail_h1{suf}_neu_ind.parquet")
    pb = pd.read_parquet(f"output/preds/rb_h1{suf}_neu_ind.parquet")
    return {"AE集成": rank_blend([pr, pb], [1, 1]),
            "rb原始分": pd.read_parquet(f"output/preds/rb_h1{suf}.parquet"),
            "retail中性": pr}


if __name__ == "__main__":
    rows = []
    for per, (s, e, suf) in P.items():
        panel = load_panel("2019-09-01" if suf else "2022-09-01", e)
        bench_full = bench_series(pd.Timestamp("2019-01-01"), pd.Timestamp(e))
        T = timings(bench_full)
        panel = panel[panel["datetime"] >= s]
        bench = bench_full[bench_full.index >= s]
        for sname, sc in scores(suf).items():
            for k, M, tname, ex in itertools.product([1, 2, 3], [5, 20, 60], list(T), ["close", "open"]):
                if ex == "open" and sname != "AE集成":
                    continue
                res = simulate(panel, sc, k=k, M=M, exec_at=ex, timing=T[tname])
                d = stats(res, bench)
                rows.append(dict(区间=per, 打分=sname, k=k, M=M, 择时=tname, 成交=ex, **d))
                print(per, sname, k, M, tname, ex, f"年化 {d['年化']:.1%} 回撤 {d['最大回撤']:.1%}", flush=True)
        del panel
    res = pd.DataFrame(rows)
    os.makedirs("output/conc", exist_ok=True)
    res.to_csv("output/conc/grid.csv", index=False, encoding="utf-8-sig")
