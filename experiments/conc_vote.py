"""
集中策略：多均线投票择时（20/40/60/80/120 日线中至少 n 条在指数下方才持仓），降低对单一均线参数的依赖
  python experiments/conc_vote.py rb_h5ens
"""
import itertools
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.conc_grid import P  # noqa
from experiments.conc_lib import bench_series, load_panel, simulate, stats  # noqa

MAS = [20, 40, 60, 80, 120]


def vote_timing(bench, need=3):
    lvl = (1 + bench).cumprod()
    v = sum((lvl > lvl.rolling(m).mean()).astype(int) for m in MAS)
    return v >= need


if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else "rb_h5ens"
    rows = []
    for per, (s, e, suf) in P.items():
        panel = load_panel(s, e)
        bf = bench_series(pd.Timestamp("2019-01-01"), pd.Timestamp(e))
        bench = bf[bf.index >= s]
        sc = pd.read_parquet(f"output/preds/{tag}{suf}.parquet")
        for need, M, k, ex in itertools.product([2, 3, 4], [30, 60, 100], [1, 2, 3], ["open", "close"]):
            d = stats(simulate(panel, sc, k=k, M=M, exec_at=ex, timing=vote_timing(bf, need)), bench)
            rows.append(dict(区间=per, 票数=need, M=M, k=k, 成交=ex, **d))
        print(per, flush=True)
        del panel
    res = pd.DataFrame(rows)
    res.to_csv(f"output/conc/vote_{tag}.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    for v in ["年化", "最大回撤"]:
        print(f"\n{v}")
        print(res.pivot_table(index=["票数", "M"], columns=["区间", "k", "成交"], values=v).round(3).to_string())
    print(res.groupby(["区间", "票数"])[["年化", "最大回撤"]].median().round(3).to_string())
