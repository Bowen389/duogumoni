"""
集中策略参数邻域：均线长度 × 缓冲 M × k，看结果是否平滑（防止挑中孤立的好参数）
  python experiments/conc_neighbors.py rb_h5ens
"""
import itertools
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.conc_grid import P  # noqa
from experiments.conc_lib import bench_series, load_panel, simulate, stats  # noqa

tag = sys.argv[1] if len(sys.argv) > 1 else "rb_h5ens"
rows = []
for per, (s, e, suf) in P.items():
    panel = load_panel(s, e)
    bf = bench_series(pd.Timestamp("2019-01-01"), pd.Timestamp(e))
    lvl = (1 + bf).cumprod()
    bench = bf[bf.index >= s]
    sc = pd.read_parquet(f"output/preds/{tag}{suf}.parquet")
    for ma, M, k, ex in itertools.product([40, 60, 80, 120], [30, 60, 100], [2, 3], ["open", "close"]):
        tm = lvl > lvl.rolling(ma).mean()
        d = stats(simulate(panel, sc, k=k, M=M, exec_at=ex, timing=tm), bench)
        rows.append(dict(区间=per, MA=ma, M=M, k=k, 成交=ex, 年化=d["年化"], 最大回撤=d["最大回撤"]))
    print(per, flush=True)
    del panel
res = pd.DataFrame(rows)
res.to_csv(f"output/conc/neighbors_{tag}.csv", index=False, encoding="utf-8-sig")
pd.set_option("display.width", 250)
for v in ["年化", "最大回撤"]:
    print(f"\n{v}（行：MA × M，列：区间 × k × 成交）")
    print(res.pivot_table(index=["MA", "M"], columns=["区间", "k", "成交"], values=v).round(3).to_string())
