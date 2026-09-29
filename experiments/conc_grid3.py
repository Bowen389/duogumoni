"""
集中持股网格 3：5 日收益模型（rb_h5）做集中持仓
  python experiments/conc_grid3.py
"""
import itertools
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.common import DATA, KEY, read_parts  # noqa
from experiments.candidates_lib import rank_blend  # noqa
from experiments.conc_grid import P, scores, timings  # noqa
from experiments.conc_grid2 import two_stage  # noqa
from experiments.conc_lib import bench_series, load_panel, simulate, stats  # noqa

if __name__ == "__main__":
    rows = []
    for per, (s, e, suf) in P.items():
        panel = load_panel(s, e)
        bench_full = bench_series(pd.Timestamp("2019-01-01"), pd.Timestamp(e))
        T = timings(bench_full)
        T = {k: T[k] for k in ["不择时", "指数>MA60"]}
        bench = bench_full[bench_full.index >= s]
        h5n = pd.read_parquet(f"output/preds/rb_h5{suf}_neu_ind.parquet")
        h5r = pd.read_parquet(f"output/preds/rb_h5{suf}.parquet")
        ae = scores(suf)["AE集成"]
        fr = read_parts(os.path.join(DATA, "feat_retail"), columns=KEY + ["VOL20"], start=s, end=e)
        fr["vol"] = -fr["VOL20"]
        mix = rank_blend([ae, h5n], [1, 1])
        S = {"h5中性": h5n, "h5原始": h5r,
             "h5中性前10%最低波": two_stage(h5n, fr, "vol", 0.9, True),
             "h5原始前10%最低波": two_stage(h5r, fr, "vol", 0.9, True),
             "AE+h5前10%最低波": two_stage(mix, fr, "vol", 0.9, True)}
        del fr
        for sname, sc in S.items():
            for k, M, tname, ex in itertools.product([1, 2, 3], [20, 60, 150, 300], list(T), ["close", "open"]):
                res = simulate(panel, sc, k=k, M=M, exec_at=ex, timing=T[tname])
                d = stats(res, bench)
                rows.append(dict(区间=per, 打分=sname, k=k, M=M, 择时=tname, 成交=ex, **d))
            print(per, sname, flush=True)
        del panel
    res = pd.DataFrame(rows)
    res.to_csv("output/conc/grid3.csv", index=False, encoding="utf-8-sig")
    w = res.pivot_table(index=["打分", "k", "M", "择时", "成交"], columns="区间", values=["年化", "最大回撤"]).round(3)
    w["min年化"] = w[("年化", "研究期")].combine(w[("年化", "样本外")], min)
    pd.set_option("display.width", 250)
    print(w.sort_values("min年化", ascending=False).head(25).to_string())
    g = res.groupby(["区间", "打分"])["年化"].median().unstack(0).round(3)
    print("\n各打分所有参数的年化中位数\n", g.to_string())
