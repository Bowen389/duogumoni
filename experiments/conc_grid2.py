"""
集中持股网格 2：在 AE 高分股里按波动率二次挑选
  python experiments/conc_grid2.py
"""
import itertools
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.common import DATA, KEY, read_parts  # noqa
from experiments.conc_grid import P, scores, timings  # noqa
from experiments.conc_lib import bench_series, load_panel, simulate, stats  # noqa


def two_stage(ae, feat, col, top=0.9, low=True):
    """AE 排名前 (1-top) 的股票里，按 col 排序（low=True 取 col 小的）"""
    x = ae.merge(feat[KEY + [col]], on=KEY, how="left")
    x["pct"] = x.groupby("datetime")["score"].rank(pct=True)
    sgn = -1 if low else 1
    x["score"] = np.where(x["pct"] >= top, 10 + sgn * x.groupby("datetime")[col].rank(pct=True), x["pct"])
    return x[KEY + ["score"]]


if __name__ == "__main__":
    rows = []
    for per, (s, e, suf) in P.items():
        panel = load_panel(s, e)
        bench_full = bench_series(pd.Timestamp("2019-01-01"), pd.Timestamp(e))
        T = timings(bench_full)
        T = {k: T[k] for k in ["不择时", "指数>MA60"]}
        bench = bench_full[bench_full.index >= s]
        ae = scores(suf)["AE集成"]
        fr = read_parts(os.path.join(DATA, "feat_retail"), columns=KEY + ["VOL20", "MAX20"], start=s, end=e)
        fr["vol"] = -fr["VOL20"]
        S = {"AE前10%里最低波": two_stage(ae, fr, "vol", 0.9, True),
             "AE前10%里最高波": two_stage(ae, fr, "vol", 0.9, False),
             "AE前30%里最低波": two_stage(ae, fr, "vol", 0.7, True)}
        del fr
        for sname, sc in S.items():
            for k, M, tname, ex in itertools.product([1, 2, 3], [20, 60, 150], list(T), ["close", "open"]):
                res = simulate(panel, sc, k=k, M=M, exec_at=ex, timing=T[tname])
                d = stats(res, bench)
                rows.append(dict(区间=per, 打分=sname, k=k, M=M, 择时=tname, 成交=ex, **d))
            print(per, sname, flush=True)
        del panel
    res = pd.DataFrame(rows)
    res.to_csv("output/conc/grid2.csv", index=False, encoding="utf-8-sig")
    w = res.pivot_table(index=["打分", "k", "M", "择时", "成交"], columns="区间", values=["年化", "最大回撤"]).round(3)
    w["min年化"] = w[("年化", "研究期")].combine(w[("年化", "样本外")], min)
    pd.set_option("display.width", 250)
    print(w.sort_values("min年化", ascending=False).head(20).to_string())
