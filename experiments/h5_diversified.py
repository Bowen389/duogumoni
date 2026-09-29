"""
5 日模型（三种子集成、原始打分）做分散持仓：k=10/20/50，缓冲 M，是否择时
  python experiments/h5_diversified.py
"""
import itertools
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.conc_grid import P  # noqa
from experiments.conc_lib import bench_series, load_panel, simulate, stats  # noqa
from experiments.conc_vote import vote_timing  # noqa

if __name__ == "__main__":
    rows = []
    for per, (s, e, suf) in P.items():
        panel = load_panel(s, e)
        bf = bench_series(pd.Timestamp("2019-01-01"), pd.Timestamp(e))
        bench = bf[bf.index >= s]
        tm = vote_timing(bf, 3)
        for tag in ["rb_h5ens", "AE"]:
            if tag == "AE":
                from experiments.conc_grid import scores
                sc = scores(suf)["AE集成"]
            else:
                sc = pd.read_parquet(f"output/preds/{tag}{suf}.parquet")
            for k, mult, tn, ex in itertools.product([10, 20, 50], [2, 4, 8], ["不择时", "投票"], ["open", "close"]):
                if tag == "AE" and (tn != "不择时" or ex != "close"):
                    continue
                res = simulate(panel, sc, k=k, M=k * mult, exec_at=ex, timing=tm if tn == "投票" else None)
                d = stats(res, bench)
                b = bench.reindex(res["datetime"]).fillna(0).values
                ex_r = res["ret"].values - b * (res["n_hold"].values > 0)
                d["超额年化"] = ex_r.mean() * 244
                d["超额IR"] = ex_r.mean() / ex_r.std() * 244 ** 0.5
                rows.append(dict(区间=per, 模型=tag, k=k, M=k * mult, 择时=tn, 成交=ex, **d))
            print(per, tag, flush=True)
        del panel
    res = pd.DataFrame(rows)
    res.to_csv("output/conc/h5_diversified.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 250)
    print(res.pivot_table(index=["模型", "k", "M", "择时", "成交"], columns="区间",
                          values=["年化", "最大回撤", "超额年化", "超额IR"]).round(3).to_string())
