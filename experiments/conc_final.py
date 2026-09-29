"""
集中策略最终候选的细节：分年、滑点、交易次数、各种子单独表现
  python experiments/conc_final.py
"""
import itertools
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.conc_grid import P  # noqa
from experiments.conc_lib import bench_series, load_panel, simulate, stats  # noqa
from experiments.conc_vote import vote_timing  # noqa

rows = []
for per, (s, e, suf) in P.items():
    panel = load_panel(s, e)
    bf = bench_series(pd.Timestamp("2019-01-01"), pd.Timestamp(e))
    bench = bf[bf.index >= s]
    tm = vote_timing(bf, 3)
    for tag, k, ex, slip in itertools.product(["rb_h5ens", "rb_h5", "rb_h5s1", "rb_h5s2"], [2, 3], ["open", "close"],
                                              [0.001, 0.002, 0.003]):
        if tag != "rb_h5ens" and slip != 0.001:
            continue
        sc = pd.read_parquet(f"output/preds/{tag}{suf}.parquet")
        res = simulate(panel, sc, k=k, M=60, exec_at=ex, timing=tm, slip=slip)
        d = stats(res, bench)
        d["年换股次数"] = res["turnover"].sum() * k / (len(res) / 244)
        rows.append(dict(区间=per, 模型=tag, k=k, 成交=ex, 滑点=slip, **d))
        if tag == "rb_h5ens" and slip == 0.001:
            res.to_csv(f"output/conc/daily_k{k}_{ex}_{'oos' if suf else 'insample'}.csv", index=False)
    del panel
res = pd.DataFrame(rows)
res.to_csv("output/conc/final.csv", index=False, encoding="utf-8-sig")
pd.set_option("display.width", 300)
print(res.round(3).drop(columns=[c for c in res.columns if c in ("卡玛",)]).to_string())
