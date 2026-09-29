"""
集中策略：换随机种子 / 换特征集重训后结果是否稳定
  python experiments/conc_seeds.py rb_h5 rb_h5s1 rb_h5s2 retail_h5
"""
import itertools
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.conc_grid import P, timings  # noqa
from experiments.conc_lib import bench_series, load_panel, simulate, stats  # noqa

tags = sys.argv[1:]
rows = []
for per, (s, e, suf) in P.items():
    panel = load_panel(s, e)
    bench_full = bench_series(pd.Timestamp("2019-01-01"), pd.Timestamp(e))
    T = timings(bench_full)
    bench = bench_full[bench_full.index >= s]
    for tag in tags:
        sc = pd.read_parquet(f"output/preds/{tag}{suf}.parquet")
        for k, M, tn, ex in itertools.product([1, 2, 3], [60, 150, 300], ["不择时", "指数>MA60"], ["open", "close"]):
            d = stats(simulate(panel, sc, k=k, M=M, exec_at=ex, timing=T[tn]), bench)
            rows.append(dict(区间=per, 模型=tag, k=k, M=M, 择时=tn, 成交=ex, 年化=d["年化"], 最大回撤=d["最大回撤"]))
        print(per, tag, flush=True)
    del panel
res = pd.DataFrame(rows)
res.to_csv("output/conc/seeds.csv", index=False, encoding="utf-8-sig")
pd.set_option("display.width", 250)
print("所有参数组合的年化中位数 / 25%分位 / 回撤中位数")
print(res.groupby(["模型", "区间"]).agg(年化中位=("年化", "median"), 年化25分位=("年化", lambda x: x.quantile(.25)),
                                      回撤中位=("最大回撤", "median"), 正收益占比=("年化", lambda x: (x > 0).mean())).round(3).to_string())
c1 = res[(res.k == 2) & (res.M == 300) & (res.择时 == "指数>MA60")]
print("\nC1 类（k2 M300 MA60）\n", c1.pivot_table(index=["模型", "成交"], columns="区间", values=["年化", "最大回撤"]).round(3).to_string())
