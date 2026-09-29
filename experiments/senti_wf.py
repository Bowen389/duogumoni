"""
进攻版（方案 A）最近一年的滚动检验：完全未见过的 2025-09-19 ~ 2026-09-18
  先：bash strategies2/setup_data.sh
      python strategies2/live.py train --strategy K_top3          # 实盘 5 日模型（训练截至 2024-08）
      SENTI_WF=1 python experiments/senti.py train cls,dn         # 大涨/大跌模型（训练截至 2023-08，验证至 2025-09）
  再：python experiments/senti_wf.py
输出 output/senti/wf_recent.csv
注意：实盘 5 日模型的早停验证期（2024-09~2026-09）覆盖了这一年，只用于选树的棵数；大涨/大跌模型则完全没见过这一年
"""
import importlib.util
import itertools
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from engine.common import DATA, KEY  # noqa: E402
from experiments.p2p import Market, stats  # noqa: E402
from experiments.senti import simulate, timings  # noqa: E402
from strategies.live import rank_blend  # noqa: E402

S, E = "2025-09-19", "2026-09-18"
H5 = ["k_rb_s0", "k_rb_s1", "k_rb_s2", "k_rbh_s0", "k_rbh_s1", "k_rbh_s2"]


def main():
    spec = importlib.util.spec_from_file_location("l2", "strategies2/live.py")
    l2 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(l2)
    h5 = rank_blend([l2.predict(S, E, [n], neutral=False)[0] for n in H5])
    up = pd.read_parquet("output/preds/senti_cls_wf.parquet")[KEY + ["score"]]
    dn = pd.read_parquet("output/preds/senti_dn_wf.parquet")[KEY + ["score"]]
    skew = up.merge(dn, on=KEY, suffixes=("", "_d"))
    skew["score"] = skew["score"] - skew["score_d"]
    skew = skew[KEY + ["score"]]
    scores = {"大涨概率": up, "不对称(涨-跌)": skew, "5日模型h5": h5,
              "大涨概率+h5": rank_blend([up, h5]), "不对称+h5": rank_blend([skew, h5])}
    TIM = timings()
    mk = Market(S, E)
    b = pd.read_parquet(os.path.join(DATA, "bench.parquet")).set_index("datetime")["bench"]
    bn = (1 + b[(b.index >= S) & (b.index <= E)]).prod() - 1
    rows = []
    for sname, sc in scores.items():
        R = mk.ranks(sc, depth=100)
        for k, (hold, M), tn in itertools.product([2, 3], [(5, 0), (0, 30), (0, 60)], list(TIM)):
            res = simulate(mk, *R, k=k, M=M, hold=hold, timing=TIM[tn])
            st = stats(res, k)
            rows.append(dict(打分=sname, 持股=k, 卖出=f"持{hold}天" if hold else f"跌出前{M}", 择时=tn,
                             收益=(1 + res["ret"]).prod() - 1, 最大回撤=st["最大回撤"], 中证1000=bn))
    d = pd.DataFrame(rows)
    os.makedirs("output/senti", exist_ok=True)
    d.to_csv("output/senti/wf_recent.csv", index=False, encoding="utf-8-sig", float_format="%.4f")
    pd.set_option("display.width", 250)
    print(f"最近一年 {S} ~ {E}，中证1000 {bn:.1%}")
    print(d.groupby("打分").agg(收益中位数=("收益", "median"), 最好=("收益", "max"), 最差=("收益", "min"),
                              正收益比例=("收益", lambda x: (x > 0).mean()), 回撤中位数=("最大回撤", "median")).round(3).to_string())
    x = d[(d["持股"] == 3) & (d["卖出"] == "跌出前60")]
    x = x.assign(v=[f"{a:.1%}/{c:.0%}" for a, c in zip(x["收益"], x["最大回撤"])])
    print("\n持 3 只、跌出前 60（收益/回撤）")
    print(x.pivot_table(index="择时", columns="打分", values="v", aggfunc="first").to_string())


if __name__ == "__main__":
    main()
