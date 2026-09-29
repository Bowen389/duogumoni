"""
用“实盘模型”（strategies2 训练，训练截至 2024-08，2024-09~2026-09 只用于早停选树数）检验逐只精选方案的最近一两年表现
  先：bash strategies2/setup_data.sh && python strategies2/live.py train --strategy K_top3
  再：python experiments/p2p_prod.py
输出 output/p2p/prod_recent.csv
"""
import importlib.util
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from experiments.p2p import MAS, Market, simulate, stats  # noqa: E402
from engine.common import DATA  # noqa: E402
from strategies.live import rank_blend  # noqa: E402
import pandas as pd  # noqa: E402

RB = ["k_rb_s0", "k_rb_s1", "k_rb_s2"]
RBH = ["k_rbh_s0", "k_rbh_s1", "k_rbh_s2"]
CANDS = [("不择时", 10, 100), ("不择时", 10, 60), ("不择时", 8, 60), ("不择时", 5, 60),
         ("投票≥3", 6, 60), ("投票≥3", 5, 40), ("投票≥3", 3, 60)]


def main():
    spec = importlib.util.spec_from_file_location("l2", "strategies2/live.py")
    l2 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(l2)
    raw = {n: l2.predict("2024-09-01", "2026-09-28", [n], neutral=False)[0] for n in RB + RBH}
    scores = {"研究版h5ens6": pd.read_parquet("output/preds/h5ens6.parquet")[["datetime", "instrument", "score"]],
              "实盘3模型(rb)": rank_blend([raw[n] for n in RB]),
              "实盘6模型": rank_blend([raw[n] for n in RB + RBH])}
    b = pd.read_parquet(os.path.join(DATA, "bench.parquet")).set_index("datetime")["bench"]
    lvl = (1 + b).cumprod()
    vote = sum((lvl > lvl.rolling(m).mean()).astype(int) for m in MAS)
    rows = []
    for s, e, wn in [("2024-09-02", "2026-09-18", "近两年"), ("2025-09-19", "2026-09-18", "近一年")]:
        mk = Market(s, e)
        bn = (1 + b[(b.index >= s) & (b.index <= e)]).prod() - 1
        for tag, sc in scores.items():
            R = mk.ranks(sc)
            for tn, k, M in CANDS:
                res = simulate(mk, *R, k=k, M=M, exec_at="open", timing=None if tn == "不择时" else vote >= 3)
                st = stats(res, k)
                rows.append(dict(窗口=wn, 模型=tag, 方案=f"{tn} k{k} M{M}", 累计收益=(1 + res.ret).prod() - 1,
                                 最大回撤=st["最大回撤"], 中证1000=bn))
    d = pd.DataFrame(rows)
    d.to_csv("output/p2p/prod_recent.csv", index=False, encoding="utf-8-sig", float_format="%.4f")
    d["v"] = [f"{a:.1%}/{b:.0%}" for a, b in zip(d["累计收益"], d["最大回撤"])]
    pd.set_option("display.width", 260)
    print(d.pivot_table(index="方案", columns=["窗口", "模型"], values="v", aggfunc="first", sort=False).to_string())


if __name__ == "__main__":
    main()
