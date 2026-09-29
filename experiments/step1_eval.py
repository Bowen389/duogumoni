"""
第一步：新数据（股东户数）能否做出更好的分散策略？
与 A 策略完全相同的回测口径（行业中性、dropout、R3、100 万、尾盘成交），比较不同模型组合
  python experiments/step1_eval.py
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine.backtest import Backtester  # noqa
from strategies.backtest import run  # noqa
from strategies.config import STRATEGIES  # noqa
from strategies.live import rank_blend  # noqa

P = {"研究期": ("2023-01-01", "2026-09-18", "_neu_ind", "_oos_neu_ind"),
     "样本外": ("2020-01-01", "2022-12-31", None, None)}
COMBOS = {
    "A（retail）": ["retail_h1"],
    "AE（retail+rb）": ["retail_h1", "rb_h1"],
    "rbh 单模型（短期+长期+股东户数）": ["rbh_h1"],
    "retail+rbh": ["retail_h1", "rbh_h1"],
    "retail+rb+rbh": ["retail_h1", "rb_h1", "rbh_h1"],
}
extra = sys.argv[1:]          # 可追加：模型名（如 rbhm_h1）
for t in extra:
    COMBOS[f"{t} 单模型"] = [t]
    COMBOS[f"retail+{t}"] = ["retail_h1", t]
rows = []
for per, (s, e, _, _) in P.items():
    suf = "_neu_ind" if per == "研究期" else "_oos_neu_ind"
    bt = Backtester(s, e)
    for cname, models in COMBOS.items():
        pred = rank_blend([pd.read_parquet(f"output/preds/{m}{suf}.parquet") for m in models])
        for sname in ["A_top50", "A_top20"]:
            cfg = STRATEGIES[sname]
            r = run(bt, pred, cfg)
            m = bt.metrics(r)
            ms = bt.metrics(run(bt, pred, cfg, slip=0.001))
            ex = r["excess"]
            yr = {f"Y{y}": (1 + v).prod() - 1 for y, v in ex.groupby(ex.index.year)}
            rows.append(dict(区间=per, 组合=cname, 持仓=sname[2:], 超额=m["超额年化"], IR=m["信息比率"], 超额回撤=m["超额回撤"],
                             千1滑点超额=ms["超额年化"], **yr))
            print(per, cname, sname, f"{m['超额年化']:.1%} IR {m['信息比率']:.2f}", flush=True)
res = pd.DataFrame(rows)
os.makedirs("output/newdata", exist_ok=True)
res.to_csv("output/newdata/step1_eval.csv", index=False, encoding="utf-8-sig", float_format="%.4f")
pd.set_option("display.width", 250)
print(res.pivot_table(index=["组合", "持仓"], columns="区间", values=["超额", "IR", "千1滑点超额"], sort=False).round(3).to_string())
