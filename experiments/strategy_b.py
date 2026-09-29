"""
新策略 B："反散户"规则组合（不用机器学习）
  1) 只用 2016-2019 的数据挑因子：28 个散户因子（原 16 + 长期犯错 12）中，行业中性 ICIR 绝对值 >= 阈值、且 4 年里至少 3 年同号的
  2) 每天把入选因子按方向转成截面排名、等权平均 → 行业中性 → Top50 每天换 2 / R3（与 A 相同的交易规则）
  3) 两段测试 2020-22 / 2023-26 对于"挑因子"都是样本外
  python experiments/strategy_b.py
"""
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import engine.backtest as EB  # noqa
from engine.common import DATA, EXTRA, KEY, read_parts  # noqa
from experiments.candidates_lib import r3, rank_blend  # noqa

TH = float(os.environ.get("B_TH", 0.30))
SEL = ("2016-01-01", "2019-12-31")
ind = pd.read_parquet(os.path.join(EXTRA, "industry.parquet"))[["instrument", "industry"]]


def load(start, end, flt=None):
    a = read_parts(os.path.join(DATA, "feat_retail"), start=start, end=end, filters_extra=flt)
    b = read_parts(os.path.join(DATA, "feat_behavior"), start=start, end=end, filters_extra=flt)
    df = a.merge(b, on=KEY).merge(ind, on="instrument", how="left")
    df["industry"] = df["industry"].fillna("UNK")
    return df


# ---------- 1) 挑因子（2016-2019，每 2 天抽 1 天） ----------
d = pd.read_parquet(os.path.join(DATA, "bench.parquet"))["datetime"]
days = d[(d >= SEL[0]) & (d <= SEL[1])].iloc[::2]
flt = ds.field("datetime").isin(pd.to_datetime(days).values)
df = load(*SEL, flt)
lab = read_parts(os.path.join(DATA, "panel"), columns=KEY + ["in_pool", "nb", "ret5"], start=SEL[0], end=SEL[1], filters_extra=flt)
df = df.merge(lab[lab["in_pool"] & ~lab["nb"]], on=KEY)
cols = [c for c in df.columns if c not in KEY + ["industry", "in_pool", "nb", "ret5"]]
g = df.groupby("datetime")
y = g["ret5"].rank(pct=True)
stats = []
for c in cols:
    r = g[c].rank(pct=True)
    r = r - r.groupby([df["datetime"], df["industry"]]).transform("mean")
    t = pd.DataFrame({"d": df["datetime"], "x": r, "y": y}).dropna()
    ic = t.groupby("d").apply(lambda z: z["x"].corr(z["y"]))
    yr = ic.groupby(ic.index.year).mean()
    stats.append({"因子": c, "IC": ic.mean(), "ICIR": ic.mean() / ic.std(),
                  "同号年数": int((np.sign(yr) == np.sign(ic.mean())).sum())})
st = pd.DataFrame(stats).set_index("因子").sort_values("ICIR", key=abs, ascending=False)
chosen = st[(st["ICIR"].abs() >= TH) & (st["同号年数"] >= 3)]
os.makedirs("output/strategy_b", exist_ok=True)
st.to_csv("output/strategy_b/factor_selection_2016_2019.csv", encoding="utf-8-sig", float_format="%.4f")
print(st.round(3).to_string())
print(f"\n入选 {len(chosen)} 个（|ICIR|>={TH}）：", ", ".join(f"{c}({'+' if s > 0 else '-'})" for c, s in chosen["ICIR"].items()))
del df, lab
SIGN = np.sign(chosen["ICIR"])


# ---------- 2) 合成打分（2020-2026 每天） ----------
def composite(start, end):
    f = load(start, end)
    gg = f.groupby("datetime")
    sc = sum(gg[c].rank(pct=True) * s for c, s in SIGN.items()) / len(SIGN)
    f["s"] = sc
    f["s"] = f.groupby("datetime")["s"].rank(pct=True) - 0.5
    f["s"] = f["s"] - f.groupby(["datetime", "industry"])["s"].transform("mean")
    return f[KEY + ["s"]].rename(columns={"s": "score"})


PER = {"研究期": ("2023-01-01", "2026-09-18", ""), "样本外": ("2020-01-01", "2022-12-31", "_oos")}
rows, curves = [], {}
for per, (s, e, sfx) in PER.items():
    comp = composite(pd.Timestamp(s) - pd.Timedelta(days=10), e)
    comp.to_parquet(f"output/preds/stratB{sfx}.parquet", index=False)
    A = pd.read_parquet(f"output/preds/retail_h1{sfx}_neu_ind.parquet")
    AB = rank_blend([A, comp], [1, 1])
    bt = EB.Backtester(s, e)
    for name, pred, k, nd in [("A 基线 Top50", A, 50, 2), ("B 反散户规则 Top50", comp, 50, 2), ("B 反散户规则 Top20", comp, 20, 1),
                              ("A+B 打分平均 Top50", AB, 50, 2), ("A 基线 Top20", A, 20, 1), ("A+B 打分平均 Top20", AB, 20, 1)]:
        r = r3(bt, bt.run(pred, topk=k, n_drop=nd))
        rs = r3(bt, bt.run(pred, topk=k, n_drop=nd, slip=0.001))
        m = bt.metrics(r)
        rows.append({"区间": per, "方案": name, "超额年化": m["超额年化"], "信息比率": m["信息比率"], "超额回撤": m["超额回撤"],
                     "策略年化": m["策略年化"], "策略回撤": m["策略回撤"], "日均换手": m["日均换手"],
                     "+千1滑点": bt.metrics(rs)["超额年化"], **{f"超额{y}": v for y, v in m.items() if str(y).startswith("超额20")}})
        curves[(per, name)] = r
        print(per, name, f"{m['超额年化']:.1%} IR {m['信息比率']:.2f}", flush=True)
    ex = pd.DataFrame({n: curves[(per, n)]["excess"] for n in ["A 基线 Top50", "B 反散户规则 Top50"]})
    print(per, "A 与 B 日超额相关", round(ex.corr().iloc[0, 1], 3))
    bt.plot({n: curves[(per, n)] for n in ["A+B 打分平均 Top50", "A 基线 Top50", "B 反散户规则 Top50"]},
            f"output/strategy_b/nav_{'is' if per == '研究期' else 'oos'}.png", f"{per}：A / B / A+B 累计超额（含 R3）")
res = pd.DataFrame(rows)
res.to_csv("output/strategy_b/summary.csv", index=False, encoding="utf-8-sig", float_format="%.4f")
pd.set_option("display.width", 300)
print(res.drop(columns=[c for c in res if c.startswith("超额20")]).round(3).to_string(index=False))
