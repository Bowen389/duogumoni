"""
第一步：单因子检验 —— 先确认每个“散户愚蠢”在数据里是否真实存在、是否还有效。

输出每个因子的：
  RankIC 均值 / ICIR / IC>0 占比、分年度 RankIC、五分组多空年化收益
用法：
  python factor_ic.py --pool csi1000 --start 2016-01-01 --end 2026-09-25 --horizon 5
"""
import argparse
import numpy as np
import pandas as pd
import qlib
from qlib.data import D

from retail_factors import RETAIL_FACTORS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="~/.qlib/qlib_data/cn_data")
    ap.add_argument("--pool", default="csi1000")
    ap.add_argument("--start", default="2016-01-01")
    ap.add_argument("--end", default="2026-09-25")
    ap.add_argument("--horizon", type=int, default=5, help="预测未来 N 日收益")
    ap.add_argument("--kernels", type=int, default=1)
    ap.add_argument("--out", default="output/factor_ic.csv")
    a = ap.parse_args()

    qlib.init(provider_uri=a.provider, region="cn", kernels=a.kernels)
    inst = D.instruments(a.pool)
    # 标签：T+1 收盘买入，持有 N 日（避免用当日收盘价成交的未来函数）
    label = f"Ref($close,-{a.horizon + 1})/Ref($close,-1)-1"
    # 可交易过滤：次日一字涨停买不进，剔除
    tradable = "Ref($high,-1)>Ref($low,-1)"
    fields = [f for f, _, _ in RETAIL_FACTORS] + [label, tradable]
    names = [n for _, n, _ in RETAIL_FACTORS] + ["LABEL", "TRADABLE"]

    df = D.features(inst, fields, a.start, a.end).astype("float32")
    df.columns = names
    df = df[df["TRADABLE"] > 0].drop(columns="TRADABLE").dropna(subset=["LABEL"])
    df = df.replace([np.inf, -np.inf], np.nan)

    rows = []
    for _, n, desc in RETAIL_FACTORS:
        sub = df[[n, "LABEL"]].dropna()
        g = sub.groupby(level="datetime")
        ic = g.apply(lambda x: x[n].rank().corr(x["LABEL"].rank()) if len(x) > 50 else np.nan).dropna()

        # 五分组：每日按因子分组，看 Top-Bottom 的 N 日收益（非重叠，年化）
        def q_ret(x):
            if len(x) < 100:
                return np.nan
            q = pd.qcut(x[n].rank(method="first"), 5, labels=False)
            return x["LABEL"][q == 4].mean() - x["LABEL"][q == 0].mean()
        ls = g.apply(q_ret).dropna().iloc[:: a.horizon]
        ann_ls = (1 + ls).prod() ** (252 / a.horizon / max(len(ls), 1)) - 1

        yearly = ic.groupby(ic.index.year).mean()
        row = {"因子": n, "含义": desc, "RankIC": ic.mean(), "ICIR": ic.mean() / ic.std(),
               "IC>0占比": (ic > 0).mean(), "多空年化": ann_ls}
        row.update({f"IC_{y}": v for y, v in yearly.items()})
        rows.append(row)
        print(f"{n:10s} RankIC={ic.mean():+.4f} ICIR={row['ICIR']:+.2f} 多空年化={ann_ls:+.1%}  {desc}")

    res = pd.DataFrame(rows).sort_values("ICIR", ascending=False)
    import os
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    res.to_csv(a.out, index=False, encoding="utf-8-sig", float_format="%.4f")
    print(f"\n已保存 {a.out}")


if __name__ == "__main__":
    main()
