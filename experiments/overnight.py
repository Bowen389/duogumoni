"""
方案 C：隔夜 / 竞价超短线（只用日线数据，中证1000 股票池）
  python experiments/senti.py build           # 先生成 senti_stock（含标签 yon / yid）
  python experiments/overnight.py desc        # 描述统计：隔夜 vs 日内收益、单特征分组
  python experiments/overnight.py train [on,id]   # LightGBM 预测隔夜(on) / 日内(id) 收益，研究期+样本外(+SENTI_WF=1 滚动近一年)
  python experiments/overnight.py bt          # 每日尾盘买前 k、次日开盘卖（on）；或持底仓做 T（id）的回测

标签（见 senti.py build）
  yon：T 日收盘价买入（收盘封涨停/停牌 → 买不进，剔除）→ T+1 开盘价卖出（开盘一字跌停/停牌 → 按 T+1 收盘卖）
  yid：T+1 开盘价买入（一字涨停/停牌 → 剔除）→ T+1 收盘价卖出。T+1 制度下只有持有底仓才能当天卖（“做 T”）
成本（每次买卖一轮）：佣金 万2.5×2 + 印花税（卖出，2023-08-28 前千1，之后万5）+ 两边滑点
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from engine.common import DATA, KEY  # noqa: E402
from experiments.senti import PERIODS, WF, STOCK_F, load_xy  # noqa: E402

OUT = "output/overnight"
PER_ALL = {"样本外 2020-22": ("2020-01-01", "2022-12-31"), "研究期 2023-26": ("2023-01-01", "2026-09-18")}


def stamp(dt):
    return np.where(pd.to_datetime(dt) < pd.Timestamp("2023-08-28"), 0.001, 0.0005)


def roundtrip_cost(dt, slip):
    return 0.00025 * 2 + stamp(dt) + 2 * slip


# ------------------------------------------------------------------ desc
def desc():
    os.makedirs(OUT, exist_ok=True)
    pd.set_option("display.width", 250)
    S = pd.read_parquet(os.path.join(DATA, "senti_stock.parquet"), columns=KEY + STOCK_F + ["yon", "yid"])
    S = S[S["datetime"] >= "2016-01-01"]
    S["年"] = S["datetime"].dt.year
    print("== 中证1000 池内等权：每只股票每天的平均隔夜收益(yon) / 次日日内收益(yid)，单位 % ==")
    t = S.groupby("年")[["yon", "yid"]].mean().mul(100).round(3)
    t["隔夜为正的天数占比"] = S.groupby(["年", "datetime"])["yon"].mean().gt(0).groupby(level=0).mean().round(2)
    print(t.to_string())
    print("全期：yon %.3f%%  yid %.3f%%；一轮成本（万2.5×2 + 印花税万5 + 滑点万5×2）约 0.20%%" %
          (S["yon"].mean() * 100, S["yid"].mean() * 100))
    rows = []
    for per, (s0, e0) in PER_ALL.items():
        P = S[(S["datetime"] >= s0) & (S["datetime"] <= e0)]
        for f in STOCK_F:
            x = P[["datetime", f, "yon", "yid"]].dropna(subset=[f, "yon"])
            if x[f].nunique() < 20:     # 0/1 型特征：直接分组
                g = x.groupby(x[f].clip(upper=3).round())[["yon", "yid"]].mean()
                for b, r in g.iterrows():
                    rows.append(dict(区间=per, 特征=f, 分组=f"={b:g}", yon=r["yon"], yid=r["yid"]))
                continue
            q = x.groupby("datetime")[f].rank(pct=True)
            x["d"] = np.minimum((q * 10).astype(int), 9)
            g = x.groupby("d")[["yon", "yid"]].mean()
            for b, r in g.iterrows():
                rows.append(dict(区间=per, 特征=f, 分组=f"D{b + 1}", yon=r["yon"], yid=r["yid"]))
    d = pd.DataFrame(rows)
    d.to_csv(f"{OUT}/desc_sorts.csv", index=False, encoding="utf-8-sig", float_format="%.5f")
    v = d.pivot_table(index=["特征", "分组"], columns="区间", values="yon").mul(100).round(3)
    print("\n== 单特征分组后的平均隔夜收益 yon（%），D1=最小 D10=最大 ==")
    print(v.to_string())


# ------------------------------------------------------------------ train
def train():
    import gc
    import lightgbm as lgb
    os.makedirs("output/preds", exist_ok=True)
    os.makedirs(OUT, exist_ok=True)
    base = dict(objective="regression", metric="l2", colsample_bytree=0.8, learning_rate=0.03, subsample=0.8,
                subsample_freq=1, lambda_l1=10.0, lambda_l2=500.0, max_depth=6, num_leaves=31, min_data_in_leaf=1000,
                verbosity=-1, num_threads=2, seed=0)
    only = sys.argv[2].split(",") if len(sys.argv) > 2 else ["on"]
    periods = WF if os.environ.get("SENTI_WF") == "1" else PERIODS
    imps = []
    for per, cfg in periods.items():
        for tag in only:
            ycol = "y" + tag[:2]          # on / id：原始收益；onx / idx：减去当日均值（只学选股）
            xs = tag.endswith("x")

            def tf(k_, y_):
                y_ = np.clip(y_, -0.1, 0.1)
                if xs:
                    y_ = y_ - pd.Series(y_).groupby(k_["datetime"].values).transform("mean").values
                return y_.astype("float32")
            ktr, Xtr, ytr, feats = load_xy(*cfg["train"], sample=int(os.environ.get("SENTI_SAMPLE", "2")), ycol=ycol)
            dtr = lgb.Dataset(Xtr, tf(ktr, ytr), feature_name=feats, free_raw_data=True)
            dtr.construct()
            del Xtr, ytr, ktr
            gc.collect()
            kva, Xva, yva, _ = load_xy(*cfg["valid"], ycol=ycol)
            dva = lgb.Dataset(Xva, tf(kva, yva), reference=dtr)
            dva.construct()
            del Xva, yva, kva
            gc.collect()
            bst = lgb.train(base, dtr, num_boost_round=2000, valid_sets=[dva],
                            callbacks=[lgb.early_stopping(100, verbose=False)])
            del dtr, dva
            gc.collect()
            outs = []
            for yr in range(int(cfg["test"][0][:4]), int(cfg["test"][1][:4]) + 1):
                s0, e0 = max(f"{yr}-01-01", cfg["test"][0]), min(f"{yr}-12-31", cfg["test"][1])
                kte, Xte, yte, _ = load_xy(s0, e0, need_y=False, ycol=ycol)
                kte["score"] = bst.predict(Xte, num_iteration=bst.best_iteration).astype("float32")
                kte["y"] = yte
                outs.append(kte)
                del Xte
                gc.collect()
            out = pd.concat(outs, ignore_index=True)
            out[KEY + ["score"]].to_parquet(f"output/preds/overnight_{tag}{cfg['suf']}.parquet", index=False)
            t = out.dropna(subset=["y"]).copy()
            t["rk"] = t.groupby("datetime")["score"].rank(ascending=False)
            ic = t.groupby("datetime")[["score", "y"]].corr(method="spearman").xs("score", level=1)["y"]
            print(per, tag, "best_iter", bst.best_iteration, f"| RankIC {ic.mean():.3f} ICIR {ic.mean() / ic.std():.2f}",
                  " | 每日前N平均 y（%）：" + "  ".join(f"前{n} {t[t.rk <= n]['y'].mean() * 100:.3f}" for n in [1, 3, 5, 10, 30]),
                  f"| 全体 {t['y'].mean() * 100:.3f}", flush=True)
            imp = pd.Series(bst.feature_importance("gain"), index=feats).sort_values(ascending=False)
            imps.append(imp.rename(f"{per}_{tag}"))
            print("  重要特征：", ", ".join(imp.index[:15]), flush=True)
            del out, t
            gc.collect()
    fn = f"{OUT}/importance.csv"
    old = pd.read_csv(fn, index_col=0) if os.path.exists(fn) else None
    new = pd.concat(imps, axis=1)
    (new if old is None else old.drop(columns=[c for c in new.columns if c in old.columns]).join(new, how="outer")).to_csv(
        fn, encoding="utf-8-sig", float_format="%.1f")


# ------------------------------------------------------------------ bt
def daily(pred, lab, k, slip, thr=None, max_amt_rank=None):
    """每天按分数取前 k（只在可买入的股票里），等权；thr：预测值低于 thr 的不买（可空仓）。返回每日净收益"""
    x = pred.merge(lab, on=KEY).dropna(subset=["y"])
    x = x.sort_values(["datetime", "score"], ascending=[True, False])
    if thr is not None:
        x = x[x["score"] > thr]
    top = x.groupby("datetime").head(k)
    g = top.groupby("datetime")
    gross = g["y"].sum() / k                      # 没买满 k 只时剩余现金收益为 0
    n = g.size()
    cost = roundtrip_cost(gross.index, slip) * n / k
    days = lab["datetime"].drop_duplicates().sort_values()
    days = days[(days >= x["datetime"].min()) & (days <= x["datetime"].max())]
    r = (gross - cost).reindex(days).fillna(0.0)
    return r, gross.reindex(days).fillna(0.0), (n / k).reindex(days).fillna(0.0)


def perf(r):
    nav = (1 + r).cumprod()
    yrs = len(r) / 243
    ann = nav.iloc[-1] ** (1 / yrs) - 1 if nav.iloc[-1] > 0 else -1.0
    return ann, (nav / nav.cummax() - 1).min()


def bt():
    os.makedirs(OUT, exist_ok=True)
    pd.set_option("display.width", 250)
    tag = sys.argv[2] if len(sys.argv) > 2 else "on"
    lab = pd.read_parquet(os.path.join(DATA, "senti_stock.parquet"), columns=KEY + ["y" + tag[:2]]).rename(columns={"y" + tag[:2]: "y"})
    periods = dict(PERIODS)
    if os.path.exists(f"output/preds/overnight_{tag}_wf.parquet"):
        periods.update(WF)
    rows, yrow = [], []
    for per, cfg in periods.items():
        pred = pd.read_parquet(f"output/preds/overnight_{tag}{cfg['suf']}.parquet")
        pred = pred[(pred["datetime"] >= cfg["test"][0]) & (pred["datetime"] <= cfg["test"][1])]
        for k in [1, 3, 5, 10, 20]:
            for thr in [None, 0.0, 0.001, 0.002, 0.003]:
                for slip in [0.0, 0.0005, 0.001]:
                    r, gross, expo = daily(pred, lab, k, slip, thr)
                    ann, mdd = perf(r)
                    rows.append(dict(区间=per, 持股=k, 门槛=("不设" if thr is None else f"{thr:.1%}"), 滑点=slip,
                                     年化=ann, 最大回撤=mdd, 日均毛收益=gross.mean(), 交易天数占比=(expo > 0).mean()))
                    if slip == 0.0005 and k in (5, 10) and thr in (None, 0.002):
                        yrow.append(dict(区间=per, 持股=k, 门槛=("不设" if thr is None else f"{thr:.1%}"),
                                         **{f"Y{y}": (1 + v).prod() - 1 for y, v in r.groupby(r.index.year)}))
    d = pd.DataFrame(rows)
    d.to_csv(f"{OUT}/bt_{tag}.csv", index=False, encoding="utf-8-sig", float_format="%.5f")
    d["v"] = [f"{a:.1%}/{b:.0%}" for a, b in zip(d["年化"], d["最大回撤"])]
    print(f"== {tag}：年化/最大回撤（滑点为单边）==")
    print(d.pivot_table(index=["持股", "门槛"], columns=["区间", "滑点"], values="v", aggfunc="first").to_string())
    print("\n== 日均毛收益（%，扣成本前）与交易天数占比 ==")
    e = d[d["滑点"] == 0.0].pivot_table(index=["持股", "门槛"], columns="区间", values=["日均毛收益", "交易天数占比"])
    e["日均毛收益"] *= 100
    print(e.round(3).to_string())
    print("\n== 分年（滑点万5）==")
    print(pd.DataFrame(yrow).set_index(["区间", "持股", "门槛"]).round(3).to_string())


if __name__ == "__main__":
    {"desc": desc, "train": train, "bt": bt}[sys.argv[1]]()
