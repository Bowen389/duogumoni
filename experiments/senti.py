"""
进攻版（方案 A）：市场情绪周期 + 预测“大涨概率”，中证1000 内持 1-3 只
  python experiments/senti.py build     # 情绪指标 + 个股短线特征 + 标签 → {DATA}/senti_market.parquet、{DATA}/senti_stock.parquet
  python experiments/senti.py train [cls,q90,dn]   # 两段各训练：cls=3日涨≥10% 分类，q90=90%分位数回归，dn=3日跌≥10% 分类
  python experiments/senti.py grid      # 策略网格 → output/senti/grid.csv
  SENTI_WF=1 python experiments/senti.py train cls,dn   # 滚动检验：2023-08 前训练，最近一年（2025-09~2026-09）完全未见过
  python experiments/senti_wf.py        # 用滚动模型 + 实盘 5 日模型检验最近一年

情绪指标（T 日收盘可知，全部来自面板：曾属中证1000 的约 2700 只中小盘股）
  涨停/跌停家数占比、炸板率（盘中触及涨停但没封住）、最高连板数、2连板以上家数、
  昨日涨停股今日平均收益与平均隔夜收益（接力赚钱效应）、上涨家数占比、中位数涨幅、大跌（<-7%）家数、成交额相对 20 日均值
个股短线特征：连板数、今日涨停/触板/炸板、10 日涨停次数、1/3/5 日涨幅、距 20 日最高、振幅、收盘位置、隔夜跳空、量比
标签：T+1 开盘买入 → T+3 收盘，收益 y3；分类目标 y3 ≥ 10%；T+1 开盘一字涨停或停牌（买不进）的样本剔除
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from engine.common import DATA, KEY  # noqa: E402

PCOLS = ["datetime", "instrument", "open", "high", "low", "close", "raw_close", "change", "factor", "amount", "lim",
         "up_lim", "dn_lim", "susp", "in_pool", "ret", "ret_on", "ret_id", "up_open", "dn_open", "nbo"]
STOCK_F = ["LB", "UP0", "TOUCH0", "ZB0", "NUP10", "R1", "R3", "R5", "DH20", "AMP", "CPOS", "GAP", "VR5", "VR20"]
MKT_F = ["M_UP", "M_DN", "M_ZB", "M_LBMAX", "M_LB2", "M_RELAY", "M_RELAY_ON", "M_BREADTH", "M_MED", "M_BIGDN", "M_AMT"]
WF = {"滚动近一年": dict(train=("2015-01-01", "2023-08-31"), valid=("2023-09-01", "2025-09-12"),
                    test=("2025-09-19", "2026-09-18"), suf="_wf")}   # 最近一年完全未见过（新模型）
PERIODS = {"研究期": dict(train=("2015-01-01", "2020-12-31"), valid=("2021-01-01", "2022-12-31"),
                       test=("2023-01-01", "2026-09-18"), suf=""),
           "样本外": dict(train=("2015-01-01", "2018-12-31"), valid=("2019-01-01", "2019-12-31"),
                       test=("2020-01-01", "2022-12-31"), suf="_oos")}


# ------------------------------------------------------------------ build
def build():
    pdir = os.path.join(DATA, "panel")
    stock_parts, mk = [], []
    for fn in sorted(x for x in os.listdir(pdir) if x.endswith(".parquet")):
        p = pd.read_parquet(os.path.join(pdir, fn), columns=PCOLS)
        p = p.sort_values(["instrument", "datetime"]).reset_index(drop=True)
        g = p.groupby("instrument", sort=False)
        act = ~p["susp"]
        prev_raw = p["raw_close"] / (1 + p["change"])
        up_p = np.round(prev_raw * (1 + p["lim"]) + 1e-6, 2)
        touch = (p["high"] / p["factor"] >= up_p - 0.0051) & act
        up = p["up_lim"] & act
        zb = touch & ~up
        # 连板数
        blk = (~up).groupby(p["instrument"], sort=False).cumsum()
        lb = up.groupby([p["instrument"], blk], sort=False).cumsum().astype("float32")
        f = pd.DataFrame({"datetime": p["datetime"], "instrument": p["instrument"]})
        f["LB"] = lb
        f["UP0"] = up.astype("float32")
        f["TOUCH0"] = touch.astype("float32")
        f["ZB0"] = zb.astype("float32")
        f["NUP10"] = up.astype(float).groupby(p["instrument"], sort=False).transform(lambda x: x.rolling(10, 1).sum()).astype("float32")
        c = p["close"]
        f["R1"] = (c / g["close"].shift(1) - 1).astype("float32")
        f["R3"] = (c / g["close"].shift(3) - 1).astype("float32")
        f["R5"] = (c / g["close"].shift(5) - 1).astype("float32")
        f["DH20"] = (c / g["high"].transform(lambda x: x.rolling(20, 1).max()) - 1).astype("float32")
        pc = g["close"].shift(1)
        f["AMP"] = ((p["high"] - p["low"]) / pc).astype("float32")
        f["CPOS"] = ((c - p["low"]) / (p["high"] - p["low"]).replace(0, np.nan)).astype("float32")
        f["GAP"] = p["ret_on"].astype("float32")
        am = p["amount"].where(act)
        f["VR5"] = (am / g["amount"].transform(lambda x: x.where(x > 0).rolling(5, 1).mean().shift(1))).astype("float32")
        f["VR20"] = (am / g["amount"].transform(lambda x: x.where(x > 0).rolling(20, 1).mean().shift(1))).astype("float32")
        # 标签：T+1 开盘买入，T+3 收盘
        o1 = g["open"].shift(-1)
        c3 = g["close"].shift(-3)
        y3 = (c3 / o1 - 1).astype("float32")
        y3[p["nbo"]] = np.nan
        f["y3"] = y3
        # 方案 C 标签。yon：T 日收盘买（收盘封涨停/停牌买不进 → 剔除）→ T+1 开盘卖（开盘一字跌停/停牌卖不出 → 按 T+1 收盘卖）
        c1 = g["close"].shift(-1)
        nsell = (g["dn_open"].shift(-1).fillna(False) | g["susp"].shift(-1).fillna(False)).astype(bool)
        yon = pd.Series(np.where(nsell, c1 / c - 1, o1 / c - 1), index=p.index).astype("float32")
        yon[p["up_lim"] | p["susp"]] = np.nan
        f["yon"] = yon
        # yid：T+1 开盘买（一字涨停/停牌买不进 → 剔除）→ T+1 收盘卖（需要底仓做 T）
        yid = (c1 / o1 - 1).astype("float32")
        yid[p["nbo"]] = np.nan
        f["yid"] = yid
        # 下一日开盘到收盘 / 隔夜，用于检查
        f["in_pool"] = p["in_pool"]
        stock_parts.append(f[p["in_pool"]].reset_index(drop=True))
        # 市场情绪：按日汇总（加和，后面合并各分块）
        prev_up = up.groupby(p["instrument"], sort=False).shift(1).fillna(False).astype(bool)
        # 情绪统计剔除上市不满 60 个交易日的新股（新股连续一字板会严重扭曲涨停/连板/接力指标）
        age = act.astype(int).groupby(p["instrument"], sort=False).cumsum()
        act, up, touch, zb = act & (age >= 60), up & (age >= 60), touch & (age >= 60), zb & (age >= 60)
        prev_up = prev_up & (age >= 61)
        lb = lb.where(age >= 60, 0)
        m = pd.DataFrame({"datetime": p["datetime"], "n": act.astype(int), "up": up.astype(int),
                          "dn": (p["dn_lim"] & act).astype(int), "touch": touch.astype(int), "zb": zb.astype(int),
                          "lb2": (lb >= 2).astype(int), "lbmax": lb,
                          "relay_s": np.where(prev_up & act, p["ret"], 0.0), "relay_n": (prev_up & act).astype(int),
                          "relayo_s": np.where(prev_up & act, p["ret_on"], 0.0),
                          "pos": ((p["ret"] > 0) & act).astype(int), "bigdn": ((p["ret"] < -0.07) & act).astype(int),
                          "amt": p["amount"].where(act, 0.0)})
        agg = m.groupby("datetime").agg(n=("n", "sum"), up=("up", "sum"), dn=("dn", "sum"), touch=("touch", "sum"),
                                        zb=("zb", "sum"), lb2=("lb2", "sum"), lbmax=("lbmax", "max"),
                                        relay_s=("relay_s", "sum"), relay_n=("relay_n", "sum"),
                                        relayo_s=("relayo_s", "sum"), pos=("pos", "sum"), bigdn=("bigdn", "sum"),
                                        amt=("amt", "sum"))
        med = p.loc[act, ["datetime", "ret"]].groupby("datetime")["ret"].apply(lambda x: x.values.astype("float32"))
        mk.append((agg, med))
        print(fn, len(f), flush=True)
        del p, f, m
    A = mk[0][0]
    for a, _ in mk[1:]:
        A = A.add(a, fill_value=0)   # lbmax 的加和无意义，下面单独取最大值
    A["lbmax"] = pd.concat([a["lbmax"] for a, _ in mk], axis=1).max(axis=1)
    meds = {}
    for _, md in mk:
        for d, v in md.items():
            meds.setdefault(d, []).append(v)
    M = pd.DataFrame(index=A.index)
    M["M_UP"] = A["up"] / A["n"]
    M["M_DN"] = A["dn"] / A["n"]
    M["M_ZB"] = A["zb"] / A["touch"].replace(0, np.nan)
    M["M_LBMAX"] = A["lbmax"]
    M["M_LB2"] = A["lb2"] / A["n"]
    M["M_RELAY"] = A["relay_s"] / A["relay_n"].replace(0, np.nan)
    M["M_RELAY_ON"] = A["relayo_s"] / A["relay_n"].replace(0, np.nan)
    M["M_BREADTH"] = A["pos"] / A["n"]
    M["M_MED"] = pd.Series({d: float(np.nanmedian(np.concatenate(v))) for d, v in meds.items()})
    M["M_BIGDN"] = A["bigdn"] / A["n"]
    M["M_AMT"] = A["amt"] / A["amt"].rolling(20, 5).mean()
    M = M.sort_index()
    # 平滑与相对水平（均只用 T 日及以前）
    for ccol in MKT_F:
        M[ccol + "_5"] = M[ccol].rolling(5, 1).mean()
        M[ccol + "_z"] = (M[ccol] - M[ccol].rolling(120, 20).mean()) / (M[ccol].rolling(120, 20).std() + 1e-9)
    M.index.name = "datetime"
    M = M.reset_index().astype({c: "float32" for c in M.columns if c.startswith("M_")})
    M.to_parquet(os.path.join(DATA, "senti_market.parquet"), index=False)
    S = pd.concat(stock_parts, ignore_index=True)
    S.to_parquet(os.path.join(DATA, "senti_stock.parquet"), index=False)
    print("market", M.shape, "stock", S.shape, "y3>=10% 占比", round((S["y3"] >= 0.10).mean(), 4), flush=True)


# ------------------------------------------------------------------ train
def load_xy(start, end, sample=1, need_y=True, ycol="y3"):
    """省内存版：原有散户+行为因子矩阵 + 个股短线特征 + 市场情绪，按索引对齐写入预分配矩阵（不做 DataFrame 合并）"""
    import gc
    from engine import model as EM
    keys, X, names, _ = EM.load(["retail", "behavior"], start, end, 1, sample=sample, with_label=False)
    dt = keys["datetime"].values
    ins = pd.Categorical(keys["instrument"].values)
    del keys
    S = pd.read_parquet(os.path.join(DATA, "senti_stock.parquet"), columns=KEY + STOCK_F + [ycol],
                        filters=[("datetime", ">=", pd.Timestamp(start)), ("datetime", "<=", pd.Timestamp(end))])
    S = S[S["instrument"].isin(ins.categories)].reset_index(drop=True)
    S = S.drop_duplicates(KEY).reset_index(drop=True)
    sidx = pd.MultiIndex.from_arrays([S["datetime"].values, pd.Categorical(S["instrument"].values, categories=ins.categories)])
    pos = sidx.get_indexer(pd.MultiIndex.from_arrays([dt, ins]))
    del sidx
    y = np.where(pos >= 0, S[ycol].values[np.maximum(pos, 0)], np.nan).astype("float32")
    keep = ~np.isnan(y) if need_y else np.ones(len(y), dtype=bool)
    M = pd.read_parquet(os.path.join(DATA, "senti_market.parquet")).set_index("datetime")
    mcols = list(M.columns)
    mpos = M.index.get_indexer(dt)
    n = int(keep.sum())
    out = np.empty((n, X.shape[1] + len(STOCK_F) + len(mcols)), dtype="float32")
    out[:, :X.shape[1]] = X[keep]
    del X
    gc.collect()
    sv = S[STOCK_F].values.astype("float32")
    del S
    p = pos[keep]
    blk = sv[np.maximum(p, 0)]
    blk[p < 0] = np.nan
    out[:, len(names):len(names) + len(STOCK_F)] = blk
    del blk, sv
    mv = M.values.astype("float32")
    mp = mpos[keep]
    blk = mv[np.maximum(mp, 0)]
    blk[mp < 0] = np.nan
    out[:, len(names) + len(STOCK_F):] = blk
    del blk
    out[~np.isfinite(out)] = np.nan
    kk = pd.DataFrame({"datetime": dt[keep], "instrument": np.asarray(ins)[keep]})
    gc.collect()
    return kk, out, y[keep], names + STOCK_F + mcols


def train():
    import gc
    import lightgbm as lgb
    os.makedirs("output/preds", exist_ok=True)
    os.makedirs("output/senti", exist_ok=True)
    base = dict(colsample_bytree=0.8, learning_rate=0.03, subsample=0.8, subsample_freq=1, lambda_l1=10.0,
                lambda_l2=500.0, max_depth=6, num_leaves=31, min_data_in_leaf=1000, verbosity=-1, num_threads=2, seed=0)
    TASKS = {"cls": (dict(base, objective="binary", metric="auc"), lambda y: (y >= 0.10).astype("float32")),
             "q90": (dict(base, objective="quantile", alpha=0.9, metric="quantile"), lambda y: np.clip(y, -0.5, 1.0)),
             "dn": (dict(base, objective="binary", metric="auc"), lambda y: (y <= -0.10).astype("float32"))}
    only = sys.argv[2].split(",") if len(sys.argv) > 2 else ["cls", "q90"]
    TASKS = {k: v for k, v in TASKS.items() if k in only}
    imps = []
    periods = WF if os.environ.get("SENTI_WF") == "1" else PERIODS
    for per, cfg in periods.items():
        for tag, (params, tf) in TASKS.items():
            _, Xtr, ytr, feats = load_xy(*cfg["train"], sample=int(os.environ.get("SENTI_SAMPLE", "2")))  # 训练集抽样，省内存
            dtr = lgb.Dataset(Xtr, tf(ytr), feature_name=feats, free_raw_data=True)
            dtr.construct()
            del Xtr, ytr
            gc.collect()
            _, Xva, yva, _ = load_xy(*cfg["valid"])
            dva = lgb.Dataset(Xva, tf(yva), reference=dtr)
            dva.construct()
            del Xva, yva
            gc.collect()
            bst = lgb.train(params, dtr, num_boost_round=2000, valid_sets=[dva],
                            callbacks=[lgb.early_stopping(100, verbose=False)])
            del dtr, dva
            gc.collect()
            outs = []
            y0, y1 = int(cfg["test"][0][:4]), int(cfg["test"][1][:4])
            for yr in range(y0, y1 + 1):                                 # 按年预测
                s0, e0 = max(f"{yr}-01-01", cfg["test"][0]), min(f"{yr}-12-31", cfg["test"][1])
                kte, Xte, yte, _ = load_xy(s0, e0, need_y=False)
                kte["score"] = bst.predict(Xte, num_iteration=bst.best_iteration).astype("float32")
                kte["y3"] = yte
                outs.append(kte)
                del Xte
                gc.collect()
            out = pd.concat(outs, ignore_index=True)
            out[KEY + ["score"]].to_parquet(f"output/preds/senti_{tag}{cfg['suf']}.parquet", index=False)
            t = out.dropna(subset=["y3"]).copy()
            t["rk"] = t.groupby("datetime")["score"].rank(ascending=False)
            top = t[t.rk <= 10]
            print(per, tag, "best_iter", bst.best_iteration,
                  f"| 每日前10名：3日涨≥10% 命中率 {(top.y3 >= 0.10).mean():.1%}（全体 {(t.y3 >= 0.10).mean():.1%}）"
                  f"，平均 y3 {top.y3.mean():.2%}（全体 {t.y3.mean():.2%}），y3 中位数 {top.y3.median():.2%}", flush=True)
            imp = pd.Series(bst.feature_importance("gain"), index=feats).sort_values(ascending=False)
            imps.append(imp.rename(f"{per}_{tag}"))
            print("  重要特征：", ", ".join(imp.index[:12]), flush=True)
            del out, t
            gc.collect()
    fn = "output/senti/importance.csv"
    old = pd.read_csv(fn, index_col=0) if os.path.exists(fn) else None
    new = pd.concat(imps, axis=1)
    (new if old is None else old.drop(columns=[c for c in new.columns if c in old.columns]).join(new, how="outer")).to_csv(
        fn, encoding="utf-8-sig", float_format="%.1f")


# ------------------------------------------------------------------ grid
def simulate(mk, rkS, ordS, k=3, M=60, hold=0, exec_at="open", timing=None, buy_cost=0.0005, sell_cost=0.0010,
             slip=0.001):
    """hold>0：持满 hold 天强制卖出（不看排名）；hold=0：排名跌出前 M 才卖"""
    m, days = mk.m, mk.days
    hd = {}
    out = []
    w = 1.0 / k
    for i in range(1, len(days)):
        t, s = days[i], days[i - 1]
        rk = rkS.get(s, {})
        on = True if timing is None else bool(timing.get(s, True))
        sells, keeps = [], []
        for x, h in hd.items():
            want = (not on) or not m["in_pool"][i, x] or ((h >= hold) if hold else (rk.get(x, 1e9) > M))
            if exec_at == "mixed":     # 前一日尾盘卖出（用前一日收盘前的信号近似）：看前一日是否跌停/停牌
                blocked = m["susp"][i - 1, x] or m["dn_lim"][i - 1, x]
            else:
                blocked = m["susp"][i, x] or (m["dn_lim"][i, x] if exec_at == "close" else m["dn_open"][i, x])
            (sells if want and not blocked else keeps).append(x)
        buys = []
        if on and k - len(keeps) > 0:
            for x in ordS.get(s, []):
                if len(buys) >= k - len(keeps):
                    break
                if x in hd or not m["in_pool"][i, x]:
                    continue
                if not (m["susp"][i, x] or (m["up_lim"][i, x] if exec_at == "close" else m["up_open"][i, x])):
                    buys.append(x)
        r = 0.0
        if exec_at == "close":
            for x in keeps + sells:
                r += w * m["ret"][i, x]
        else:
            for x in keeps:
                r += w * m["ret"][i, x]
            if exec_at != "mixed":     # mixed：卖出已在前一日尾盘完成，今天不再承担隔夜
                for x in sells:
                    r += w * m["ret_on"][i, x]
            for x in buys:
                r += w * m["ret_id"][i, x]
        r -= w * (len(sells) * (sell_cost + slip) + len(buys) * (buy_cost + slip))
        hd = {x: hd[x] + 1 for x in keeps}
        for x in buys:
            hd[x] = 1
        out.append((t, r, len(hd), w * (len(sells) + len(buys)) / 2))
    return pd.DataFrame(out, columns=["datetime", "ret", "n_hold", "turnover"]).set_index("datetime")


def timings():
    M = pd.read_parquet(os.path.join(DATA, "senti_market.parquet")).set_index("datetime")
    b = pd.read_parquet(os.path.join(DATA, "bench.parquet")).set_index("datetime")["bench"]
    lvl = (1 + b).cumprod()
    vote = sum((lvl > lvl.rolling(mm).mean()).astype(int) for mm in [20, 40, 60, 80, 120])
    return {"不择时": None,
            "均线投票≥3": vote >= 3,
            "接力赚钱(5日)": M["M_RELAY_5"] > 0,
            "涨停多于跌停(5日)": M["M_UP_5"] > M["M_DN_5"],
            "连板高度≥4(5日均)": M["M_LBMAX_5"] >= 4,
            "非恐慌(大跌家数z<1)": M["M_BIGDN_z"] < 1.0}


def grid():
    import itertools
    from experiments.p2p import Market, stats
    from strategies.live import rank_blend
    TIM = timings()
    rows = []
    for per, cfg in PERIODS.items():
        mk = Market(*cfg["test"])
        suf = cfg["suf"]
        rd = lambda f: pd.read_parquet(f"output/preds/{f}{suf}.parquet")[KEY + ["score"]]  # noqa
        up, dn, h5 = rd("senti_cls"), rd("senti_dn"), rd("h5ens6")
        skew = up.merge(dn, on=KEY, suffixes=("", "_d"))
        skew["score"] = skew["score"] - skew["score_d"]
        skew = skew[KEY + ["score"]]
        scores = {"大涨概率": up, "不对称(涨-跌)": skew, "5日模型h5": h5,
                  "大涨概率+h5": rank_blend([up, h5]), "不对称+h5": rank_blend([skew, h5])}
        for sname, sc in scores.items():
            R = mk.ranks(sc, depth=100)
            for k, (hold, M), tn in itertools.product([1, 2, 3], [(1, 0), (2, 0), (3, 0), (5, 0), (0, 10), (0, 30), (0, 60)],
                                                      list(TIM)):
                res = simulate(mk, *R, k=k, M=M, hold=hold, timing=TIM[tn])
                rows.append(dict(区间=per, 打分=sname, 持股=k, 卖出=f"持{hold}天" if hold else f"跌出前{M}", 择时=tn,
                                 **stats(res, k)))
        print(per, "done", flush=True)
        del mk
    d = pd.DataFrame(rows)
    d.to_csv("output/senti/grid.csv", index=False, encoding="utf-8-sig", float_format="%.4f")


if __name__ == "__main__":
    {"build": build, "train": train, "grid": grid}[sys.argv[1]]()
