"""
“低位首板 + 次日低开”深入检验与优化（中证1000 股票池，日线）
  python experiments/fb_dip.py build     # 生成事件表 {DATA}/fb_events.parquet（每个首板次日一行：特征 + 多种卖出方式的净收益与持有路径）
  python experiments/fb_dip.py study     # 单变量分组（设计期 2015-2019 / 验证期 2020-2026）、卖出方式、滑点
  python experiments/fb_dip.py port      # 组合资金曲线：槽位数、排序规则、分年
  python experiments/fb_dip.py watch     # 每日收盘后：列出明天的候选股与买入价格区间（明早 9:25 看开盘价是否落在区间内）

最终版本（FINAL，见 final_mask）：10cm 股票；首板前收盘 < 60 日收盘中位数；首板当天不是开盘即涨停；
  次日低开 2%~7%；前一日中证1000 收盘在 20 日均线下；前一日全市场跌停家数占比 < 3%（恐慌日不做）
  开盘买入，次日收盘卖出；每笔 50% 资金，最多同时 2 只

事件：T-1 日涨停且 T-2 日未涨停（首板）；T 日未停牌、开盘未一字涨停、在池内
买入：T 日开盘价（集合竞价后即可知道低开幅度，下单无未来信息）
卖出（顺延规则：卖出日收盘跌停/停牌 → 卖不出，顺延到下一个能卖的收盘；开盘卖遇一字跌停 → 改当日收盘，仍不行继续顺延；最多顺延 8 天）
  C1  T+1 收盘卖
  O1  T+1 开盘卖
  C2  T+2 收盘卖
  TPx T+1 挂止盈单 +x%：开盘已超过 → 开盘价成交；盘中最高价触及 → 按 x% 成交；否则 T+1 收盘卖
  RUN T+1 收盘涨停 → 继续持有到 T+2 收盘；否则 T+1 收盘卖
成本：佣金万2.5×2 + 印花税（2023-08-28 前千1，之后万5）+ 单边滑点 slip（默认千1）
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
from engine.common import DATA, KEY  # noqa: E402

EV = os.path.join(DATA, "fb_events.parquet")
OUT = "output/fb_dip"
MAXD = 8
EXITS = ["C1", "O1", "C2", "TP3", "TP5", "TP7", "RUN"]
SPLIT = pd.Timestamp("2020-01-01")


def stamp(dt):
    return np.where(pd.to_datetime(dt) < pd.Timestamp("2023-08-28"), 0.001, 0.0005)


def fixed_cost(dt):
    return 0.0005 + stamp(dt)


# ------------------------------------------------------------------ build
def build():
    pdir = os.path.join(DATA, "panel")
    cols = KEY + ["open", "high", "low", "close", "amount", "lim", "up_lim", "dn_lim", "susp", "in_pool", "up_open", "dn_open"]
    M = pd.read_parquet(os.path.join(DATA, "senti_market.parquet")).set_index("datetime")
    b = pd.read_parquet(os.path.join(DATA, "bench.parquet")).set_index("datetime")["bench"]
    lvl = (1 + b).cumprod()
    mk = pd.DataFrame({"B_MA20": lvl / lvl.rolling(20).mean() - 1, "B_MA60": lvl / lvl.rolling(60).mean() - 1,
                       "B_R5": lvl / lvl.shift(5) - 1})
    mk = mk.join(M[["M_UP", "M_DN", "M_ZB", "M_RELAY", "M_RELAY_ON", "M_BREADTH", "M_LBMAX", "M_UP_5", "M_DN_5",
                    "M_RELAY_5", "M_BIGDN_z", "M_AMT_z"]], how="left")
    mk = mk.shift(1)   # T 日开盘时只知道 T-1 日收盘的市场状态
    parts = []
    for fn in sorted(x for x in os.listdir(pdir) if x.endswith(".parquet")):
        p = pd.read_parquet(os.path.join(pdir, fn), columns=cols,
                            filters=[("datetime", ">=", pd.Timestamp("2014-12-01"))]).sort_values(KEY).reset_index(drop=True)
        g = p.groupby("instrument", sort=False)
        sh = lambda c, k: g[c].shift(k)  # noqa: E731
        up = p["up_lim"].astype(bool)
        f = p[KEY].copy()
        f["lim"] = p["lim"].astype("float32")
        f["gap"] = (p["open"] / sh("close", 1) - 1).astype("float32")
        f["gapn"] = (f["gap"] / p["lim"]).astype("float32")                     # 低开幅度 / 涨跌停幅度
        c2 = sh("close", 2)
        for L in (20, 60, 120):
            f[f"pos{L}"] = (c2 / g["close"].transform(lambda x: x.shift(2).rolling(L, L // 3).median())).astype("float32")
        f["pre20"] = (c2 / sh("close", 22) - 1).astype("float32")               # 涨停前 20 日涨跌
        f["pre5"] = (c2 / sh("close", 7) - 1).astype("float32")
        am = p["amount"].where(~p["susp"])
        am20 = g["amount"].transform(lambda x: x.where(x > 0).rolling(20, 5).mean())
        f["vr_lim"] = (sh("amount", 1) / am20.groupby(p["instrument"]).shift(2)).astype("float32")   # 首板当天量比
        f["amt20"] = np.log(am20.groupby(p["instrument"]).shift(2)).astype("float32")               # 流动性（规模代理）
        o1, h1, l1, c1 = sh("open", 1), sh("high", 1), sh("low", 1), sh("close", 1)
        f["yizi"] = ((h1 == l1) & sh("up_lim", 1).fillna(False).astype(bool)).astype("float32")      # 首板为一字板
        f["open_lim"] = (sh("up_open", 1).fillna(False).astype(bool)).astype("float32")              # 首板开盘即涨停
        f["amp_lim"] = ((h1 - l1) / sh("close", 2)).astype("float32")                                # 首板当日振幅
        f["o_lim"] = (o1 / sh("close", 2) - 1).astype("float32")                                     # 首板当日开盘涨幅
        nup = up.astype(float).groupby(p["instrument"]).transform(lambda x: x.shift(2).rolling(60, 1).sum())
        f["nup60"] = nup.astype("float32")                                                           # 首板前 60 日涨停次数
        f["max20"] = g["close"].transform(lambda x: (x / x.shift(1) - 1).shift(2).rolling(20, 5).max()).astype("float32")
        # 事件条件
        ev = (sh("up_lim", 1).fillna(False).astype(bool) & ~sh("up_lim", 2).fillna(False).astype(bool)
              & ~p["susp"] & ~p["up_open"] & p["in_pool"])
        # 卖出：未来 MAXD+1 天的价格与可卖标记
        fut = {}
        for k in range(0, MAXD + 3):
            fut[("o", k)] = sh("open", -k).values
            fut[("h", k)] = sh("high", -k).values
            fut[("c", k)] = sh("close", -k).values
            fut[("dl", k)] = sh("dn_lim", -k).fillna(True).astype(bool).values | sh("susp", -k).fillna(True).astype(bool).values
            fut[("do", k)] = sh("dn_open", -k).fillna(True).astype(bool).values | sh("susp", -k).fillna(True).astype(bool).values
            fut[("ul", k)] = sh("up_lim", -k).fillna(False).astype(bool).values
        idx = np.where(ev.values)[0]
        if len(idx) == 0:
            continue
        E = f.iloc[idx].reset_index(drop=True)
        entry = p["open"].values[idx]
        E["entry"] = entry
        E["c0"] = p["close"].values[idx]

        def sell_close_from(k0, i):
            """从第 k0 天收盘开始找第一个能卖的收盘，返回 (天数 k, 价格)"""
            for k in range(k0, k0 + MAXD + 1):
                if k >= MAXD + 3:
                    break
                c = fut[("c", k)][i]
                if np.isnan(c):
                    return k, np.nan
                if not fut[("dl", k)][i]:
                    return k, c
            k = min(k0 + MAXD, MAXD + 2)
            return k, fut[("c", k)][i]
        res = {e: np.full(len(idx), np.nan) for e in EXITS}
        days = {e: np.zeros(len(idx), dtype=np.int8) for e in EXITS}
        paths = {e: [] for e in EXITS}   # 仅存卖出天数与价格，组合模拟时再按收盘价逐日计值
        for j, i in enumerate(idx):
            out = {}
            out["C1"] = sell_close_from(1, i)
            o1v = fut[("o", 1)][i]
            out["O1"] = (1, o1v) if (not np.isnan(o1v) and not fut[("do", 1)][i]) else sell_close_from(1, i)
            out["C2"] = sell_close_from(2, i)
            for x in (3, 5, 7):
                tgt = entry[j] * (1 + x / 100)
                if not np.isnan(o1v) and not fut[("do", 1)][i] and o1v >= tgt:
                    out[f"TP{x}"] = (1, o1v)
                elif not np.isnan(fut[("h", 1)][i]) and fut[("h", 1)][i] >= tgt and not fut[("do", 1)][i]:
                    out[f"TP{x}"] = (1, tgt)
                else:
                    out[f"TP{x}"] = out["C1"]
            out["RUN"] = sell_close_from(2, i) if fut[("ul", 1)][i] else out["C1"]
            for e in EXITS:
                k, px = out[e]
                res[e][j] = px / entry[j] - 1
                days[e][j] = k
        for e in EXITS:
            E[f"g_{e}"] = res[e].astype("float32")      # 毛收益（未扣成本）
            E[f"d_{e}"] = days[e]                       # 卖出在第几天（1 = T+1）
        # 逐日收盘价（组合按日计值用）
        for k in range(0, MAXD + 3):
            E[f"cl{k}"] = fut[("c", k)][idx].astype("float32")
        parts.append(E)
        print(fn, len(E), flush=True)
    X = pd.concat(parts, ignore_index=True)
    X = X[X["datetime"] >= "2015-06-01"]
    X = X.merge(mk, left_on="datetime", right_index=True, how="left")
    X.to_parquet(EV, index=False)
    print("events", X.shape)


def load(slip=0.001, low=True):
    X = pd.read_parquet(EV)
    X = X.dropna(subset=["g_C1"])
    for e in EXITS:
        X[f"n_{e}"] = X[f"g_{e}"] - fixed_cost(X["datetime"]) - 2 * slip
    X["期"] = np.where(X["datetime"] < SPLIT, "设计期15-19", "验证期20-26")
    X["年"] = X["datetime"].dt.year
    return X


def final_mask(X):
    return ((X["pos60"] < 1) & X["gap"].between(-0.07, -0.02) & (X["lim"] < 0.15) & (X["open_lim"] == 0)
            & (X["B_MA20"] < 0) & (X["M_DN"] < 0.03))


def base_mask(X, L=60, thr=1.0, lo=-0.07, hi=-0.02):
    return (X[f"pos{L}"] < thr) & X["gap"].between(lo, hi)


# ------------------------------------------------------------------ study
def summarize(s, col="n_C1"):
    return pd.Series({"笔数": len(s), "均值%": s[col].mean() * 100, "中位%": s[col].median() * 100,
                      "胜率": (s[col] > 0).mean(), "t": s[col].mean() / (s[col].std() / np.sqrt(max(len(s), 2)))})


def study():
    os.makedirs(OUT, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 300)
    X = load()
    B = X[base_mask(X)]
    print(f"基础版（60日低位、低开2-7%、T+1收盘卖、单边滑点千1）")
    print(B.groupby("期").apply(summarize).round(3).to_string())
    print("\n卖出顺延统计：T+1 收盘卖不出（跌停/停牌）的比例", round((B["d_C1"] > 1).mean(), 4),
          "；这些交易平均净收益", round(B.loc[B["d_C1"] > 1, "n_C1"].mean() * 100, 2), "%")

    print("\n== 1. 卖出方式（每笔净收益均值% / 胜率）==")
    rows = []
    for e in EXITS:
        for per, s in B.groupby("期"):
            rows.append(dict(卖出=e, 期=per, v=f"{s[f'n_{e}'].mean() * 100:.2f} / {(s[f'n_{e}'] > 0).mean():.0%}"))
    print(pd.DataFrame(rows).pivot(index="卖出", columns="期", values="v").to_string())

    print("\n== 2. 10cm vs 20cm ==")
    print(B.groupby([B["lim"].round(2), "期"]).apply(summarize).round(3).to_string())
    print("\n  以“低开幅度/涨跌停幅度”定义（gapn 在 -0.7~-0.2 之间，20cm 相当于低开 4-14%）")
    B2 = X[(X["pos60"] < 1) & X["gapn"].between(-0.7, -0.2)]
    print(B2.groupby([B2["lim"].round(2), "期"]).apply(summarize).round(3).to_string())

    print("\n== 3. 单变量三分位（在基础版内部；看设计期与验证期方向是否一致）==")
    feats = ["gap", "pos60", "pre20", "pre5", "vr_lim", "amt20", "amp_lim", "o_lim", "nup60", "max20",
             "B_MA20", "B_MA60", "B_R5", "M_UP", "M_DN", "M_ZB", "M_RELAY", "M_RELAY_ON", "M_BREADTH", "M_LBMAX",
             "M_UP_5", "M_RELAY_5", "M_BIGDN_z", "M_AMT_z"]
    rows = []
    for f in feats:
        s = B.dropna(subset=[f])
        try:
            q = pd.qcut(s[f].rank(method="first"), 3, labels=["低", "中", "高"])
        except ValueError:
            continue
        for per in ["设计期15-19", "验证期20-26"]:
            for lab in ["低", "中", "高"]:
                t = s[(q == lab) & (s["期"] == per)]
                rows.append(dict(特征=f, 期=per, 组=lab, v=t["n_C1"].mean() * 100, n=len(t)))
    d = pd.DataFrame(rows)
    d.to_csv(f"{OUT}/univariate.csv", index=False, encoding="utf-8-sig", float_format="%.4f")
    w = d.pivot_table(index="特征", columns=["期", "组"], values="v").round(2)
    sp = {}
    for per in ["设计期15-19", "验证期20-26"]:
        sp[per] = w[(per, "高")] - w[(per, "低")]
    w["高-低 设计"] = sp["设计期15-19"].round(2)
    w["高-低 验证"] = sp["验证期20-26"].round(2)
    w["方向一致"] = np.sign(w["高-低 设计"]) == np.sign(w["高-低 验证"])
    print(w.to_string())
    for c in ["yizi", "open_lim"]:
        print(f"\n  {c}：", B.groupby([c, "期"]).apply(summarize).round(3).to_string())

    print("\n== 4. 滑点敏感度（T+1 收盘卖，每笔净收益均值%）==")
    for slip in (0.0, 0.0005, 0.001, 0.002, 0.003):
        Y = load(slip)
        Y = Y[base_mask(Y)]
        print(f"  单边 {slip:.2%}：", "  ".join(f"{per} {s['n_C1'].mean() * 100:.2f}" for per, s in Y.groupby("期")))


# ------------------------------------------------------------------ port
def simulate(S, K=3, exit_="C1", order="gap", cap_days=None):
    """槽位制组合：每只占 1/K；T 日开盘买，按 exit_ 的卖出日与价格卖出；持有期间按收盘价逐日计值。返回日收益序列"""
    days = pd.Series(sorted(pd.read_parquet(os.path.join(DATA, "bench.parquet"))["datetime"]))
    days = days[days >= "2015-06-01"].reset_index(drop=True)
    di = {d: i for i, d in enumerate(days)}
    pnl = np.zeros(len(days))
    used = np.zeros(len(days))
    busy_until = []
    asc = order in ("gap", "pos60", "pre20")          # 这些按从小到大（低开更深/位置更低优先）
    S = S.sort_values(["datetime", order], ascending=[True, asc])
    for d, grp in S.groupby("datetime", sort=True):
        i = di.get(d)
        if i is None:
            continue
        free = K - sum(1 for e in busy_until if e >= i)
        for _, r in grp.head(max(free, 0)).iterrows():
            k_end = int(r[f"d_{exit_}"])
            if i + k_end >= len(days):
                continue
            entry = r["entry"] * (1 + 0.001) * (1 + 0.00025)
            prev = entry
            for k in range(0, k_end + 1):
                if k < k_end:
                    px = r[f"cl{k}"]
                else:
                    px = r["entry"] * (1 + r[f"g_{exit_}"]) * (1 - 0.001 - 0.00025 - float(stamp(d)))
                if np.isnan(px):
                    px = prev
                pnl[i + k] += (px / prev - 1) / K
                used[i + k] += 1 / K
                prev = px
            busy_until.append(i + k_end)
    r = pd.Series(pnl, index=days)
    return r, used.mean()


def perf(r):
    nav = (1 + r).cumprod()
    ann = nav.iloc[-1] ** (243 / len(r)) - 1
    return ann, (nav / nav.cummax() - 1).min()


def port():
    os.makedirs(OUT, exist_ok=True)
    pd.set_option("display.width", 250)
    X = load()
    cfg = {"基础版 60日低位 低开2-7%": base_mask(X)}
    extra = os.environ.get("FB_EXTRA")        # 例如 "pre20<0" 追加过滤条件（pandas query 语法）
    if extra:
        cfg[f"基础版 + {extra}"] = base_mask(X) & X.eval(extra)
    rows = []
    for name, m in cfg.items():
        S = X[m]
        for exit_ in ["C1", "RUN", "TP5"]:
            for K in (2, 3, 5):
                for order in ("gap", "vr_lim"):
                    r, u = simulate(S, K, exit_, order)
                    ann, mdd = perf(r)
                    a1, m1 = perf(r[r.index < SPLIT])
                    a2, m2 = perf(r[r.index >= SPLIT])
                    yr = r.groupby(r.index.year).apply(lambda x: (1 + x).prod() - 1)
                    rows.append(dict(版本=name, 卖出=exit_, 槽位=K, 排序=order, 年化=ann, 回撤=mdd, 平均仓位=u,
                                     设计期年化=a1, 设计期回撤=m1, 验证期年化=a2, 验证期回撤=m2, 亏损年数=int((yr < 0).sum()),
                                     **{f"Y{y}": v for y, v in yr.items()}))
    d = pd.DataFrame(rows)
    d.to_csv(f"{OUT}/port.csv", index=False, encoding="utf-8-sig", float_format="%.4f")
    show = d[["版本", "卖出", "槽位", "排序", "年化", "回撤", "平均仓位", "设计期年化", "验证期年化", "验证期回撤", "亏损年数"]].copy()
    for c in ["年化", "回撤", "平均仓位", "设计期年化", "验证期年化", "验证期回撤"]:
        show[c] = show[c].map(lambda v: f"{v:.1%}")
    print(show.to_string(index=False))


def watch():
    """用最新一个交易日收盘后的数据，列出明天可能触发的股票（是否低开 2-7% 要等明早 9:25 集合竞价结果）"""
    pdir = os.path.join(DATA, "panel")
    cols = KEY + ["open", "high", "low", "close", "raw_close", "lim", "up_lim", "up_open", "susp", "in_pool"]
    parts = []
    for fn in sorted(x for x in os.listdir(pdir) if x.endswith(".parquet")):
        p = pd.read_parquet(os.path.join(pdir, fn), columns=cols)
        last = p["datetime"].max()
        p = p[p["datetime"] >= last - pd.Timedelta(days=200)].sort_values(KEY)
        g = p.groupby("instrument", sort=False)
        p["pos60"] = g["close"].shift(1) / g["close"].transform(lambda x: x.shift(1).rolling(60, 20).median())
        p["prev_up"] = g["up_lim"].shift(1).fillna(False).astype(bool)
        parts.append(p[p["datetime"] == p["datetime"].max()])
    T = pd.concat(parts, ignore_index=True)
    day = T["datetime"].max()
    T = T[T["datetime"] == day]
    M = pd.read_parquet(os.path.join(DATA, "senti_market.parquet")).set_index("datetime")
    b = pd.read_parquet(os.path.join(DATA, "bench.parquet")).set_index("datetime")["bench"]
    lvl = (1 + b).cumprod()
    ma20 = float(lvl.iloc[-1] / lvl.rolling(20).mean().iloc[-1] - 1)
    mdn = float(M["M_DN"].iloc[-1]) if M.index[-1] == day else float("nan")
    c = T[T["up_lim"] & ~T["prev_up"] & ~T["up_open"] & T["in_pool"] & (T["lim"] < 0.15) & (T["pos60"] < 1)].copy()
    c["明日买入区间"] = [f"{x * 0.93:.2f} ~ {x * 0.98:.2f}" for x in c["raw_close"]]
    print(f"数据日期 {day.date()}：中证1000 相对 20 日线 {ma20:+.2%}（需 < 0），全市场跌停占比 {mdn:.2%}（需 < 3%）")
    if not (ma20 < 0 and mdn < 0.03):
        print("市场条件不满足 → 明天不做")
    print(f"候选（今日首板、60 日低位、10cm、非开盘即涨停）：{len(c)} 只；明早 9:25 开盘价落在区间内才买，最多 2 只，低开更深的优先")
    if len(c):
        print(c[["instrument", "raw_close", "pos60", "明日买入区间"]].rename(columns={"raw_close": "今日收盘", "pos60": "位置"})
              .round(3).to_string(index=False))


if __name__ == "__main__":
    {"build": build, "study": study, "port": port, "watch": watch}[sys.argv[1]]()
