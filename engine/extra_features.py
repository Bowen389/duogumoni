"""
第二优先：公开日线里没有的"散户数据"因子
  python -m engine.extra_features
依赖：{EXTRA}/turn/*.parquet（baostock 换手率/ST）、{EXTRA}/lhb.parquet（龙虎榜，可缺省）
输出 {DATA}/feat_extra/part-*.parquet（与面板行顺序一致，可直接拼接）

因子列表（未统一方向，单因子检验会给出 IC 符号）：
  换手率类  TURN20 换手率均值 | ABTURN 5日/120日换手比 | TURNVOL 换手波动 | LOGMV 流通市值(对数) | TURN_MAX 20日最大换手
  涨停细节  LIMUP20 涨停次数 | ZHABAN20 炸板次数 | ZHABAN_RATE 炸板率 | LIANBAN 当前连板高度 | YIZI20 一字板次数 | LIMDN20 跌停次数
  龙虎榜    LHB20 20日上榜次数 | LHB_NET20 上榜净买入/成交额 | LHB_GAP 距上次上榜天数(上限120)
  状态      IS_ST
"""
import os
import shutil

import numpy as np
import pandas as pd

from engine.common import DATA, EXTRA


def _roll(g, s, fn, w, minp=None):
    return s.groupby(g, sort=False).transform(lambda x: getattr(x.rolling(w, min_periods=minp or max(2, w // 2)), fn)())


def build():
    pdir = os.path.join(DATA, "panel")
    out = os.path.join(DATA, "feat_extra")
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)
    tdir = os.path.join(EXTRA, "turn")
    lhb_fp = os.path.join(EXTRA, "lhb.parquet")
    lhb = pd.read_parquet(lhb_fp) if os.path.exists(lhb_fp) else None
    if lhb is not None:
        lhb = lhb.groupby(["instrument", "date"], as_index=False).agg(net_buy=("net_buy", "sum")) \
                 .rename(columns={"date": "datetime"})
        lhb_start = lhb["datetime"].min()
        print("龙虎榜：", len(lhb), "条，起始", lhb_start.date())
    for fn in sorted(os.listdir(pdir)):
        p = pd.read_parquet(os.path.join(pdir, fn))
        codes = p["instrument"].unique()
        # ---- 换手率 / ST ----
        tl = []
        for c in codes:
            f = os.path.join(tdir, f"{c}.parquet")
            if os.path.exists(f):
                t = pd.read_parquet(f).rename(columns={"date": "datetime"})
                t["instrument"] = c
                tl.append(t[["instrument", "datetime", "turn", "isST"]])
        if tl:
            p = p.merge(pd.concat(tl), on=["instrument", "datetime"], how="left")
        else:
            p["turn"], p["isST"] = np.nan, 0
        g = p["instrument"].values
        turn = p["turn"].where(p["turn"] > 0)
        f = pd.DataFrame({"datetime": p["datetime"], "instrument": p["instrument"]})
        f["TURN20"] = _roll(g, turn, "mean", 20)
        f["ABTURN"] = _roll(g, turn, "mean", 5) / (_roll(g, turn, "mean", 120, 60) + 1e-6)
        f["TURNVOL"] = _roll(g, turn, "std", 20) / (f["TURN20"] + 1e-6)
        f["TURN_MAX"] = _roll(g, turn, "max", 20)
        f["LOGMV"] = np.log(_roll(g, p["amount"] / (turn / 100), "median", 20) + 1)   # 流通市值 = 成交额/换手率
        f["IS_ST"] = p["isST"].fillna(0).astype("float32")
        # ---- 涨停细节（用未复权价 + 板块涨跌停幅度）----
        raw_high = p["high"] / p["factor"]
        raw_open = p["open"] / p["factor"]
        raw_low = p["low"] / p["factor"]
        prev_raw = p["raw_close"] / (1 + p["change"])
        up_p = np.round(prev_raw * (1 + p["lim"]) + 1e-6, 2)
        touched = (raw_high >= up_p - 0.0051) & ~p["susp"]
        up = p["up_lim"].astype(float)
        zb = (touched & ~p["up_lim"]).astype(float)
        yizi = (p["up_lim"] & (raw_low >= up_p - 0.0051) & (raw_open >= up_p - 0.0051)).astype(float)
        f["LIMUP20"] = _roll(g, up, "sum", 20, 1)
        f["ZHABAN20"] = _roll(g, zb, "sum", 20, 1)
        f["ZHABAN_RATE"] = f["ZHABAN20"] / (f["ZHABAN20"] + f["LIMUP20"] + 1)
        f["YIZI20"] = _roll(g, yizi, "sum", 20, 1)
        f["LIMDN20"] = _roll(g, p["dn_lim"].astype(float), "sum", 20, 1)
        # 连板高度：连续涨停天数
        brk = (up == 0).astype(int).groupby(g, sort=False).cumsum()
        f["LIANBAN"] = up.groupby([g, brk.values], sort=False).cumsum().values
        # ---- 龙虎榜 ----
        if lhb is not None:
            q = p[["instrument", "datetime", "amount"]].merge(lhb, on=["instrument", "datetime"], how="left")
            on = q["net_buy"].notna().astype(float)
            f["LHB20"] = _roll(g, on, "sum", 20, 1)
            f["LHB_NET20"] = _roll(g, q["net_buy"].fillna(0), "sum", 20, 1) / (_roll(g, q["amount"], "sum", 20, 1) + 1)
            last = q["datetime"].where(on > 0).groupby(g, sort=False).ffill()
            f["LHB_GAP"] = ((q["datetime"] - last).dt.days).clip(upper=120).fillna(120)
            before = (p["datetime"] < lhb_start).values   # 没有数据的年份置空，避免把"没数据"当成"没上榜"
            f.loc[before, ["LHB20", "LHB_NET20", "LHB_GAP"]] = np.nan
        keep = p["in_pool"].values & (p["datetime"] >= pd.Timestamp("2015-01-01")).values
        f = f[keep].reset_index(drop=True)
        cols = [c for c in f.columns if c not in ("datetime", "instrument")]
        f[cols] = f[cols].astype("float32").replace([np.inf, -np.inf], np.nan)
        f.to_parquet(os.path.join(out, fn), index=False)
        print(fn, len(f), "turn覆盖率", round(f["TURN20"].notna().mean(), 3), flush=True)


if __name__ == "__main__":
    build()
