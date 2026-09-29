"""
下载龙虎榜历史明细（东方财富数据中心，经 AKShare），按月分段拉取
  python data_fetch/fetch_lhb.py --start 2015-01-01
输出：{out}/lhb.parquet  列：date, instrument, net_buy, buy, sell, lhb_amt, total_amt, reason
"""
import argparse
import os
import time

import akshare as ak
import pandas as pd


def to_qlib(code):
    code = str(code).zfill(6)
    if code.startswith(("6", "9")):
        return "SH" + code
    if code.startswith(("4", "8")):
        return "BJ" + code
    return "SZ" + code


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--end", default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    ap.add_argument("--out", default=os.path.expanduser("~/.qlib/retail_extra"))
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "lhb_parts"), exist_ok=True)
    months = pd.date_range(a.start, a.end, freq="MS")
    for m in months:
        s, e = m, min(m + pd.offsets.MonthEnd(0), pd.Timestamp(a.end))
        fp = os.path.join(a.out, "lhb_parts", f"{s:%Y%m}.parquet")
        if os.path.exists(fp) and s < pd.Timestamp(a.end) - pd.Timedelta(days=40):
            continue
        for k in range(4):
            try:
                df = ak.stock_lhb_detail_em(start_date=f"{s:%Y%m%d}", end_date=f"{e:%Y%m%d}")
                break
            except Exception as ex:  # noqa
                print("retry", s.date(), repr(ex)[:80], flush=True)
                time.sleep(3 * (k + 1))
        else:
            continue
        if df is None or len(df) == 0:
            continue
        df = df.rename(columns={"上榜日": "date", "代码": "code", "龙虎榜净买额": "net_buy", "龙虎榜买入额": "buy",
                                "龙虎榜卖出额": "sell", "龙虎榜成交额": "lhb_amt", "市场总成交额": "total_amt",
                                "上榜原因": "reason"})
        keep = [c for c in ["date", "code", "net_buy", "buy", "sell", "lhb_amt", "total_amt", "reason"] if c in df]
        df = df[keep].copy()
        df["date"] = pd.to_datetime(df["date"])
        df["instrument"] = df["code"].map(to_qlib)
        for c in ["net_buy", "buy", "sell", "lhb_amt", "total_amt"]:
            if c in df:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df.to_parquet(fp, index=False)
        print(s.strftime("%Y-%m"), len(df), flush=True)
        time.sleep(0.5)
    parts = [pd.read_parquet(os.path.join(a.out, "lhb_parts", f)) for f in sorted(os.listdir(os.path.join(a.out, "lhb_parts")))]
    allp = pd.concat(parts, ignore_index=True)
    allp.to_parquet(os.path.join(a.out, "lhb.parquet"), index=False)
    print("done", allp.shape, allp["date"].min(), allp["date"].max())


if __name__ == "__main__":
    main()
