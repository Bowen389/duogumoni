"""
下载换手率(turn)、ST 标记、交易状态 —— 来自 baostock（免费、无需 token）
  python data_fetch/fetch_baostock.py --pool csi1000 --workers 4
输出：{out}/turn/{code}.parquet  （断点续传：已下载的跳过）
"""
import argparse
import os
import sys
from multiprocessing import Process

import pandas as pd


def qlib_to_bs(code):  # SH600000 -> sh.600000
    return code[:2].lower() + "." + code[2:]


def worker(codes, out, start, end, wid):
    import baostock as bs
    bs.login()
    for i, c in enumerate(codes):
        fp = os.path.join(out, f"{c}.parquet")
        if os.path.exists(fp):
            continue
        for _ in range(3):
            try:
                rs = bs.query_history_k_data_plus(qlib_to_bs(c), "date,turn,isST,tradestatus",
                                                  start_date=start, end_date=end, frequency="d", adjustflag="3")
                rows = []
                while rs.error_code == "0" and rs.next():
                    rows.append(rs.get_row_data())
                if rs.error_code != "0":
                    raise RuntimeError(f"baostock error {rs.error_code} {rs.error_msg}")
                df = pd.DataFrame(rows, columns=["date", "turn", "isST", "tradestatus"])
                df["turn"] = pd.to_numeric(df["turn"], errors="coerce").astype("float32")
                df["isST"] = pd.to_numeric(df["isST"], errors="coerce").fillna(0).astype("int8")
                df["tradestatus"] = pd.to_numeric(df["tradestatus"], errors="coerce").fillna(0).astype("int8")
                df["date"] = pd.to_datetime(df["date"])
                df.to_parquet(fp, index=False)
                break
            except Exception as e:  # noqa
                print(wid, c, "retry", e, flush=True)
                try:
                    bs.logout(); bs.login()
                except Exception:  # noqa
                    pass
        if i % 100 == 0:
            print(f"[w{wid}] {i}/{len(codes)}", flush=True)
    bs.logout()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default=os.path.expanduser("~/.qlib/qlib_data/cn_data"))
    ap.add_argument("--pool", default="csi1000")
    ap.add_argument("--start", default="2014-06-01")
    ap.add_argument("--end", default="2026-12-31")
    ap.add_argument("--out", default=os.path.expanduser("~/.qlib/retail_extra/turn"))
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    inst = pd.read_csv(os.path.join(a.provider, "instruments", f"{a.pool}.txt"), sep="\t", header=None)
    codes = sorted(inst[inst[2] >= "2015-01-01"][0].unique())
    print("股票数:", len(codes), flush=True)
    ps = [Process(target=worker, args=(codes[i::a.workers], a.out, a.start, a.end, i)) for i in range(a.workers)]
    [p.start() for p in ps]
    [p.join() for p in ps]
    print("done", len(os.listdir(a.out)), flush=True)


if __name__ == "__main__":
    sys.exit(main())
