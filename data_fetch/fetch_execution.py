"""Cache FB candidates' first 5-minute raw closing quotes (observed at 09:35).

This is a post-auction execution proxy, not a tick fill. No minute quote means
no assumed fill. --start/--end can backfill after historical ST data is loaded.
"""
import argparse
import json
import os
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine.common import DATA, EXTRA


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start")
    ap.add_argument("--end")
    a = ap.parse_args()
    dates = list(pd.read_parquet(os.path.join(DATA, "bench.parquet"))["datetime"])
    fp = Path("paper/accounts/FB_dip/state.json")
    state = json.loads(fp.read_text()) if fp.exists() else {}
    start = pd.Timestamp(a.start or state.get("last_date") or dates[-1])
    end = pd.Timestamp(a.end or dates[-1])
    from strategies2.fb_live import candidates, market_state
    import baostock as bs
    if bs.login().error_code != "0":
        raise RuntimeError("Minute quote login failed")
    rows = []
    try:
        for i, d in enumerate(dates):
            if not start < d <= end or i == 0:
                continue
            signal = dates[i - 1]
            ma, mdn = market_state(signal)
            if not (ma < 0 and mdn < .03):
                continue
            cands, _ = candidates(signal)
            for inst in cands.index:
                rs = bs.query_history_k_data_plus(inst[:2].lower() + "." + inst[2:], "date,time,close,volume",
                                                  start_date=str(d.date()), end_date=str(d.date()),
                                                  frequency="5", adjustflag="3")
                while rs.error_code == "0" and rs.next():
                    date, time, close, volume = rs.get_row_data()
                    ts = pd.to_datetime(time[:14], format="%Y%m%d%H%M%S", errors="coerce")
                    if ts == pd.Timestamp(d).normalize() + pd.Timedelta(hours=9, minutes=35):
                        rows.append(dict(datetime=pd.Timestamp(date), instrument=inst, observed_at=ts,
                                         price=float(close), volume=float(volume), source="baostock_5m_close"))
                        break
                if rs.error_code != "0":
                    print(f"Minute quote unavailable {d.date()} {inst}: {rs.error_msg}; no fill")
    finally:
        bs.logout()
    fp = os.path.join(EXTRA, "execution_0935.parquet")
    old = pd.read_parquet(fp) if os.path.exists(fp) else pd.DataFrame()
    fresh = pd.DataFrame(rows)
    if not fresh.empty:
        out = pd.concat([old, fresh], ignore_index=True).drop_duplicates(["datetime", "instrument"], keep="last")
        out.to_parquet(fp + ".tmp", index=False)
        os.replace(fp + ".tmp", fp)
    print(f"09:35 quotes: {len(rows)}; missing quotes are not replaced by auction prices")


if __name__ == "__main__":
    main()
