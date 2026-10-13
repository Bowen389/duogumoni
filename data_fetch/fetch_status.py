"""Incremental, dated ST/trading-status cache used by all execution paths.

Daily: fetch missing dates in market-wide batches; explicit backfill: python data_fetch/fetch_status.py --start 2014-06-01 --end YYYY-MM-DD.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine.common import EXTRA, PROVIDER, pool_spans


def batch_status(frame, day):
    """Dated names identify ST; blank names/status remain unknown, never buyable."""
    frame = frame.copy()
    frame['instrument'] = frame.code.str.replace('.', '',regex=False).str.upper()
    names = frame.code_name.fillna('').str.strip()
    frame['isST'] = names.str.upper().str.contains('ST',regex=False).astype(float)
    frame.loc[names.eq(''),'isST'] = float('nan')
    frame['tradestatus'] = pd.to_numeric(frame.tradeStatus,errors='coerce')
    frame.loc[~frame.tradestatus.isin([0,1]),'tradestatus'] = float('nan')
    frame['datetime'] = pd.Timestamp(day)
    frame['source'] = 'baostock_query_all_stock_dated_name'
    return frame[['datetime','instrument','isST','tradestatus','source']]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    end = pd.Timestamp(a.end or pd.read_csv(os.path.join(PROVIDER, "calendars/day.txt"), header=None).iloc[-1, 0])
    account_dates = []
    held = set()
    for fp in Path("paper/accounts").glob("*/state.json"):
        state = json.loads(fp.read_text())
        if state.get("last_date"):
            account_dates.append(pd.Timestamp(state["last_date"]))
        held.update(state.get("positions", {}))
    # Fetch only dates still needed by accounts, plus the current signal day.
    start = pd.Timestamp(a.start) if a.start else min(account_dates + [end]) + pd.Timedelta(days=1)
    start = min(start,end)
    spans = pool_spans()
    calendar = pd.to_datetime(pd.read_csv(os.path.join(PROVIDER, "calendars/day.txt"),header=None)[0])
    days = calendar[(calendar >= start) & (calendar <= end)]
    fp = os.path.join(EXTRA, "security_status.parquet")
    old = pd.read_parquet(fp) if os.path.exists(fp) else pd.DataFrame()
    missing = []
    requirements = {}
    for day in days:
        codes = set(spans.loc[(spans.start <= day) & (spans.end >= day), "instrument"]) | held
        codes = {j for j in codes if j.startswith(("SH","SZ"))}
        requirements[day] = codes
        known = set()
        if not old.empty:
            valid = old.datetime.eq(day) & old.isST.notna() & old.tradestatus.notna()
            known = set(old.loc[valid,"instrument"])
        if codes - known:
            missing.append(day)
    if not missing and os.path.exists(os.path.join(EXTRA,"security_basic.parquet")):
        print(f'STATUS_READY cached through {end.date()}',flush=True)
        return
    import baostock as bs
    login = bs.login()
    if login.error_code != "0":
        raise RuntimeError(f"Baostock basic login failed: {login.error_msg}")
    rs = bs.query_stock_basic()
    rows = []
    while rs.error_code == "0" and rs.next():
        rows.append(rs.get_row_data())
    bs.logout()
    if rs.error_code != "0" or not rows:
        raise RuntimeError("Stock basic data unavailable; previous cache preserved")
    basic = pd.DataFrame(rows, columns=rs.fields)
    basic["instrument"] = basic["code"].str.replace(".", "", regex=False).str.upper()
    basic["observed"] = pd.Timestamp.now(tz="Asia/Shanghai").normalize().tz_localize(None)
    os.makedirs(EXTRA, exist_ok=True)
    bfp = os.path.join(EXTRA, "security_basic.parquet")
    if os.path.exists(bfp):
        old_basic = pd.read_parquet(bfp).set_index("instrument")
        observed = old_basic.loc[old_basic["code_name"].fillna("").str.contains("退", regex=False), "observed"]
        risk = basic["code_name"].fillna("").str.contains("退", regex=False)
        basic.loc[risk, "observed"] = basic.loc[risk, "instrument"].map(observed).fillna(basic.loc[risk, "observed"])
    basic.to_parquet(bfp + ".tmp", index=False)
    os.replace(bfp + ".tmp", bfp)
    # One dated market-wide response replaces thousands of per-symbol queries.
    login = bs.login()
    if login.error_code != "0":
        raise RuntimeError(f"Baostock status login failed: {login.error_msg}")
    chunks = []
    try:
        for day in missing:
            rs = bs.query_all_stock(day=str(day.date()))
            rows = []
            while rs.error_code == "0" and rs.next():
                rows.append(rs.get_row_data())
            if rs.error_code != "0" or not rows:
                raise RuntimeError(f"Dated market status unavailable {day.date()}: {rs.error_msg}")
            fresh = batch_status(pd.DataFrame(rows,columns=rs.fields),day)
            fresh = fresh[fresh.instrument.isin(requirements[day])]
            known = set(fresh.loc[fresh.isST.notna() & fresh.tradestatus.notna(),'instrument'])
            prior = set()
            if not old.empty:
                valid = old.datetime.eq(day) & old.isST.notna() & old.tradestatus.notna()
                prior = set(old.loc[valid,'instrument'])
            unknown = requirements[day] - (known | prior)
            coverage = 1-len(unknown)/max(1,len(requirements[day]))
            print(f'STATUS_BATCH date={day.date()} known={len(requirements[day])-len(unknown)}/{len(requirements[day])} queries=1',flush=True)
            if coverage < .95:
                raise RuntimeError(f'STATUS_INCOMPLETE {day.date()}: coverage={coverage:.1%}; accounts unchanged')
            # An invalid new row must never erase a known cached status.
            chunks.append(fresh[fresh.isST.notna() & fresh.tradestatus.notna()])
    finally:
        bs.logout()
    if chunks:
        result = pd.concat([old,*chunks],ignore_index=True).drop_duplicates(['datetime','instrument'],keep='last')
        result.to_parquet(fp+'.tmp',index=False)
        os.replace(fp+'.tmp',fp)
    print(f'STATUS_READY through {end.date()} requests={len(missing)}',flush=True)


if __name__ == "__main__":
    main()
