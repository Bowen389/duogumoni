"""Validate the archive against an independent exchange calendar before no-op."""
import os
from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import pandas as pd
from data_fetch.update_daily import rows


def validate(actual, dates, today, completed_through=None):
    if dates.empty or today not in set(dates.calendar_date):
        raise RuntimeError('Trading calendar does not cover today')
    cutoff = completed_through or today
    trading = dates.loc[dates.is_trading_day.eq('1') & dates.calendar_date.le(cutoff), 'calendar_date']
    if trading.empty:
        raise RuntimeError('Independent trading calendar unavailable')
    expected = trading.iloc[-1]
    if actual != expected:
        raise RuntimeError(f'DATA_PENDING expected={expected} actual={actual}; accounts unchanged; retry after archive publication')
    return expected


def main():
    import baostock as bs
    now = datetime.now(ZoneInfo('Asia/Shanghai'))
    today = now.date()
    # Before the post-close publication window, only previous sessions are due.
    cutoff = today if now.hour >= 16 else today - timedelta(days=1)
    actual = (Path(os.environ['RA_PROVIDER'])/'calendars/day.txt').read_text().splitlines()[-1]
    login = bs.login()
    if login.error_code != '0':
        raise RuntimeError(login.error_msg)
    try:
        dates = rows(bs.query_trade_dates(start_date=(today-timedelta(days=40)).isoformat(), end_date=today.isoformat()))
        expected = validate(actual, dates, today.isoformat(), cutoff.isoformat())
        print(f'DATA_FRESH expected={expected} actual={actual}',flush=True)
    finally:
        bs.logout()


if __name__ == '__main__':
    main()
