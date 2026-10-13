"""Direct Baostock daily refresh; Qlib is only the local storage/feature engine.

The community archive supplies history. New bars are normalized against its
last bar, preserving price, volume and amount units rather than splicing raw
prices into adjusted history. Calendar publication is the final commit point.
"""
import json
import os
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd

FIELDS = 'date,open,high,low,close,volume,amount,pctChg,isST,tradestatus'


def rows(rs):
    result = []
    while rs.error_code == '0' and rs.next():
        result.append(rs.get_row_data())
    if rs.error_code != '0':
        raise RuntimeError(f'Baostock {rs.error_code}: {rs.error_msg}')
    return pd.DataFrame(result, columns=rs.fields)


def read_bin(folder, field, index):
    a = np.fromfile(folder / f'{field}.day.bin', dtype='<f4')
    offset = int(a[0])
    pos = index - offset + 1
    return float(a[pos]) if 1 <= pos < len(a) else float('nan')


def normalized(raw, anchor, old):
    """Chain adjusted returns from a verified overlap bar (including ex dates)."""
    raw = raw.copy().set_index('date').sort_index()
    for c in raw:
        raw[c] = pd.to_numeric(raw[c], errors='coerce')
    base = raw.loc[anchor]
    old_raw = old['close'] / old['factor']
    if not np.isfinite(old_raw) or abs(base['close'] / old_raw - 1) > .002:
        raise ValueError('Overlap raw close mismatch; adjustment splice refused')
    # The caller supplies scales from the same liquid overlap date.
    out = raw.loc[raw.index > anchor].copy()
    out['change'] = out['pctChg'] / 100
    if out['change'].isna().any() or (out['close'] <= 0).any():
        raise ValueError('Missing/invalid close or return')
    adj_close = old['close'] * (1 + out['change']).cumprod()
    out['factor'] = adj_close / out['close']
    for c in ('open', 'high', 'low', 'close'):
        out[c] *= out['factor']
    out['volume'] = out['volume'] * old['volume_scale'] / out['factor']
    out['amount'] *= old['amount_scale']
    out['vwap'] = out['amount'] / out['volume'] * old['vwap_scale']
    return out


def append_bin(folder, field, values, calendar):
    fp = folder / f'{field}.day.bin'
    a = np.fromfile(fp, dtype='<f4') if fp.exists() else np.array([calendar[values.index[0]]], dtype='<f4')
    offset = int(a[0])
    end = max(calendar[d] for d in values.index)
    b = np.full(end - offset + 2, np.nan, dtype='<f4')
    b[:min(len(a), len(b))] = a[:len(b)]
    for d, value in values.items():
        b[calendar[d] - offset + 1] = value
    tmp = fp.with_suffix('.tmp')
    b.tofile(tmp)
    os.replace(tmp, fp)


def main():
    import baostock as bs
    root = Path(os.environ['RA_PROVIDER'])
    extra = Path(os.environ['RA_EXTRA'])
    calfp = root / 'calendars/day.txt'
    days = calfp.read_text().splitlines()
    last = days[-1]
    today = datetime.now(ZoneInfo('Asia/Shanghai')).date().isoformat()
    login = bs.login()
    if login.error_code != '0':
        raise RuntimeError(login.error_msg)
    try:
        dates = rows(bs.query_trade_dates(start_date=last, end_date=today))
        if dates.empty or today not in set(dates.calendar_date):
            raise RuntimeError('Trading calendar does not cover today')
        expected = dates.loc[dates.is_trading_day.eq('1'), 'calendar_date'].tolist()
        if not expected:
            raise RuntimeError('Independent trading calendar unavailable')
        target = expected[-1]
        newdays = [d for d in expected if d > last]
        spans = pd.read_csv(root / 'instruments/csi1000.txt', sep='\t', header=None, names=['instrument','start','end'])
        codes = set(spans.loc[spans.end.ge(last), 'instrument'])
        for fp in Path('paper/accounts').glob('*/state.json'):
            codes.update(json.loads(fp.read_text()).get('positions', {}))
        codes.add('SH000852')
        status = []
        updates = []
        failures = []
        calendar = {d:i for i,d in enumerate(days + newdays)}
        start = (pd.Timestamp(last) - pd.Timedelta(days=45)).date().isoformat()
        for i, inst in enumerate(sorted(codes)):
            try:
                folder = root / 'features' / inst.lower()
                raw = rows(bs.query_history_k_data_plus(inst[:2].lower()+'.'+inst[2:], FIELDS if inst != 'SH000852' else FIELDS.replace(',isST,tradestatus',''),
                           start_date=start, end_date=target, frequency='d', adjustflag='3'))
                if raw.empty or not set([last] + newdays).issubset(set(raw.date)):
                    raise ValueError('Missing required daily bars')
                numeric = raw.set_index('date').apply(pd.to_numeric, errors='coerce')
                if inst != 'SH000852':
                    st = raw[['date','isST','tradestatus']].copy().rename(columns={'date':'datetime'})
                    st['datetime'] = pd.to_datetime(st.datetime)
                    st['instrument'] = inst
                    for c in ('isST','tradestatus'):
                        st[c] = pd.to_numeric(st[c], errors='coerce')
                    if st[['isST','tradestatus']].isna().any().any():
                        raise ValueError('Missing trading status')
                    status.append(st)
                if newdays:
                    if inst == 'SH000852':
                        anchor = read_bin(folder,'close',calendar[last])
                        data = numeric.loc[newdays].copy()
                        data['change'] = data['pctChg']/100
                        if not np.isfinite(anchor) or data['change'].isna().any():
                            raise ValueError('Benchmark return unavailable')
                        adjusted = anchor*(1+data['change']).cumprod()
                        scale = adjusted/data['close']
                        for field in ('open','high','low','close'):
                            data[field] *= scale
                        updates.append((folder,data[['open','high','low','close','change']]))
                        continue
                    old = {f:read_bin(folder,f,calendar[last]) for f in ('close','factor')}
                    liquid = numeric.loc[numeric.index <= last].query('volume > 0').tail(1)
                    if liquid.empty:
                        raise ValueError('No liquid overlap to validate units')
                    d = liquid.index[0]; q = liquid.iloc[0]
                    fac = read_bin(folder,'factor',calendar[d])
                    old['volume_scale'] = read_bin(folder,'volume',calendar[d])*fac/q.volume
                    old['amount_scale'] = read_bin(folder,'amount',calendar[d])/q.amount
                    av = read_bin(folder,'vwap',calendar[d])
                    aa = read_bin(folder,'amount',calendar[d]); vv = read_bin(folder,'volume',calendar[d])
                    old['vwap_scale'] = av/(aa/vv) if np.isfinite(av) else 1.
                    if not all(np.isfinite(old[x]) and old[x] > 0 for x in old):
                        raise ValueError('Invalid normalization anchor')
                    updates.append((folder,normalized(raw,last,old).loc[newdays]))
            except Exception as e:
                failures.append(f'{inst}: {e}')
            if i % 100 == 0:
                print(f'DIRECT_DAILY {i}/{len(codes)} target={target}', flush=True)
        if failures:
            raise RuntimeError(f'Daily refresh incomplete ({len(failures)}/{len(codes)}): '+ '; '.join(failures[:15]))
        # Validate all symbols before publishing any daily bars or the calendar.
        for folder, data in updates:
            for field in data.columns.intersection(['open','high','low','close','factor','volume','amount','change','vwap']):
                append_bin(folder,field,data[field],calendar)
        extra.mkdir(parents=True, exist_ok=True)
        fp = extra / 'security_status.parquet'
        fresh = pd.concat(status,ignore_index=True)
        old = pd.read_parquet(fp) if fp.exists() else pd.DataFrame()
        pd.concat([old,fresh],ignore_index=True).drop_duplicates(['datetime','instrument'],keep='last').to_parquet(str(fp)+'.tmp',index=False)
        os.replace(str(fp)+'.tmp',fp)
        # Extend only current constituents; preserve historical memberships.
        spans.loc[spans.end.ge(last),'end'] = target
        spans.to_csv(root / 'instruments/csi1000.txt',sep='\t',header=False,index=False)
        tmp = calfp.with_suffix('.tmp'); tmp.write_text('\n'.join(days+newdays)+'\n'); os.replace(tmp,calfp)
        print(f'DATA_READY source=baostock expected={target} actual={days[-1] if not newdays else newdays[-1]} symbols={len(codes)}',flush=True)
    finally:
        bs.logout()


if __name__ == '__main__':
    main()
