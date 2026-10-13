import numpy as np
import pandas as pd
import pytest
from data_fetch.update_daily import normalized, append_bin, read_bin, rows


def bars():
    return pd.DataFrame([
        dict(date='2026-10-12',open=10,high=10,low=10,close=10,volume=1000,amount=10000,pctChg=0),
        dict(date='2026-10-13',open=9,high=9.2,low=9,close=9.09,volume=2000,amount=18180,pctChg=1),
    ])


def test_adjustment_splice_preserves_price_and_volume_units():
    old = dict(close=20,factor=2,volume_scale=.01,amount_scale=.001,vwap_scale=10)
    out = normalized(bars(),'2026-10-12',old).iloc[0]
    assert out.close == pytest.approx(20.2)
    assert out.close/out.factor == pytest.approx(9.09)
    assert out.volume*out.factor == pytest.approx(20)
    assert out.amount == pytest.approx(18.18)
    assert out.change == pytest.approx(.01)


def test_wrong_overlap_cannot_publish_adjusted_prices():
    with pytest.raises(ValueError,match='mismatch'):
        normalized(bars(),'2026-10-12',dict(close=30,factor=2))


def test_append_preserves_old_history_and_is_idempotent(tmp_path):
    np.array([2,10,11],dtype='<f4').tofile(tmp_path/'close.day.bin')
    cal = {'2026-10-12':3,'2026-10-13':4}
    values = pd.Series([12.],index=['2026-10-13'])
    append_bin(tmp_path,'close',values,cal)
    append_bin(tmp_path,'close',values,cal)
    assert np.fromfile(tmp_path/'close.day.bin',dtype='<f4').tolist() == [2,10,11,12]
    assert read_bin(tmp_path,'close',3) == 11


def test_session_failure_does_not_look_like_empty_data():
    class Result:
        error_code = '10001001'
        error_msg = '用户未登录'
    with pytest.raises(RuntimeError,match='未登录'):
        rows(Result())


@pytest.mark.parametrize('missing',[False,True])
def test_refresh_requires_all_current_bars_before_calendar_publish(tmp_path,monkeypatch,missing):
    import sys
    import types
    from data_fetch import update_daily as u
    provider=tmp_path/'provider'; extra=tmp_path/'extra'
    (provider/'calendars').mkdir(parents=True)
    (provider/'instruments').mkdir()
    (provider/'calendars/day.txt').write_text('2026-10-12\n')
    (provider/'instruments/csi1000.txt').write_text('SZ000001\t2020-01-01\t2026-10-12\n')
    for inst in ('sz000001','sh000852'):
        folder=provider/'features'/inst;folder.mkdir(parents=True)
        for field,value in dict(close=20,factor=2,volume=500,amount=10000,vwap=20).items():
            np.array([0,value],dtype='<f4').tofile(folder/f'{field}.day.bin')
    class Result:
        error_code='0';error_msg=''
        def __init__(self,frame):
            self.fields=list(frame.columns);self.data=iter(frame.astype(str).values.tolist())
        def next(self):
            self.row=next(self.data,None);return self.row is not None
        def get_row_data(self):return self.row
    def query(code,fields,**kw):
        frame=bars();frame['isST']=0;frame['tradestatus']=1
        if missing and code=='sz.000001':frame=frame.iloc[:1]
        return Result(frame[fields.split(',')])
    fake=types.SimpleNamespace(login=lambda:types.SimpleNamespace(error_code='0'),logout=lambda:None,
        query_trade_dates=lambda **kw:Result(pd.DataFrame(dict(calendar_date=['2026-10-12','2026-10-13'],is_trading_day=['1','1']))),
        query_history_k_data_plus=query)
    monkeypatch.setitem(sys.modules,'baostock',fake)
    monkeypatch.setenv('RA_PROVIDER',str(provider));monkeypatch.setenv('RA_EXTRA',str(extra))
    monkeypatch.chdir(tmp_path)
    class Clock:
        @staticmethod
        def now(tz):return pd.Timestamp('2026-10-13',tz=tz)
    monkeypatch.setattr(u,'datetime',Clock)
    if missing:
        with pytest.raises(RuntimeError,match='incomplete'):u.main()
        assert (provider/'calendars/day.txt').read_text()=='2026-10-12\n'
        assert not (extra/'security_status.parquet').exists()
    else:
        u.main()
        assert (provider/'calendars/day.txt').read_text().splitlines()[-1]=='2026-10-13'
        assert read_bin(provider/'features/sz000001','close',1)==pytest.approx(20.2)
        assert len(pd.read_parquet(extra/'security_status.parquet'))==2
