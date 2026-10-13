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
