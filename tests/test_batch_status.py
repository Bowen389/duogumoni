import pandas as pd
import pytest
from data_fetch.fetch_status import batch_status
from data_fetch.check_freshness import validate


def test_dated_names_and_suspension_missing_data():
    frame=pd.DataFrame(dict(code=['sz.000001','sz.000002','sz.000003','sz.000004'],
        code_name=['平安银行','*ST测试','','普通股票'],tradeStatus=['1','0','1','']))
    out=batch_status(frame,'2026-10-13')
    assert out.instrument.tolist()==['SZ000001','SZ000002','SZ000003','SZ000004']
    assert out.isST.iloc[:2].tolist()==[0,1]
    assert pd.isna(out.isST.iloc[2])
    assert pd.isna(out.tradestatus.iloc[3])
    assert out.datetime.eq(pd.Timestamp('2026-10-13')).all()


def test_stale_archive_cannot_pass_noop_gate():
    dates=pd.DataFrame(dict(calendar_date=['2026-10-12','2026-10-13'],is_trading_day=['1','1']))
    with pytest.raises(RuntimeError,match='DATA_PENDING'):
        validate('2026-10-12',dates,'2026-10-13')
    assert validate('2026-10-13',dates,'2026-10-13')=='2026-10-13'


def test_weekend_requires_previous_trading_day():
    dates=pd.DataFrame(dict(calendar_date=['2026-10-09','2026-10-10'],is_trading_day=['1','0']))
    assert validate('2026-10-09',dates,'2026-10-10')=='2026-10-09'
