import pandas as pd
import pytest
from data_fetch.check_freshness import validate

def calendar():
    return pd.DataFrame(dict(calendar_date=['2026-10-23','2026-10-24','2026-10-25','2026-10-26','2026-10-27'],
                             is_trading_day=['1','0','0','1','1']))

def test_morning_retry_accepts_last_completed_day():
    assert validate('2026-10-26',calendar(),'2026-10-27','2026-10-26')=='2026-10-26'

def test_postclose_still_rejects_stale_archive():
    with pytest.raises(RuntimeError,match='DATA_PENDING'):
        validate('2026-10-26',calendar(),'2026-10-27','2026-10-27')

def test_monday_morning_uses_friday():
    assert validate('2026-10-23',calendar(),'2026-10-26','2026-10-25')=='2026-10-23'

def test_calendar_must_cover_current_day():
    with pytest.raises(RuntimeError,match='cover today'):
        validate('2026-10-26',calendar(),'2026-10-28','2026-10-27')
