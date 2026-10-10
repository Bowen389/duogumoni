import copy
import json
import pandas as pd
import pytest

from engine.execution import (Ledger, affordable, advance_r3, dropout_orders, signal_score,
                              execute_dropout, execute_conc, execute_fb, sell_fee)
from engine.market import PanelMarket
from engine.backtest import Backtester
from engine.security import status_for, limit_flags
from strategies.config import STRATEGIES

D = pd.Timestamp("2026-10-09")


def quote(inst, d=D, px=10.0, **kw):
    q = dict(datetime=d, instrument=inst, raw_open=px, raw_close=px, factor=1., change=0.,
             susp=False, up_open=False, dn_open=False, up_lim=False, dn_lim=False,
             buy_ok=True, status_known=True, is_st=False, is_delisted=False, in_pool=True)
    q.update(kw)
    return q


def market(*rows, execution=None):
    return PanelMarket(pd.DataFrame(rows), execution=execution)


def holding(px=10., shares=100, date="2026-10-08"):
    return dict(shares=shares, cost=shares * px, buy_date=date, factor=1., last_px=px)


def cfg(k=3, **kw):
    c = dict(topk=k, k=k, risk_degree=.95, min_cost=5., open_cost=.0005, close_cost=.001, slip=0., exec_at="open")
    c.update(kw)
    return c


def test_failed_sale_cannot_create_fourth_position():
    acc = Ledger(10000)
    acc.s["positions"] = {j: holding() for j in ["SZ000001", "SZ000002", "SZ000003"]}
    mk = market(quote("SZ000001", dn_open=True), quote("SZ000002"), quote("SZ000003"), quote("SZ000004"))
    pend = dict(sell=["SZ000001"], buy=[dict(inst="SZ000004")], backup=[], slot=20000)
    execute_conc(acc, mk, D, pend, {}, cfg())
    assert len(acc.s["positions"]) == 3
    assert "SZ000004" not in acc.s["positions"]


def test_dropout_failed_sale_cash_does_not_exceed_topk():
    acc = Ledger(10000)
    acc.s["positions"] = {"SZ000001": holding()}
    mk = market(quote("SZ000001", dn_lim=True), quote("SZ000002"))
    execute_dropout(acc, mk, D, dict(sell=["SZ000001"], buy=[dict(inst="SZ000002")], backup=[]), {}, cfg(1, exec_at="close"))
    assert list(acc.s["positions"]) == ["SZ000001"]


@pytest.mark.parametrize("field,value", [("buy_ok", False), ("up_open", True), ("susp", True), ("raw_open", float("nan"))])
def test_untradable_primary_uses_same_backup(field, value):
    acc = Ledger(10000)
    mk = market(quote("SZ000001", **{field: value}), quote("SZ000002"))
    execute_conc(acc, mk, D, dict(sell=[], buy=[dict(inst="SZ000001")], backup=[dict(inst="SZ000002")]), {}, cfg(1))
    assert list(acc.s["positions"]) == ["SZ000002"]
    assert acc.s["cash"] >= 0


def test_minimum_commission_applies_before_stamp_and_cash_never_negative():
    assert sell_fee(1000, cfg(), D) == 5.5
    assert sell_fee(1000, cfg(), pd.Timestamp("2020-01-02")) == 6.
    assert affordable("SZ000001", 1004., 10., .0005, 5.) == 0
    assert affordable("SZ000001", 1005., 10., .0005, 5.) == 100
    # STAR: minimum 200 shares, then increments of one, not batches of 200.
    assert affordable("SH688001", 2105., 10., .0005, 5.) == 210


def test_t_plus_one_rejects_same_day_sale():
    acc = Ledger(2000)
    acc.trade(D, "买入", "SZ000001", 100, 10., 5.)
    with pytest.raises(ValueError, match="T\\+1"):
        acc.trade(D, "卖出", "SZ000001", 100, 10., 5.)


def test_post_auction_quote_missing_is_not_filled_at_auction():
    acc = Ledger(100000)
    pend = dict(K=2, slot=50000, cands=[dict(inst="SZ000001", ref=10.)])
    execute_fb(acc, market(quote("SZ000001", px=9.5)), D, pend, {})
    assert not acc.s["positions"]
    assert "价格缺失" in acc.trades[-1]["备注"]


@pytest.mark.parametrize("price,volume,time,filled", [(9.51, 1000, "09:35", True), (9.56, 1000, "09:35", False),
                                                       (9.51, 0, "09:35", False), (9.51, 1000, "09:30", False)])
def test_fb_price_time_volume_and_limit(price, volume, time, filled):
    acc = Ledger(100000)
    x = pd.DataFrame([dict(datetime=D, instrument="SZ000001", price=price, volume=volume,
                           observed_at=pd.Timestamp(str(D.date()) + " " + time))])
    mk = market(quote("SZ000001", px=9.5), execution=x)
    execute_fb(acc, mk, D, dict(K=2, cands=[dict(inst="SZ000001", ref=10.)]), {})
    assert bool(acc.s["positions"]) == filled
    if filled:
        assert acc.trades[0]["成交价"] == pytest.approx(9.5195)


def test_fb_gap_is_not_changed_by_display_rounding():
    acc = Ledger(100000)
    # 9.80/9.99 is -1.9019%, outside range even if rounded high bound is 9.79/9.80.
    execute_fb(acc, market(quote("SZ000001", px=9.8)), D,
               dict(K=2, cands=[dict(inst="SZ000001", ref=9.99, lo=9.29, hi=9.8)]), {})
    assert not acc.trades


def test_fb_sells_only_previous_day_and_defers_limit_down():
    acc = Ledger(0)
    acc.s["positions"] = {"SZ000001": holding(), "SZ000002": holding(date=str(D.date()))}
    mk = market(quote("SZ000001", dn_lim=True), quote("SZ000002"))
    execute_fb(acc, mk, D, None, {})
    assert len(acc.s["positions"]) == 2
    mk2 = market(quote("SZ000001", d=D + pd.Timedelta(days=3)), quote("SZ000002", d=D + pd.Timedelta(days=3)))
    execute_fb(acc, mk2, D + pd.Timedelta(days=3), None, {})
    assert not acc.s["positions"]


def test_open_valuation_cannot_use_future_close():
    acc = Ledger(1000)
    acc.s["positions"] = {"SZ000001": holding()}
    mk = market(quote("SZ000001", raw_open=10., raw_close=100.))
    acc.mark(mk, D, at="open")
    assert acc.total() == 2000
    acc.mark(mk, D)
    assert acc.total() == 11000


def test_corporate_action_applied_once_before_execution():
    acc = Ledger(0)
    acc.s["positions"] = {"SZ000001": holding()}
    mk = market(quote("SZ000001", px=5., factor=2.))
    acc.mark(mk, D, at="open")
    acc.mark(mk, D)
    assert acc.s["positions"]["SZ000001"]["shares"] == 200
    assert acc.total() == 1000


def test_r3_retains_peak_and_is_idempotent_across_restart():
    c = STRATEGIES["A_top50"]
    st = {}
    advance_r3(st, .10, 0., D, c)
    advance_r3(st, -.10, 0., D + pd.Timedelta(days=1), c)
    assert st["exposure"] == .5
    restarted = json.loads(json.dumps(st))
    advance_r3(restarted, 9., 0., D + pd.Timedelta(days=1), c)
    assert restarted == st
    advance_r3(restarted, .07, 0., D + pd.Timedelta(days=2), c)
    assert restarted["exposure"] == 1.
    assert restarted["peak"] == pytest.approx(1.1)


def test_r3_funds_etf_from_cash_before_selling_stocks():
    acc = Ledger(9000)
    acc.s["positions"] = {"SZ000001": holding()}
    execute_dropout(acc, market(quote("SZ000001")), D, dict(sell=[], buy=[], backup=[], expo_next=.5), {}, cfg(1, exec_at="close"))
    assert acc.s["positions"]["SZ000001"]["shares"] == 100
    assert acc.s["etf_value"] == 5000
    assert acc.s["cash"] == 3995


def test_historical_st_limit_is_five_percent():
    frame = pd.DataFrame([quote("SZ000001", raw_close=9.5, raw_open=9.5, change=-.05, is_st=True)])
    f = limit_flags(frame.instrument, frame.datetime, frame.raw_close, frame.raw_open, frame.change,
                    frame.susp, frame.is_st, frame.status_known)
    assert f["lim"][0] == pytest.approx(.05)
    assert bool(f["dn_lim"].iloc[0])


def test_status_missing_cannot_buy_and_current_name_not_backdated(tmp_path, monkeypatch):
    import engine.security as sec
    monkeypatch.setattr(sec, "EXTRA", str(tmp_path))
    frame = pd.DataFrame([quote("SZ000001", d=D - pd.Timedelta(days=1)), quote("SZ000001")])
    pd.DataFrame([dict(instrument="SZ000001", code_name="某某退", ipoDate="2000-01-01", outDate="", observed=D)]).to_parquet(tmp_path / "security_basic.parquet")
    st = status_for(frame)
    assert not st.buy_ok.any()
    assert list(st.is_delisted) == [False, True]
    pd.DataFrame([dict(datetime=D, instrument="SZ000001", isST=1, tradestatus=1)]).to_parquet(tmp_path / "security_status.parquet")
    st = status_for(frame)
    assert list(st.status_known) == [False, True]
    assert st.is_st.iloc[1]
    assert not st.buy_ok.any()


def test_backtest_and_daily_account_replay_match_with_r3_and_backups():
    dates = pd.bdate_range("2026-09-28", periods=9)
    rows = []
    scores = []
    for i, d in enumerate(dates):
        for j, base in [("SZ000001", 10), ("SZ000002", 11), ("SZ000003", 12)]:
            px = base * (1 if i < 3 else .80 if i < 6 else .90)
            rows.append(quote(j, d=d, px=px, up_lim=(i == 1 and j == "SZ000001")))
            scores.append(dict(datetime=d, instrument=j, score=3 - int(j[-1]) + (2 if i > 4 and j == "SZ000003" else 0)))
    panel, pred = pd.DataFrame(rows), pd.DataFrame(scores)
    bench = pd.Series(0., index=dates)
    c = dict(STRATEGIES["A_top20"], topk=2, n_drop=1, account=100000)
    bt = Backtester(dates[1], dates[-1], panel=panel, bench=bench)
    result = bt.run(pred, topk=2, n_drop=1, account=100000, r3_cfg=c)
    mk = PanelMarket(panel, bench)
    actual, ref = Ledger(100000), Ledger(100000)
    st, values = {}, []
    for i in range(1, len(dates)):
        d, signal = dates[i], dates[i - 1]
        expo = st.get("exposure", 1.)
        prev_ref = ref.total()
        actual.mark(mk, d)
        ref.mark(mk, d)
        day = mk.day(signal)
        sc = signal_score(pred[pred.datetime == signal].set_index("instrument").score, day)
        for acc, e in [(actual, expo), (ref, 1.)]:
            sell, buy, backup, _ = dropout_orders(sc, list(acc.s["positions"]), 2, 1, day.buy_ok)
            pending = dict(sell=sell, buy=[dict(inst=j) for j in buy], backup=[dict(inst=j) for j in backup], expo_next=e)
            execute_dropout(acc, mk, d, pending, {}, c)
            acc.mark(mk, d)
        advance_r3(st, ref.total() / prev_ref - 1, 0., d, c)
        values.append(actual.total())
    assert list(result.value) == pytest.approx(values)
    assert bt.last_r3["nav"] == pytest.approx(st["nav"])
    assert bt.last_account["positions"] == actual.s["positions"]
    assert result.exposure.min() == .5


def test_historical_backtest_refuses_unknown_status():
    rows = [quote("SZ000001", status_known=False)]
    with pytest.raises(ValueError, match="Historical ST status missing"):
        Backtester(D, D, panel=pd.DataFrame(rows), bench=pd.Series(0., index=[D]))
