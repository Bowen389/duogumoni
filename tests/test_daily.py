import argparse
import copy
import json
from pathlib import Path
import pandas as pd
import pytest
from tests.test_execution import D, quote, holding, market


def fake_prediction(start, end, names):
    pred = pd.DataFrame([dict(datetime=D, instrument=j, score=3 - i) for i, j in enumerate(["SZ000001", "SZ000002", "SZ000003"])])
    metas = {name: dict(sets=["retail"], train=["2015-01-01", "2020-12-31"], best_iter=1) for name in names}
    return pred, metas


def install_fake_signal_data(monkeypatch):
    from strategies import live as A
    panel = pd.DataFrame([quote(j) for j in ["SZ000001", "SZ000002", "SZ000003"]])
    monkeypatch.setattr(A, "predict", fake_prediction)
    monkeypatch.setattr(A, "read_parts", lambda *args, **kw: panel.copy())
    industry = pd.DataFrame([dict(instrument=j, code_name=j, industry="test") for j in panel.instrument])
    monkeypatch.setattr(pd, "read_parquet", lambda *args, **kw: industry.copy())
    return A


def test_signal_and_shadow_pending_use_the_same_rankings(tmp_path, monkeypatch):
    from paper import sim
    from engine.execution import Ledger
    A = install_fake_signal_data(monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sim, "PAPER", str(tmp_path / "paper"))
    monkeypatch.setattr(sim, "SIG_DIR", str(tmp_path / "paper/signals"))
    acc = sim.Account("A_top20")
    acc.s = dict(cash=100000., etf_value=0., positions={}, r3=dict(exposure=1., dd=0., last_date=str(D.date())),
                 reference=Ledger(100000).s)
    pend, out, _ = sim.gen_signal(acc, D)
    assert [x["inst"] for x in pend["buy"]] == [x["inst"] for x in acc.s["reference_pending"]["buy"]]
    assert pend["expo_next"] == 1.
    assert len(out[out["操作"] == "买入"]) == 3


def test_c_signal_restores_a_model_globals(tmp_path, monkeypatch):
    from strategies2 import live as S2
    A = install_fake_signal_data(monkeypatch)
    monkeypatch.setattr(S2, "predict", lambda s, e, names, neutral=True: fake_prediction(s, e, names))
    originals = A.STRATEGIES, A.MODELS, A.predict
    monkeypatch.chdir(tmp_path)
    ns = argparse.Namespace(strategy="C_top50", date=str(D.date()), holdings=None, cash=100000., etf_value=0.,
                            lookback=365, r3_override=dict(exposure=1., dd=0.))
    out = S2.signal_dropout(ns, S2.STRATEGIES["C_top50"])
    assert "score" in out.attrs
    assert A.STRATEGIES is originals[0]
    assert A.MODELS is originals[1]
    assert A.predict is originals[2]


def test_c_signal_restores_globals_on_failure(monkeypatch):
    from strategies import live as A
    from strategies2 import live as S2
    originals = A.STRATEGIES, A.MODELS, A.predict
    def broken(*args):
        raise RuntimeError("expected")
    monkeypatch.setattr(A, "cmd_signal", broken)
    ns = argparse.Namespace(strategy="C_top50", lookback=365)
    with pytest.raises(RuntimeError, match="expected"):
        S2.signal_dropout(ns, S2.STRATEGIES["C_top50"])
    assert (A.STRATEGIES, A.MODELS, A.predict) == originals


def test_migration_preserves_holdings_cash_and_half_exposure():
    from paper import sim
    acc = sim.Account("A_top20")
    acc.s = dict(cash=1000., etf_value=5000., positions={"SZ000001": holding()},
                 pending=dict(expo_next=.5, dd=-.10))
    before = copy.deepcopy(acc.s)
    sim.migrate_execution(acc, D)
    for k in ("cash", "etf_value", "positions"):
        assert acc.s[k] == before[k]
    assert acc.s["r3"]["exposure"] == .5
    assert acc.s["reference"]["cash"] == 6000
    assert acc.s["execution_start"] == str(D.date())
    snapshot = copy.deepcopy(acc.s)
    sim.migrate_execution(acc, D + pd.Timedelta(days=1))
    assert acc.s == snapshot


def test_account_save_then_restart_is_idempotent(tmp_path, monkeypatch):
    from paper import sim
    monkeypatch.setattr(sim, "ACC_DIR", str(tmp_path / "accounts"))
    monkeypatch.setattr(sim, "SIG_DIR", str(tmp_path / "signals"))
    acc = sim.Account("K_top3")
    acc.init(D)
    acc.s["last_date"] = str(D.date())
    acc.trades = [dict(日期=str(D.date()), 操作="买入", 代码="000001.SZ", 名称="", 股数=100, 成交价=10., 金额=1000., 费用=5., 备注="")]
    acc.navs = [dict(日期=str(D.date()), 总资产=100000)]
    acc.save({})
    restarted = sim.Account("K_top3")
    assert restarted.s == acc.s
    restarted.trades = [dict(日期=str(D.date()), 操作="买入", 代码="000001.SZ", 名称="", 股数=100, 成交价=10., 金额=1000., 费用=5., 备注="")]
    restarted.save({})
    trades = pd.read_csv(Path(restarted.dir) / "trades.csv")
    assert len(trades) == 1


def test_signal_refuses_stale_r3_state(tmp_path):
    from strategies import live as A
    state = tmp_path / "risk.json"
    state.write_text(json.dumps(dict(r3=dict(exposure=1., last_date="2026-10-08"))))
    ns = argparse.Namespace(strategy="A_top20", date=str(D.date()), r3_state=str(state))
    with pytest.raises(ValueError, match="stale"):
        A.cmd_signal(ns)
