"""FB cash/share replay using exactly the daily account execution rules."""
import argparse
import os
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine.common import DATA, EXTRA, read_parts
from engine.execution import Ledger, execute_fb
from engine.market import PanelMarket
from strategies2.fb_live import candidates, market_state, K


def run(start, end, account=100_000):
    bench = pd.read_parquet(os.path.join(DATA, "bench.parquet")).set_index("datetime")["bench"]
    days = list(bench.index[(bench.index >= pd.Timestamp(start)) & (bench.index <= pd.Timestamp(end))])
    panel = read_parts(os.path.join(DATA, "panel"), start=pd.Timestamp(start) - pd.Timedelta(days=10), end=end)
    active = panel[panel["in_pool"]]
    if not active["status_known"].fillna(False).all():
        raise ValueError("Backfill dated ST status and rebuild panel first")
    fp = os.path.join(EXTRA, "execution_0935.parquet")
    execution = pd.read_parquet(fp) if os.path.exists(fp) else pd.DataFrame()
    mk, acc = PanelMarket(panel, bench, execution), Ledger(account)
    out = []
    for d in days:
        signal_day = bench.index[bench.index < d][-1]
        ma, mdn = market_state(signal_day)
        cand, _ = candidates(signal_day)
        pending = dict(K=K, cands=[dict(inst=j, ref=float(r.raw_close)) for j, r in cand.iterrows()]
                       if ma < 0 and mdn < .03 else [])
        prev = acc.total()
        acc.mark(mk, d, at="open")
        execute_fb(acc, mk, d, pending, {})
        acc.mark(mk, d)
        out.append(dict(datetime=d, value=acc.total(), cash=acc.s["cash"], ret=acc.total() / prev - 1,
                        n_hold=len(acc.s["positions"])))
    return pd.DataFrame(out), pd.DataFrame(acc.trades)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--account", type=float, default=100_000)
    a = ap.parse_args()
    nav, trades = run(a.start, a.end, a.account)
    Path("strategies2/results").mkdir(exist_ok=True)
    nav.to_csv("strategies2/results/fb_execution_v2_nav.csv", index=False, encoding="utf-8-sig")
    trades.to_csv("strategies2/results/fb_execution_v2_trades.csv", index=False, encoding="utf-8-sig")
    print("Written FB replay; missing 09:35 quotes are recorded as unfilled buys")
