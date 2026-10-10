"""
A 股回测器（按 Qlib TopkDropout 的规则复刻，另外支持周期调仓+缓冲区、拥挤度控仓）

规则：T-1 日收盘后的分数 → T 日收盘价成交；涨停阻买、跌停阻卖、停牌阻买卖；
      买入费率 open_cost，卖出费率 close_cost（含印花税），滑点 slip 双边另计。
用法（Python）：
    from engine.backtest import Backtester
    bt = Backtester("2023-01-01", "2026-09-18")
    res = bt.run(pred_df, mode="dropout", topk=50, n_drop=5)
    bt.metrics(res)
"""
import os

import numpy as np
import pandas as pd

from engine.common import DATA, read_parts

N_YEAR = 238  # 与 Qlib 年化口径一致


class Backtester:
    def __init__(self, start, end, st_filter=True, panel=None, bench=None, strict_status=True):
        self.start, self.end = pd.Timestamp(start), pd.Timestamp(end)
        pre = self.start - pd.Timedelta(days=40)
        self.panel = panel if panel is not None else read_parts(os.path.join(DATA, "panel"),
            columns=["datetime", "instrument", "raw_close", "raw_open", "factor", "change", "susp", "up_open", "dn_open",
                     "up_lim", "dn_lim", "buy_ok", "in_pool", "is_st", "is_delisted", "status_known"], start=pre, end=self.end)
        self.bench = bench if bench is not None else pd.read_parquet(os.path.join(DATA, "bench.parquet")).set_index("datetime")["bench"]
        self.bench = self.bench.sort_index()
        self.dates = list(self.bench.index[(self.bench.index >= pre) & (self.bench.index <= self.end)])
        if strict_status:
            active = self.panel[self.panel["in_pool"] & (self.panel["datetime"] >= self.start)]
            if "status_known" not in active or not active["status_known"].fillna(False).all():
                raise ValueError("Historical ST status missing. Backfill data_fetch/fetch_status.py --start START --end END, then rebuild panel. No unverifiable backtest is published.")
        self.limit_rule = "board"

    def use_limit_rule(self, rule="board"):
        self.limit_rule = rule

    def run(self, pred, mode="dropout", topk=50, n_drop=5, rebalance=5, buffer=1.5,
            open_cost=0.0005, close_cost=0.0010, slip=0.0, risk_degree=0.95, account=1e6, lot=True,
            min_cost=5.0, every=1, exec_at="close", r3_cfg=None):
        from engine.execution import Ledger, dropout_orders, signal_score, execute_dropout, advance_r3
        from engine.market import PanelMarket
        if not lot:
            raise ValueError("Execution v2 uses real share quantities; fractional lot=False research is no longer supported")
        if exec_at not in ("open", "close"):
            raise ValueError(exec_at)
        if r3_cfg and (mode != "dropout" or exec_at != "close"):
            raise ValueError("R3 is defined for close-executed dropout accounts")
        panel = self.panel.copy()
        if self.limit_rule == "qlib":
            panel["up_lim"] |= panel["change"] >= .095
            panel["dn_lim"] |= panel["change"] <= -.095
        mk = PanelMarket(panel, self.bench)
        cfg = dict(topk=topk, n_drop=n_drop, every=every, open_cost=open_cost, close_cost=close_cost,
                   min_cost=min_cost, risk_degree=risk_degree, slip=slip, exec_at=exec_at)
        actual, reference = Ledger(account), Ledger(account)
        r3 = dict(nav=1.0, peak=1.0, exposure=1.0, dd=0.0)
        scores = {pd.Timestamp(d): g.set_index("instrument")["score"] for d, g in pred.groupby("datetime")}
        rec = []
        def plan(acc, score, day):
            eligible = day["buy_ok"].fillna(False)
            if mode == "dropout":
                sell, buy, backup, _ = dropout_orders(score, list(acc.s["positions"]), topk, n_drop, eligible)
            else:
                ranks = {j: r for r, j in enumerate(score.sort_values(ascending=False, kind="stable").index)}
                keep = [j for j in acc.s["positions"] if ranks.get(j, 1e9) < topk * buffer]
                sell = [j for j in acc.s["positions"] if j not in keep]
                cand = [j for j in score.sort_values(ascending=False, kind="stable").index
                        if j not in acc.s["positions"] and eligible.get(j, False)]
                n = max(topk - len(keep), 0)
                buy, backup = cand[:n], cand[n:n + max(5, 2 * n)]
            return dict(sell=sell, buy=[dict(inst=j) for j in buy], backup=[dict(inst=j) for j in backup])
        test_days = [d for d in self.dates if d >= self.start]
        if not test_days:
            raise ValueError("Empty backtest date range")
        for i, d in enumerate(test_days):
            prev = actual.total()
            prev_ref = reference.total()
            at = exec_at
            actual.mark(mk, d, at=at)
            reference.mark(mk, d, at=at)
            b = float(self.bench.get(d, 0))
            actual.s["etf_value"] *= 1 + b
            idx = self.dates.index(d)
            signal_day = self.dates[idx - 1] if idx else None
            raw = scores.get(signal_day, pd.Series(dtype=float))
            day = mk.day(signal_day) if signal_day is not None else pd.DataFrame()
            score = signal_score(raw, day) if not day.empty else pd.Series(dtype=float)
            do_trade = not score.empty and i % (every if mode == "dropout" else rebalance) == 0
            n0 = len(actual.trades)
            exposure = r3["exposure"] if r3_cfg else 1.0
            if do_trade:
                pa = plan(actual, score, day)
                pa["expo_next"] = exposure
                execute_dropout(actual, mk, d, pa, {}, cfg)
                if r3_cfg:
                    pr = plan(reference, score, day)
                    pr["expo_next"] = 1.0
                    execute_dropout(reference, mk, d, pr, {}, cfg)
            actual.mark(mk, d)
            reference.mark(mk, d)
            if r3_cfg:
                r3["previous_exposure"] = exposure
                advance_r3(r3, reference.total() / prev_ref - 1, b, d, r3_cfg)
            trades = [t for t in actual.trades[n0:] if t["操作"] in ("买入", "卖出")]
            traded = sum(t["金额"] for t in trades)
            cost = sum(t["费用"] for t in trades)
            value = actual.total()
            ret = value / prev - 1
            rec.append(dict(datetime=d, value=value, turnover=traded / prev, cost=cost / prev,
                            n_hold=len(actual.s["positions"]), cash=actual.s["cash"], etf_value=actual.s["etf_value"],
                            return_=ret, bench=b, excess=ret - b, exposure=exposure,
                            reference_value=reference.total() if r3_cfg else value))
        self.last_r3, self.last_account = r3, actual.s
        self.trades = pd.DataFrame(actual.trades)
        return pd.DataFrame(rec).rename(columns={"return_": "return"}).set_index("datetime")

    # ------------------------------------------------------------------
    @staticmethod
    def overlay(res, exposure, switch_cost=0.0002):
        """拥挤度控仓：未暴露部分换成指数（期货/ETF），超额按 exposure 缩放。exposure 必须只用 T-1 及以前的信息。"""
        e = exposure.reindex(res.index).ffill().fillna(1.0).clip(0, 1)
        out = res.copy()
        chg = e.diff().abs().fillna(0)
        out["return"] = e * res["return"] + (1 - e) * res["bench"] - chg * switch_cost
        out["excess"] = out["return"] - out["bench"]
        out["exposure"] = e
        return out

    @staticmethod
    def metrics(res):
        ex = res["excess"]
        cum = ex.cumsum()
        nav = (1 + res["return"]).cumprod()
        yearly = ex.groupby(ex.index.year).apply(lambda x: (1 + x).prod() - 1)
        m = {
            "超额年化": ex.mean() * N_YEAR,
            "信息比率": ex.mean() / (ex.std() + 1e-12) * np.sqrt(N_YEAR),
            "超额回撤": (cum - cum.cummax()).min(),
            "策略年化": nav.iloc[-1] ** (N_YEAR / len(nav)) - 1,
            "策略回撤": (nav / nav.cummax() - 1).min(),
            "日均换手": res["turnover"].mean(),
            "年化成本": res["cost"].mean() * N_YEAR,
        }
        m.update({f"超额{y}": v for y, v in yearly.items()})
        return m

    @staticmethod
    def plot(results: dict, path, title=""):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Noto Serif CJK SC", "WenQuanYi Micro Hei",
                                           "SimHei", "Microsoft YaHei", "PingFang SC", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
        for k, r in results.items():
            (1 + r["excess"]).cumprod().plot(ax=ax[0], label=k)
        ax[0].set_title(title or "cumulative excess return vs CSI1000 (after cost)")
        ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
        first = list(results.values())[0]
        (1 + first["return"]).cumprod().plot(ax=ax[1], label=f"strategy: {list(results)[0]}")
        (1 + first["bench"]).cumprod().plot(ax=ax[1], label="CSI1000")
        ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
        fig.tight_layout(); fig.savefig(path, dpi=105); plt.close(fig)
