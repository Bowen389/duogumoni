"""
模拟盘：每个策略一个独立账户（默认 10 万），每个交易日按 live 工具选股清单操作，成交时重新检查资金/空位并计算股数

  python paper/sim.py run                      # 处理所有未处理的交易日（成交 → 盯市 → 生成下一日清单），可重复运行
  python paper/sim.py run --strategy K_top3    # 只跑一个
  python paper/sim.py summary                  # 只重新生成 paper/README.md 里的总览

每个交易日 d 的处理顺序（成交为日线/5分钟代理，不保证实盘成交）：
  1) 成交：执行上一信号日生成的清单
       A_top50 / A_top20 / AE_top20 / C_top50  d 日收盘价成交（尾盘集合竞价）；涨停/停牌买不进用替补，跌停/停牌卖不出继续持有
                                                R3：按清单提示把持仓卖出约一半换成中证1000ETF，解除时卖出全部 ETF
       K_top3                                   d 日开盘价成交（开盘集合竞价），滑点千1；开盘一字涨停/停牌用替补
       FB_dip                                   d 日开盘价用于低开筛选，09:35首5分钟价近似成交（限价/缺价不买），
                                                买入后的下一个交易日收盘卖出（跌停/停牌顺延）
  2) 盯市：按 d 日收盘价估值（除权除息按复权因子折算成股数，相当于分红再投资）
  3) 出清单：调用各策略自己的 signal 工具，用账户的真实持仓和现金生成 d+1 的操作清单

状态保存在 paper/accounts/<策略>/：state.json（现金、持仓、待执行清单）、nav.csv（每日净值）、trades.csv（成交记录）、
positions.csv（当前持仓）；每日清单保存到 paper/signals/<日期>/<策略>.csv
说明：
  - 社区数据包只用于历史底座；Baostock 直接补齐日线和状态，未达到独立交易日历要求时任务失败，不更新账户
  - 中证1000ETF 没有单独的价格数据，用中证1000 指数日收益近似
"""
import argparse
import contextlib
import io
import json
import math
import os
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

PAPER = os.path.join(ROOT, "paper")
ACC_DIR = os.path.join(PAPER, "accounts")
SIG_DIR = os.path.join(PAPER, "signals")
INIT_CASH = float(os.environ.get("PAPER_CASH", 100_000))

# kind: dropout = A/C 系列（尾盘成交 + R3）；conc = K_top3（开盘成交）；fb = 机会仓
STRATS = {
    "A_top50": dict(kind="dropout", family="A", desc="散户因子 1 模型，持 50 只，每天换 2 只，R3 风控"),
    "A_top20": dict(kind="dropout", family="A", desc="散户因子 1 模型，持 20 只，每天换 1 只，R3 风控"),
    "AE_top20": dict(kind="dropout", family="A", desc="两模型集成，持 20 只，每天换 1 只，R3 风控"),
    "C_top50": dict(kind="dropout", family="C", desc="三模型集成（含股东户数），持 50 只，每天换 2 只，R3 风控"),
    "K_top3": dict(kind="conc", family="C", desc="六个 5 日模型集成，最多 3 只，均线择时空仓，开盘成交"),
    "FB_dip": dict(kind="fb", family="FB", desc="低位首板次日低开 2–7% 09:35首5分钟价近似买，次日收盘卖，最多 2 只"),
}
from engine.execution import (Ledger, ETF_CODE, ADJ_TOL, EXECUTION_VERSION, code_of, lot_of, fee,
                              execute_dropout, execute_conc, execute_fb, dropout_orders, advance_r3)
from engine.market import Market
import copy


# ============================================================================ 工具
def inst_of(code):
    """'603103.SH' -> 'SH603103'"""
    c = str(code).strip()
    if "." in c:
        n, ex = c.split(".")
        return ex.upper() + n
    from strategies.live import norm_code
    return norm_code(c)


def trading_days():
    from engine.common import PROVIDER
    cal = pd.read_csv(os.path.join(PROVIDER, "calendars", "day.txt"), header=None)[0]
    return [pd.Timestamp(x) for x in cal]


def names_table():
    from engine.common import EXTRA
    return pd.read_parquet(os.path.join(EXTRA, "industry.parquet")).set_index("instrument")["code_name"].to_dict()


# ============================================================================ 账户
class Account(Ledger):
    def __init__(self, name):
        self.name = name
        self.dir = os.path.join(ACC_DIR, name)
        self.fn = os.path.join(self.dir, "state.json")
        if os.path.exists(self.fn):
            self.s = json.load(open(self.fn, encoding="utf-8"))
        else:
            self.s = None
        self.trades = []
        self.navs = []

    @property
    def exists(self):
        return self.s is not None

    def init(self, start):
        self.s = dict(strategy=self.name, init_cash=INIT_CASH, start_date=str(start.date()), last_date=None,
                      cash=INIT_CASH, etf_value=0.0, positions={}, pending=None)

    # ----- 保存
    def save(self, names):
        os.makedirs(self.dir, exist_ok=True)
        if self.trades:
            fn = os.path.join(self.dir, "trades.csv")
            df = pd.DataFrame(self.trades)
            if os.path.exists(fn):
                old = pd.read_csv(fn, encoding="utf-8-sig")
                df = pd.concat([old[~old["日期"].isin(df["日期"])], df], ignore_index=True)
            df.to_csv(fn, index=False, encoding="utf-8-sig")
            self.trades = []
        if self.navs:
            fn = os.path.join(self.dir, "nav.csv")
            df = pd.DataFrame(self.navs)
            if os.path.exists(fn):
                old = pd.read_csv(fn, encoding="utf-8-sig")
                df = pd.concat([old[~old["日期"].isin(df["日期"])], df], ignore_index=True)
            df.to_csv(fn, index=False, encoding="utf-8-sig")
            self.navs = []
        rows = []
        for j, p in sorted(self.s["positions"].items()):
            mv = p["shares"] * p.get("last_px", 0.0)
            rows.append(dict(代码=code_of(j), 名称=names.get(j, ""), 股数=round(p["shares"], 2), 现价=p.get("last_px"),
                             市值=round(mv, 2), 成本=round(p["cost"], 2), 浮动盈亏=round(mv - p["cost"], 2),
                             买入日期=p.get("buy_date", "")))
        if self.s["etf_value"] > 0:
            rows.append(dict(代码=ETF_CODE, 名称="中证1000ETF（指数近似）", 股数="", 现价="", 市值=round(self.s["etf_value"], 2),
                             成本="", 浮动盈亏="", 买入日期=""))
        pd.DataFrame(rows, columns=["代码", "名称", "股数", "现价", "市值", "成本", "浮动盈亏", "买入日期"]).to_csv(
            os.path.join(self.dir, "positions.csv"), index=False, encoding="utf-8-sig")
        if self.s.get("last_date"):      # 每天的收盘持仓另存一份到当天的清单文件夹（日报用）
            dd = os.path.join(SIG_DIR, self.s["last_date"])
            os.makedirs(dd, exist_ok=True)
            pd.DataFrame(rows, columns=["代码", "名称", "股数", "现价", "市值", "成本", "浮动盈亏", "买入日期"]).to_csv(
                os.path.join(dd, f"{self.name}_positions.csv"), index=False, encoding="utf-8-sig")
        tmp = self.fn + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.s, f, ensure_ascii=False, indent=1)
        os.replace(tmp, self.fn)


# ============================================================================ 生成清单（调用各策略自己的 live 工具）
def holdings_file(acc, with_date=False):
    fn = os.path.join(PAPER, ".tmp", f"{acc.name}_hold.csv")
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    rows = [dict(code=code_of(j), shares=p["shares"], buy_date=p.get("buy_date", "")) for j, p in acc.s["positions"].items()]
    df = pd.DataFrame(rows, columns=["code", "shares", "buy_date"])
    if not with_date:
        df = df[["code", "shares"]]
    df.to_csv(fn, index=False)
    return fn if len(rows) else None


def gen_signal(acc, d):
    """返回 (pending dict, 清单 DataFrame, 屏幕输出)"""
    name, kind = acc.name, STRATS[acc.name]["kind"]
    hfile = holdings_file(acc, with_date=(kind == "fb"))
    ns = argparse.Namespace(strategy=name, holdings=hfile, cash=float(acc.s["cash"]), etf_value=float(acc.s["etf_value"]),
                            date=str(d.date()), r3_lookback=365, day_index=0, lookback=365)
    buf = io.StringIO()
    r3 = {}
    ns.r3_override = acc.s.get("r3", dict(exposure=1.0, dd=0.0))
    if kind == "dropout":
        from strategies import live as A
        with contextlib.redirect_stdout(buf):
            if STRATS[name]["family"] == "A":
                result = A.cmd_signal(ns)
            else:
                from strategies2 import live as S2
                result = S2.cmd_signal(ns)
        r3.update(expo_next=float(ns.r3_override["exposure"]), dd=float(ns.r3_override.get("dd", 0)))
        if "reference" in acc.s:
            cfg = strat_cfg(name)
            sell, buy, backup, _ = dropout_orders(result.attrs["score"], list(acc.s["reference"]["positions"]),
                                                  cfg["topk"], cfg["n_drop"], result.attrs["eligible"])
            acc.s["reference_pending"] = dict(sell=sell, buy=[dict(inst=j) for j in buy],
                                               backup=[dict(inst=j) for j in backup], expo_next=1.0)
        fn = f"strategies/signals/{name}_{d.date()}.csv"
    elif kind == "conc":
        from strategies2 import live as S2
        ns.lookback = 0          # 只出清单，不重复跑近一年模拟（本模拟盘自己就是实盘记录）
        with contextlib.redirect_stdout(buf):
            S2.cmd_signal(ns)
        fn = f"strategies2/signals/{name}_{d.date()}.csv"
    else:
        from strategies2 import fb_live as FB
        with contextlib.redirect_stdout(buf):
            FB.cmd_signal(argparse.Namespace(holdings=hfile, cash=float(acc.s["cash"]), date=str(d.date())))
        fn = f"strategies2/signals/FB_dip_{d.date()}.csv"
    text = buf.getvalue()
    try:
        out = pd.read_csv(fn, dtype={"代码": str}, encoding="utf-8-sig")
    except pd.errors.EmptyDataError:
        out = pd.DataFrame(columns=["操作", "代码"])
    pend = dict(signal_date=str(d.date()), kind=kind)
    if kind == "dropout":
        pend.update(r3)
        pend["sell"] = [inst_of(c) for c in out.loc[out["操作"] == "卖出", "代码"]]
        pend["buy"] = [dict(inst=inst_of(r.代码), shares=int(r.股数)) for r in out[out["操作"] == "买入"].itertuples()]
        pend["backup"] = [dict(inst=inst_of(r.代码), shares=int(r.股数)) for r in out[out["操作"] == "替补"].itertuples()]
    elif kind == "conc":
        from strategies2.config import STRATEGIES as S2S
        pend["sell"] = [inst_of(c) for c in out.loc[out["操作"] == "卖出", "代码"]]
        pend["why"] = {inst_of(r.代码): str(r.备注) for r in out[out["操作"] == "卖出"].itertuples()}
        pend["buy"] = [dict(inst=inst_of(c)) for c in out.loc[out["操作"] == "买入", "代码"]]
        pend["backup"] = [dict(inst=inst_of(c)) for c in out.loc[out["操作"] == "替补", "代码"]]
        pend["slot"] = acc.total() / S2S["K_top3"]["k"]
        pend["k"] = S2S["K_top3"]["k"]
    else:
        from strategies2 import fb_live as FB
        ok = "【可以买】" in text
        c = out[out["操作"].astype(str).str.startswith("候选")] if len(out) else out
        cands = []
        for r in c.itertuples():
            lo, hi = [float(x) for x in str(r.买入区间).split("~")]
            cands.append(dict(inst=inst_of(r.代码), lo=lo, hi=hi, ref=float(r.参考价)))
        pend.update(cands=cands if ok else [], K=FB.K, slot=acc.total() / FB.K)
    # 保存清单副本
    dd = os.path.join(SIG_DIR, str(d.date()))
    os.makedirs(dd, exist_ok=True)
    out.to_csv(os.path.join(dd, f"{name}.csv"), index=False, encoding="utf-8-sig")
    with open(os.path.join(dd, f"{name}.txt"), "w", encoding="utf-8") as f:
        f.write(text)
    return pend, out, text


# ============================================================================ 主流程
def strat_cfg(name):
    if STRATS[name]["family"] == "A":
        from strategies.config import STRATEGIES
        return STRATEGIES[name]
    if name == "FB_dip":
        return {}
    from strategies2.config import STRATEGIES
    return STRATEGIES[name]


def migrate_execution(acc, d):
    if acc.s.get("execution_version") == EXECUTION_VERSION:
        return
    acc.s["execution_version"] = EXECUTION_VERSION
    acc.s["execution_start"] = str(d.date())
    if STRATS[acc.name]["kind"] == "dropout":
        ref = copy.deepcopy({k: acc.s[k] for k in ("cash", "etf_value", "positions")})
        ref["cash"] += ref["etf_value"]
        ref["etf_value"] = 0.0
        acc.s["reference"] = ref
        old = acc.s.get("pending") or {}
        expo = float(old.get("expo_next", 1.0))
        # Retain an existing half-exposure regime across migration. The baseline
        # relative drawdown starts at the entry threshold, rather than auto-unhedging.
        dd = min(float(old.get("dd", -0.08)), -0.08) if expo < 1 else 0.0
        acc.s["r3"] = dict(nav=1.0 + dd, peak=1.0, exposure=expo, dd=dd, last_date=str(d.date()),
                           migration_baseline=True)


def run_one(name, start=None, until=None):
    mk = Market()
    names = names_table()
    days = trading_days()
    data_last = mk.bench.index.max()
    days = [d for d in days if d <= data_last]
    if until:
        days = [d for d in days if d <= pd.Timestamp(until)]
    acc = Account(name)
    if not acc.exists:
        s = pd.Timestamp(start) if start else days[-1]
        s = max(d for d in days if d <= s)
        acc.init(s)
        acc.navs.append(dict(日期=str(s.date()), 现金=round(acc.s["cash"], 2), 股票市值=0.0, ETF市值=0.0,
                             总资产=round(INIT_CASH, 2), 当日收益=0.0, 累计收益=0.0, 中证1000当日=0.0,
                             中证1000累计=0.0, 超额累计=0.0, 持仓数=0, 成交笔数=0))
        acc.s["bench_cum"] = 1.0
        print(f"[{name}] 新建账户：初始资金 {INIT_CASH:,.0f}，起始日 {s.date()}", flush=True)
        migrate_execution(acc, s)
        pend, out, text = gen_signal(acc, s)
        acc.s["pending"] = pend
        acc.s["last_date"] = str(s.date())
        acc.save(names)
        print(text, flush=True)
        todo = [d for d in days if d > s]
    else:
        todo = [d for d in days if d > pd.Timestamp(acc.s["last_date"])]
    if not todo:
        print(f"[{name}] 已是最新（{acc.s['last_date']}），数据最新交易日 {data_last.date()}", flush=True)
        return
    cfg = strat_cfg(name)
    kind = STRATS[name]["kind"]
    if acc.s.get("execution_version") != EXECUTION_VERSION:
        prior = pd.Timestamp(acc.s["last_date"])
        migrate_execution(acc, prior)
        acc.s["pending"], _, _ = gen_signal(acc, prior)
    for d in todo:
        prev_total = acc.total()
        n0 = len(acc.trades)
        pend = acc.s.get("pending")
        # 盯市到开盘之前需要先做除权折算（K/FB 开盘成交），ETF 按指数当日收益
        acc.mark(mk, d, at="close" if kind == "dropout" else "open")
        acc.s["etf_value"] *= 1 + float(mk.bench.get(d, 0.0))     # 昨日持有的 ETF 吃当日收益，再按收盘成交
        if kind == "dropout" and pend:
            execute_dropout(acc, mk, d, pend, names, cfg)
        elif kind == "conc" and pend:
            execute_conc(acc, mk, d, pend, names, cfg)
        elif kind == "fb":
            execute_fb(acc, mk, d, pend, names)
        acc.mark(mk, d)
        if kind == "dropout":
            reference = Ledger(state=acc.s["reference"])
            prev_ref = reference.total()
            reference.mark(mk, d)
            execute_dropout(reference, mk, d, acc.s["reference_pending"], names, cfg)
            reference.mark(mk, d)
            advance_r3(acc.s["r3"], reference.total() / prev_ref - 1, float(mk.bench.get(d, 0)), d, cfg)
        tot = acc.total()
        b = float(mk.bench.get(d, 0.0))
        quotes = mk.day(d)
        inpool = quotes[quotes["in_pool"]]
        acc.s["data_quality"] = dict(date=str(d.date()), pool_count=len(inpool),
                                      unknown_status=int((~inpool["status_known"].fillna(False)).sum()),
                                      eligible_buys=int(inpool["buy_ok"].fillna(False).sum()))
        acc.s["bench_cum"] = acc.s.get("bench_cum", 1.0) * (1 + b)
        cum = tot / acc.s["init_cash"] - 1
        acc.navs.append(dict(日期=str(d.date()), 现金=round(acc.s["cash"], 2), 股票市值=round(acc.stock_value(), 2),
                             ETF市值=round(acc.s["etf_value"], 2), 总资产=round(tot, 2), 当日收益=round(tot / prev_total - 1, 6),
                             累计收益=round(cum, 6), 中证1000当日=round(b, 6), 中证1000累计=round(acc.s["bench_cum"] - 1, 6),
                             超额累计=round(cum - (acc.s["bench_cum"] - 1), 6), 持仓数=len(acc.s["positions"]),
                             成交笔数=len([t for t in acc.trades[n0:] if t["操作"] in ("买入", "卖出")])))
        print(f"[{name}] {d.date()} 成交 {len(acc.trades) - n0} 笔，总资产 {tot:,.0f}（当日 {tot / prev_total - 1:+.2%}，"
              f"累计 {cum:+.2%}），持仓 {len(acc.s['positions'])} 只", flush=True)
        pend, out, text = gen_signal(acc, d)
        acc.s["pending"] = pend
        acc.s["last_date"] = str(d.date())
        acc.save(names)
    print(text, flush=True)


def summary():
    rows, orders = [], []
    for name, meta in STRATS.items():
        acc = Account(name)
        if not acc.exists:
            continue
        nav = pd.read_csv(os.path.join(acc.dir, "nav.csv"), encoding="utf-8-sig")
        last = nav.iloc[-1]
        v = nav["总资产"]
        mdd = float((v / v.cummax() - 1).min())
        rows.append(f"| **{name}** | {meta['desc']} | {acc.s['start_date']} | {last['日期']} | {last['总资产']:,.0f} | "
                    f"{last['当日收益']:+.2%} | {last['累计收益']:+.2%} | {last['中证1000累计']:+.2%} | {last['超额累计']:+.2%} | "
                    f"{mdd:.2%} | {int(last['持仓数'])} | {acc.s['cash']:,.0f} |")
        p = acc.s.get("pending") or {}
        sd = p.get("signal_date", "")
        fn = os.path.join(SIG_DIR, sd, f"{name}.csv")
        if os.path.exists(fn):
            try:
                o = pd.read_csv(fn, dtype={"代码": str}, encoding="utf-8-sig")
            except pd.errors.EmptyDataError:
                o = pd.DataFrame(columns=["操作"])
            if len(o):
                bk = o[o["操作"] == "替补"].head(5)          # 买入目标买不进/资金不足一手时按顺序用替补
                o = pd.concat([o[~o["操作"].isin(["替补", "继续持有"])], bk]) if (o["操作"] == "买入").any() else \
                    o[~o["操作"].isin(["替补", "继续持有"])]
            extra = ""
            if p.get("kind") == "dropout" and p.get("expo_next", 1) < 1:
                extra = f"（R3 半仓：股票仓位 {p['expo_next']:.0%}）"
            if p.get("kind") == "fb" and not p.get("cands") and len(o) == 0:
                extra = "（市场条件不满足或没有候选，明天不买）"
            lines = [f"\n### {name}｜信号日 {sd}{extra}\n"]
            if len(o):
                cols = [c for c in ["操作", "代码", "名称", "股数", "参考价", "买入区间", "参考金额", "备注"] if c in o.columns]
                o = o[cols].fillna("")
                lines.append("| " + " | ".join(cols) + " |")
                lines.append("|" + "---|" * len(cols))
                for r in o.itertuples(index=False):
                    lines.append("| " + " | ".join(str(x) for x in r) + " |")
            else:
                lines.append("无操作")
            quality = acc.s.get("data_quality", {})
            if quality.get("unknown_status", 0):
                lines.append(f"状态数据缺失：{quality['unknown_status']}/{quality['pool_count']} 只池内股票禁止新买入。")
            orders.append("\n".join(lines))
    md = ["# 模拟盘（每个策略独立 10 万账户）", "",
          "> 执行v2：旧历史保留，切换日期见state.json的execution_start；[规则和限制](../EXECUTION_V2.md)。", "",
          "由 GitHub Actions 每个交易日 18:47（北京时间）自动运行，20:17 / 22:47 重试：用当天收盘价成交上一交易日的清单 → 盯市 → 生成下一交易日清单。",
          "当日日线与交易状态直接从 Baostock 获取；数据不完整时任务报错，账户不更新。说明见 [paper/HOWTO.md](HOWTO.md)。", "",
          "## 账户总览", "",
          "| 策略 | 规则 | 起始日 | 最新日 | 总资产 | 当日 | 累计 | 中证1000累计 | 超额累计 | 最大回撤 | 持仓 | 现金 |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|", *rows, "",
          "明细：`paper/accounts/<策略>/` 下的 nav.csv（每日净值）、trades.csv（成交）、positions.csv（持仓）；"
          "每日完整清单在 `paper/signals/<日期>/`。", "",
          "## 下一交易日操作清单", *orders, "",
          "> 仅供学习研究，不构成投资建议。模拟盘按对应时点报价近似成交（FB为09:35首5分钟价）；缺价不买，未计真实排队和冲击成本。"]
    open(os.path.join(PAPER, "README.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    sys.path.insert(0, PAPER)
    import report                    # 每日日报 + 历史总览（paper/report.py）
    report.build(STRATS, "模拟盘")
    print("\n".join(rows))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--strategy", choices=list(STRATS), action="append")
    r.add_argument("--start", help="新建账户的起始信号日（默认最新交易日）")
    r.add_argument("--until", help="只处理到这一天（测试用）")
    r.add_argument("--inproc", action="store_true", help="在同一进程里跑（默认每个策略一个子进程，互不影响）")
    r.add_argument("--no-summary", action="store_true", help=argparse.SUPPRESS)
    sub.add_parser("summary")
    a = ap.parse_args()
    if a.cmd == "summary":
        return summary()
    names = a.strategy or list(STRATS)
    failed = []
    for n in names:
        if a.inproc or len(names) == 1:
            run_one(n, a.start, a.until)
        else:
            cmd = [sys.executable, os.path.abspath(__file__), "run", "--strategy", n, "--no-summary"]
            if a.start:
                cmd += ["--start", a.start]
            if a.until:
                cmd += ["--until", a.until]
            print(f"\n================ {n} ================", flush=True)
            if subprocess.call(cmd) != 0:
                failed.append(n)
    if not a.no_summary:
        summary()
    if failed:
        sys.exit(f"失败的策略：{failed}")


if __name__ == "__main__":
    main()
