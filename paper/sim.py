"""
模拟盘：每个策略一个独立账户（默认 10 万），每个交易日严格按各策略自带的 live 工具给出的清单操作

  python paper/sim.py run                      # 处理所有未处理的交易日（成交 → 盯市 → 生成下一日清单），可重复运行
  python paper/sim.py run --strategy K_top3    # 只跑一个
  python paper/sim.py summary                  # 只重新生成 paper/README.md 里的总览

每个交易日 d 的处理顺序（与各策略的实盘规则一致）：
  1) 成交：执行上一信号日生成的清单
       A_top50 / A_top20 / AE_top20 / C_top50  d 日收盘价成交（尾盘集合竞价）；涨停/停牌买不进用替补，跌停/停牌卖不出继续持有
                                                R3：按清单提示把持仓卖出约一半换成中证1000ETF，解除时卖出全部 ETF
       K_top3                                   d 日开盘价成交（开盘集合竞价），滑点千1；开盘一字涨停/停牌用替补
       FB_dip                                   d 日开盘价落在买入区间才买（低开深的优先，最多 2 只，每只半仓），
                                                买入后的下一个交易日收盘卖出（跌停/停牌顺延）
  2) 盯市：按 d 日收盘价估值（除权除息按复权因子折算成股数，相当于分红再投资）
  3) 出清单：调用各策略自己的 signal 工具，用账户的真实持仓和现金生成 d+1 的操作清单

状态保存在 paper/accounts/<策略>/：state.json（现金、持仓、待执行清单）、nav.csv（每日净值）、trades.csv（成交记录）、
positions.csv（当前持仓）；每日清单保存到 paper/signals/<日期>/<策略>.csv
说明：
  - 数据截止日由 Qlib 数据决定。17:00 运行时如果当天数据还没发布，本次不会处理当天，下次运行会自动补上（成交价仍按当天真实价格）
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
    "FB_dip": dict(kind="fb", family="FB", desc="低位首板次日低开 2–7% 开盘买，次日收盘卖，最多 2 只"),
}
ETF_CODE = "512100"
# 数据源的复权因子每天有约 0.01% 的计算噪声；变化超过 0.3% 才认为是除权除息（更小的分红忽略）
ADJ_TOL = 0.003
FB_BUY_COMM, FB_SELL_COMM, FB_STAMP, FB_SLIP = 0.00025, 0.00025, 0.0005, 0.001


# ============================================================================ 工具
def inst_of(code):
    """'603103.SH' -> 'SH603103'"""
    c = str(code).strip()
    if "." in c:
        n, ex = c.split(".")
        return ex.upper() + n
    from strategies.live import norm_code
    return norm_code(c)


def code_of(inst):
    return inst[2:] + "." + inst[:2]


def lot_of(inst):
    return 200 if inst.startswith("SH688") else 100


def fee(amount, rate, min_cost=5.0, stamp=0.0):
    if amount <= 0:
        return 0.0
    return max(amount * rate, min_cost) + amount * stamp


def trading_days():
    from engine.common import PROVIDER
    cal = pd.read_csv(os.path.join(PROVIDER, "calendars", "day.txt"), header=None)[0]
    return [pd.Timestamp(x) for x in cal]


class Market:
    """某一日的行情（未复权价、复权因子、涨跌停/停牌），带缓存"""

    COLS = ["datetime", "instrument", "raw_close", "raw_open", "factor", "up_lim", "dn_lim", "up_open", "dn_open", "susp"]

    def __init__(self):
        from engine.common import DATA
        self.DATA = DATA
        self.cache = {}
        b = pd.read_parquet(os.path.join(DATA, "bench.parquet")).set_index("datetime")["bench"]
        self.bench = b

    def day(self, d):
        d = pd.Timestamp(d)
        if d not in self.cache:
            from engine.common import read_parts
            p = read_parts(os.path.join(self.DATA, "panel"), columns=self.COLS, start=d, end=d)
            self.cache[d] = p.set_index("instrument")
        return self.cache[d]

    def quote(self, d, inst):
        """返回 dict(close, open, factor, up_lim, dn_lim, up_open, dn_open, susp)；面板外的股票从 Qlib 全市场取"""
        t = self.day(d)
        if inst in t.index:
            r = t.loc[inst]
            q = {k: r[k] for k in self.COLS[2:]}
        else:
            q = self._qlib_quote(d, inst)
        for k in ("raw_close", "raw_open", "factor"):
            q[k] = float(q[k]) if q[k] is not None and not pd.isna(q[k]) else float("nan")
        for k in ("up_lim", "dn_lim", "up_open", "dn_open", "susp"):
            q[k] = bool(q[k]) if q[k] is not None and not pd.isna(q[k]) else (k == "susp")
        for k in ("raw_close", "raw_open"):
            if q[k] > 0:
                q[k] = round(q[k], 2)       # 复权价反推的未复权价有微小误差，还原到分
        if not q["raw_close"] > 0:
            q["susp"] = True
        return q

    def _qlib_quote(self, d, inst):
        from engine.common import init_qlib
        from qlib.data import D
        init_qlib()
        try:
            f = D.features([inst], ["$close/$factor", "$open/$factor", "$factor", "$volume", "$change"], d, d)
            r = f.iloc[0].values
            vol = r[3]
            return dict(raw_close=r[0], raw_open=r[1], factor=r[2], up_lim=False, dn_lim=False, up_open=False,
                        dn_open=False, susp=not (vol > 0))
        except Exception:  # noqa: BLE001
            return dict(raw_close=np.nan, raw_open=np.nan, factor=np.nan, up_lim=False, dn_lim=False, up_open=False,
                        dn_open=False, susp=True)


def names_table():
    from engine.common import EXTRA
    return pd.read_parquet(os.path.join(EXTRA, "industry.parquet")).set_index("instrument")["code_name"].to_dict()


# ============================================================================ 账户
class Account:
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

    # ----- 估值
    def mark(self, mk, d):
        """复权折算股数 + 按 d 日收盘估值"""
        val = 0.0
        for j, p in self.s["positions"].items():
            q = mk.quote(d, j)
            f0 = p.get("factor") or 0
            if q["factor"] > 0 and f0 > 0 and abs(q["factor"] / f0 - 1) > ADJ_TOL:     # 除权除息：送转/分红折算成股数
                p["shares"] = round(p["shares"] * q["factor"] / f0, 2)
                p["factor"] = q["factor"]
            elif q["factor"] > 0 and f0 <= 0:
                p["factor"] = q["factor"]
            if q["raw_close"] > 0:
                p["last_px"] = q["raw_close"]
            val += p["shares"] * p.get("last_px", 0.0)
        return val

    def total(self):
        return self.s["cash"] + self.s["etf_value"] + sum(p["shares"] * p.get("last_px", 0.0)
                                                          for p in self.s["positions"].values())

    def stock_value(self):
        return sum(p["shares"] * p.get("last_px", 0.0) for p in self.s["positions"].values())

    # ----- 成交
    def trade(self, d, action, inst, shares, px, f, note="", name=""):
        amt = shares * px
        if action == "买入":
            self.s["cash"] -= amt + f
            p = self.s["positions"].get(inst)
            if p:
                p["cost"] += amt + f
                p["shares"] += shares
            else:
                self.s["positions"][inst] = dict(shares=float(shares), cost=amt + f, buy_date=str(d.date()),
                                                 last_px=px, factor=None)
        else:
            self.s["cash"] += amt - f
            p = self.s["positions"][inst]
            if shares >= p["shares"] - 1e-6:
                del self.s["positions"][inst]
            else:
                p["cost"] *= 1 - shares / p["shares"]
                p["shares"] -= shares
        self.trades.append(dict(日期=str(d.date()), 操作=action, 代码=code_of(inst) if inst != "ETF" else ETF_CODE,
                                名称=name, 股数=int(shares) if float(shares).is_integer() else round(shares, 2), 成交价=round(px, 3), 金额=round(amt, 2),
                                费用=round(f, 2), 备注=note))

    # ----- 保存
    def save(self, names):
        os.makedirs(self.dir, exist_ok=True)
        json.dump(self.s, open(self.fn, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        if self.trades:
            fn = os.path.join(self.dir, "trades.csv")
            df = pd.DataFrame(self.trades)
            df.to_csv(fn, mode="a", header=not os.path.exists(fn), index=False, encoding="utf-8-sig")
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


# ============================================================================ 成交逻辑
def execute_dropout(acc, mk, d, pend, names, cfg):
    """A/C 系列：d 日收盘价成交"""
    oc, cc, mc = cfg["open_cost"], cfg["close_cost"], cfg["min_cost"]
    pos = acc.s["positions"]
    # 1) 卖出清单
    for j in pend["sell"]:
        if j not in pos:
            continue
        q = mk.quote(d, j)
        if q["susp"] or q["dn_lim"]:
            acc.trades.append(dict(日期=str(d.date()), 操作="卖出失败", 代码=code_of(j), 名称=names.get(j, ""), 股数=round(pos[j]["shares"], 2),
                                   成交价="", 金额="", 费用="", 备注="跌停/停牌，继续持有"))
            continue
        sh = pos[j]["shares"]
        acc.trade(d, "卖出", j, sh, q["raw_close"], fee(sh * q["raw_close"], cc, mc), "清单卖出", names.get(j, ""))
    # 2) R3
    expo_next = pend.get("expo_next", 1.0)
    total = acc.total()
    if expo_next < 1:
        target = total * (1 - expo_next)
        if acc.s["etf_value"] < 0.9 * target:
            need = target - acc.s["etf_value"]
            sv = acc.stock_value()
            frac = min(1.0, need / sv) if sv > 0 else 0
            # 每只卖出约一半（按手取整）；只有 1 手的小仓位取整后为 0，再逐手补卖，直到凑够 ETF 目标
            plan = {}
            for j in pos:
                q = mk.quote(d, j)
                if q["susp"] or q["dn_lim"]:
                    continue
                plan[j] = [math.floor(pos[j]["shares"] * frac / lot_of(j)) * lot_of(j), q["raw_close"]]
            val = lambda: sum(sh * px for sh, px in plan.values())  # noqa: E731
            while val() < need * 0.95:
                left = [(pos[j]["shares"] - sh) * px for j, (sh, px) in plan.items()]
                if not plan or max(left) < 1:
                    break
                j = list(plan)[int(np.argmax(left))]
                plan[j][0] = min(plan[j][0] + lot_of(j), pos[j]["shares"])
            got = 0.0
            for j, (sh, px) in plan.items():
                if sh <= 0:
                    continue
                if pos[j]["shares"] - sh < lot_of(j):      # 剩下不足 1 手（除权后的零股）就整笔卖掉
                    sh = pos[j]["shares"]
                amt = sh * px
                f = fee(amt, cc, mc)
                acc.trade(d, "卖出", j, sh, px, f, "R3 半仓：卖出约一半换 ETF", names.get(j, ""))
                got += amt - f
            buy_amt = min(need, got, acc.s["cash"])
            if buy_amt > 100:
                f = fee(buy_amt, oc, mc)
                acc.s["cash"] -= buy_amt
                acc.s["etf_value"] += buy_amt - f
                acc.trades.append(dict(日期=str(d.date()), 操作="买入", 代码=ETF_CODE, 名称="中证1000ETF（指数近似）", 股数="",
                                       成交价="", 金额=round(buy_amt - f, 2), 费用=round(f, 2), 备注="R3 半仓"))
    elif acc.s["etf_value"] > 0:
        v = acc.s["etf_value"]
        f = fee(v, oc, mc)
        acc.s["cash"] += v - f
        acc.s["etf_value"] = 0.0
        acc.trades.append(dict(日期=str(d.date()), 操作="卖出", 代码=ETF_CODE, 名称="中证1000ETF（指数近似）", 股数="",
                               成交价="", 金额=round(v, 2), 费用=round(f, 2), 备注="解除 R3，卖出全部 ETF"))
    # 3) 买入（买不进/资金不足一手 → 按顺序用替补）
    backups = [b for b in pend["backup"]]
    done = set(pos)

    def try_buy(j, sh, note):
        if j in done or sh <= 0:
            return False
        q = mk.quote(d, j)
        if q["susp"] or q["up_lim"] or not q["raw_close"] > 0:
            return False
        px, lot = q["raw_close"], lot_of(j)
        while sh > 0 and sh * px + fee(sh * px, oc, mc) > acc.s["cash"]:
            sh -= lot
        if sh <= 0:
            return False
        acc.trade(d, "买入", j, sh, px, fee(sh * px, oc, mc), note, names.get(j, ""))
        pos[j]["factor"] = q["factor"]
        done.add(j)
        return True

    for b in pend["buy"]:
        if try_buy(b["inst"], b["shares"], "清单买入"):
            continue
        while backups:
            r = backups.pop(0)
            if try_buy(r["inst"], r["shares"], f"替补（替 {code_of(b['inst'])}）"):
                break


def execute_conc(acc, mk, d, pend, names, cfg):
    """K_top3：d 日开盘价成交，滑点 slip"""
    oc, cc, mc, slip = cfg["open_cost"], cfg["close_cost"], cfg["min_cost"], cfg["slip"]
    pos = acc.s["positions"]
    for j in pend["sell"]:
        if j not in pos:
            continue
        q = mk.quote(d, j)
        if q["susp"] or q["dn_open"] or not q["raw_open"] > 0:
            acc.trades.append(dict(日期=str(d.date()), 操作="卖出失败", 代码=code_of(j), 名称=names.get(j, ""), 股数=round(pos[j]["shares"], 2),
                                   成交价="", 金额="", 费用="", 备注="开盘跌停/停牌，继续持有"))
            continue
        px = q["raw_open"] * (1 - slip)
        sh = pos[j]["shares"]
        acc.trade(d, "卖出", j, sh, px, fee(sh * px, cc, mc), pend.get("why", {}).get(j, "清单卖出"), names.get(j, ""))
    slot = pend["slot"]
    backups = list(pend["backup"])
    done = set(pos)

    def try_buy(j, note):
        if j in done:
            return False
        q = mk.quote(d, j)
        if q["susp"] or q["up_open"] or not q["raw_open"] > 0:
            return False
        px, lot = q["raw_open"] * (1 + slip), lot_of(j)
        sh = math.floor(min(slot, acc.s["cash"]) / (1 + oc) / (px * lot)) * lot
        while sh > 0 and sh * px + fee(sh * px, oc, mc) > acc.s["cash"]:
            sh -= lot
        if sh <= 0:
            return False
        acc.trade(d, "买入", j, sh, px, fee(sh * px, oc, mc), note, names.get(j, ""))
        pos[j]["factor"] = q["factor"]
        done.add(j)
        return True

    for b in pend["buy"]:
        if try_buy(b["inst"], "清单买入（开盘）"):
            continue
        while backups:
            r = backups.pop(0)
            if try_buy(r["inst"], f"替补（替 {code_of(b['inst'])}）"):
                break


def execute_fb(acc, mk, d, pend, names):
    """FB_dip：开盘按区间买入；收盘卖出 d 之前买入的持仓"""
    pos = acc.s["positions"]
    held_before = [j for j, p in pos.items() if p["buy_date"] < str(d.date())]
    # 开盘买入
    if pend and pend.get("cands"):
        free = pend["K"] - len(pos)
        opts = []
        for c in pend["cands"]:
            j = c["inst"]
            if j in pos:
                continue
            q = mk.quote(d, j)
            if q["susp"] or q["up_open"] or not q["raw_open"] > 0:
                continue
            o = round(q["raw_open"], 2)
            if c["lo"] - 1e-9 <= o <= c["hi"] + 1e-9:
                opts.append((o / c["ref"] - 1, j, o, q))
        for gap, j, o, q in sorted(opts)[:max(free, 0)]:
            px, lot = o * (1 + FB_SLIP), lot_of(j)
            sh = math.floor(min(pend["slot"], acc.s["cash"]) / (1 + FB_BUY_COMM) / (px * lot)) * lot
            while sh > 0 and sh * px + fee(sh * px, FB_BUY_COMM) > acc.s["cash"]:
                sh -= lot
            if sh <= 0:
                continue
            acc.trade(d, "买入", j, sh, px, fee(sh * px, FB_BUY_COMM), f"开盘低开 {gap:+.2%}，落在买入区间", names.get(j, ""))
            pos[j]["factor"] = q["factor"]
    # 收盘卖出
    for j in held_before:
        q = mk.quote(d, j)
        p = pos[j]
        if q["susp"] or q["dn_lim"]:
            acc.trades.append(dict(日期=str(d.date()), 操作="卖出失败", 代码=code_of(j), 名称=names.get(j, ""), 股数=round(p["shares"], 2),
                                   成交价="", 金额="", 费用="", 备注="收盘跌停/停牌，顺延到下一交易日"))
            continue
        px = q["raw_close"] * (1 - FB_SLIP)
        sh = p["shares"]
        acc.trade(d, "卖出", j, sh, px, fee(sh * px, FB_SELL_COMM, stamp=FB_STAMP), "买入次日收盘卖出", names.get(j, ""))


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
    if kind == "dropout":
        from strategies import live as A
        orig = A.r3_state

        def rec(excess, cfg):
            out = orig(excess, cfg)
            r3.update(expo_next=float(out[0]), expo_today=float(out[1]), dd=float(out[2]))
            return out

        A.r3_state = rec
        try:
            with contextlib.redirect_stdout(buf):
                if STRATS[name]["family"] == "A":
                    A.cmd_signal(ns)
                else:
                    from strategies2 import live as S2
                    S2.cmd_signal(ns)
        finally:
            A.r3_state = orig
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
    for d in todo:
        prev_total = acc.total()
        n0 = len(acc.trades)
        pend = acc.s.get("pending")
        # 盯市到开盘之前需要先做除权折算（K/FB 开盘成交），ETF 按指数当日收益
        acc.mark(mk, d)
        acc.s["etf_value"] *= 1 + float(mk.bench.get(d, 0.0))     # 昨日持有的 ETF 吃当日收益，再按收盘成交
        if kind == "dropout" and pend:
            execute_dropout(acc, mk, d, pend, names, cfg)
        elif kind == "conc" and pend:
            execute_conc(acc, mk, d, pend, names, cfg)
        elif kind == "fb":
            execute_fb(acc, mk, d, pend, names)
        acc.mark(mk, d)
        tot = acc.total()
        b = float(mk.bench.get(d, 0.0))
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
            orders.append("\n".join(lines))
    md = ["# 模拟盘（每个策略独立 10 万账户）", "",
          "由 GitHub Actions 每个交易日 17:00（北京时间）自动运行：用当天收盘价成交上一交易日的清单 → 盯市 → 生成下一交易日清单。",
          "数据源当天没更新时会在之后的运行里自动补齐（按真实成交日价格）。说明见 [paper/HOWTO.md](HOWTO.md)。", "",
          "## 账户总览", "",
          "| 策略 | 规则 | 起始日 | 最新日 | 总资产 | 当日 | 累计 | 中证1000累计 | 超额累计 | 最大回撤 | 持仓 | 现金 |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|", *rows, "",
          "明细：`paper/accounts/<策略>/` 下的 nav.csv（每日净值）、trades.csv（成交）、positions.csv（持仓）；"
          "每日完整清单在 `paper/signals/<日期>/`。", "",
          "## 下一交易日操作清单", *orders, "",
          "> 仅供学习研究，不构成投资建议。模拟盘按收盘/开盘价成交，未计冲击成本，实盘会有差距。"]
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
