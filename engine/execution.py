"""Shared cash/share execution for historical replay and daily paper accounts.

Orders are decided on the previous signal day. Only execution prices, transaction
eligibility and cash/slots are checked on the execution day. ETF values remain an
explicit index-return proxy; corporate actions remain a total-return share proxy.
"""
import math

ETF_CODE = "512100"
ADJ_TOL = 0.003
FB_BUY_COMM, FB_SELL_COMM, FB_SLIP = 0.00025, 0.00025, 0.001
EXECUTION_VERSION = 2


def code_of(inst):
    return inst[2:] + "." + inst[:2]


def lot_of(inst):
    return 200 if inst.startswith("SH688") else 100


def fee(amount, rate, min_cost=5.0, stamp=0.0):
    return max(amount * rate, min_cost) + amount * stamp if amount > 0 else 0.0


def stamp_rate(d):
    return 0.001 if str(d.date()) < "2023-08-28" else 0.0005


def sell_fee(amount, cfg, d):
    # Config close_cost includes the current 0.05% stamp duty. Minimum applies
    # to the commission component, never to commission + stamp duty together.
    return fee(amount, max(cfg["close_cost"] - 0.0005, 0), cfg["min_cost"], stamp_rate(d))


def affordable(inst, budget, px, rate, min_cost):
    if not (math.isfinite(px) and px > 0 and budget > 0):
        return 0
    step = 1 if inst.startswith("SH688") else 100
    sh = math.floor(budget / (1 + rate) / px / step) * step
    while sh > 0 and sh * px + fee(sh * px, rate, min_cost) > budget + 1e-8:
        sh -= step
    return sh if sh >= lot_of(inst) else 0


class Ledger:
    def __init__(self, cash=0.0, state=None):
        self.s = state if state is not None else dict(cash=float(cash), etf_value=0.0, positions={})
        self.trades = []

    def total(self):
        return self.s["cash"] + self.s["etf_value"] + self.stock_value()

    def stock_value(self):
        return sum(p["shares"] * p.get("last_px", 0.0) for p in self.s["positions"].values())

    def mark(self, mk, d, at="close"):
        for j, p in self.s["positions"].items():
            q = mk.quote(d, j)
            f0 = p.get("factor") or 0
            f = q.get("factor", float("nan"))
            if f > 0 and f0 > 0 and abs(f / f0 - 1) > ADJ_TOL:
                p["shares"] = round(p["shares"] * f / f0, 6)
                p["factor"] = f
            elif f > 0 and f0 <= 0:
                p["factor"] = f
            px = q.get("raw_" + at, float("nan"))
            if px > 0:
                p["last_px"] = px
        return self.stock_value()

    def trade(self, d, action, inst, shares, px, f, note="", name=""):
        if not (shares > 0 and px > 0 and math.isfinite(px) and f >= 0):
            raise ValueError("Invalid trade")
        amt = shares * px
        if action == "买入":
            if amt + f > self.s["cash"] + 1e-7:
                raise ValueError("Buy exceeds available cash")
            self.s["cash"] = max(0.0, self.s["cash"] - amt - f)
            p = self.s["positions"].get(inst)
            if p:
                p["cost"] += amt + f
                p["shares"] += shares
            else:
                self.s["positions"][inst] = dict(shares=float(shares), cost=amt + f, buy_date=str(d.date()),
                                                 last_px=px, factor=None)
        else:
            p = self.s["positions"][inst]
            if p.get("buy_date", "") >= str(d.date()):
                raise ValueError("T+1: same-day stock sale is forbidden")
            if shares > p["shares"] + 1e-7:
                raise ValueError("Sell exceeds holding")
            self.s["cash"] += amt - f
            if shares >= p["shares"] - 1e-7:
                del self.s["positions"][inst]
            else:
                p["cost"] *= 1 - shares / p["shares"]
                p["shares"] -= shares
        self.trades.append(dict(日期=str(d.date()), 操作=action, 代码=code_of(inst), 名称=name, 股数=shares,
                                成交价=round(px, 4), 金额=round(amt, 2), 费用=round(f, 2), 备注=note))


def rejected(acc, d, action, j, reason, names):
    acc.trades.append(dict(日期=str(d.date()), 操作=action + "失败", 代码=code_of(j), 名称=names.get(j, ""),
                           股数=0, 成交价="", 金额="", 费用="", 备注=reason))


def allowed(q, side, at):
    px = q.get("raw_" + at, float("nan"))
    limit = ("up_" if side == "buy" else "dn_") + ("open" if at == "open" else "lim")
    return (math.isfinite(px) and px > 0 and not q.get("susp", True) and not q.get(limit, True)
            and (side != "buy" or q.get("buy_ok", False)))


def dropout_orders(score, hold, topk, n_drop, eligible=None):
    s = score.dropna().sort_values(ascending=False, kind="stable")
    extra = [j for j in hold if j not in s.index]
    last = sorted([j for j in hold if j in s.index], key=lambda j: s[j], reverse=True)
    cand = [j for j in s.index if j not in set(last) and (eligible is None or eligible.get(j, False))]
    today = cand[:max(n_drop + topk - len(last), 0)]
    comb = sorted(last + today, key=lambda j: s[j], reverse=True)
    worst = set(comb[-n_drop:]) if n_drop > 0 else set()
    sell = [j for j in last if j in worst]
    buy = today[:max(len(sell) + topk - len(last), 0)]
    backup = [j for j in cand if j not in set(buy)][:max(5, 2 * len(buy))]
    return extra + sell, buy, backup, s


def execute_dropout(acc, mk, d, pend, names, cfg):
    at = cfg.get("exec_at", "close")
    oc, mc, slip = cfg["open_cost"], cfg["min_cost"], cfg.get("slip", 0.0)
    pos = acc.s["positions"]
    for j in pend["sell"]:
        if j not in pos:
            continue
        q = mk.quote(d, j)
        if not allowed(q, "sell", at):
            rejected(acc, d, "卖出", j, "跌停/停牌/缺价，继续持有", names)
            continue
        px, sh = q["raw_" + at] * (1 - slip), pos[j]["shares"]
        acc.trade(d, "卖出", j, sh, px, sell_fee(sh * px, cfg, d), "清单卖出", names.get(j, ""))

    expo = pend.get("expo_next", 1.0)
    target = acc.total() * (1 - expo)
    # ETF target is funded from existing cash first. Sell only the shortfall.
    if expo < 1 and acc.s["etf_value"] < 0.9 * target:
        need = max(target - acc.s["etf_value"], 0)
        short = max(need * (1 + oc) + mc - acc.s["cash"], 0)
        frac = min(short / max(acc.stock_value(), 1), 1)
        plan = {}
        for j in pos:
            q = mk.quote(d, j)
            if allowed(q, "sell", at):
                sh = math.floor(pos[j]["shares"] * frac / lot_of(j)) * lot_of(j)
                plan[j] = [sh, q["raw_" + at] * (1 - slip)]
        def proceeds():
            return sum(sh * px - sell_fee(sh * px, cfg, d) for sh, px in plan.values() if sh > 0)
        while proceeds() < short:
            left = [(pos[j]["shares"] - sh) * px for j, (sh, px) in plan.items()]
            if not left or max(left) <= 1e-7:
                break
            j = list(plan)[left.index(max(left))]
            plan[j][0] = min(plan[j][0] + lot_of(j), pos[j]["shares"])
        for j, (sh, px) in plan.items():
            if sh <= 0:
                continue
            if pos[j]["shares"] - sh < lot_of(j):
                sh = pos[j]["shares"]
            acc.trade(d, "卖出", j, sh, px, sell_fee(sh * px, cfg, d), "R3 换 ETF", names.get(j, ""))
        amt = min(need, max((acc.s["cash"] - mc) / (1 + oc), 0))
        if amt > 100:
            f = fee(amt, oc, mc)
            acc.s["cash"] -= amt + f
            acc.s["etf_value"] += amt
            acc.trades.append(dict(日期=str(d.date()), 操作="买入", 代码=ETF_CODE, 名称="指数ETF近似", 股数="",
                                   成交价="", 金额=round(amt, 2), 费用=round(f, 2), 备注="R3"))
    elif expo >= 1 and acc.s["etf_value"] > 0:
        v = acc.s["etf_value"]
        f = fee(v, oc, mc)  # ETF no stock stamp duty
        acc.s["cash"] += v - f
        acc.s["etf_value"] = 0.0
        acc.trades.append(dict(日期=str(d.date()), 操作="卖出", 代码=ETF_CODE, 名称="指数ETF近似", 股数="",
                               成交价="", 金额=round(v, 2), 费用=round(f, 2), 备注="解除R3"))

    buys = [b["inst"] if isinstance(b, dict) else b for b in pend["buy"]]
    backup = [b["inst"] if isinstance(b, dict) else b for b in pend["backup"]]
    free = max(cfg["topk"] - len(pos), 0)
    slots = min(len(buys), free)
    sleeve = acc.total() * expo
    budget = min(max(acc.s["cash"] - (1 - cfg["risk_degree"]) * sleeve, 0),
                 max(sleeve * cfg["risk_degree"] - acc.stock_value(), 0))
    per = min(budget / max(slots, 1), sleeve * cfg["risk_degree"] / cfg["topk"] * 1.5)
    used = set(pos)
    for primary in buys[:slots]:
        queue = [primary]
        while queue:
            j = queue.pop(0)
            q = mk.quote(d, j)
            px = q.get("raw_" + at, float("nan")) * (1 + slip)
            sh = affordable(j, min(per, acc.s["cash"]), px, oc, mc) if allowed(q, "buy", at) else 0
            if j not in used and sh > 0:
                acc.trade(d, "买入", j, sh, px, fee(sh * px, oc, mc),
                          "清单买入" if j == primary else "替补 " + code_of(primary), names.get(j, ""))
                pos[j]["factor"] = q["factor"]
                used.add(j)
                break
            rejected(acc, d, "买入", j, "不符合交易状态/封板/资金不足", names)
            if backup:
                queue.append(backup.pop(0))


def concentrated_orders(score, hold, k, keep, on=True, eligible=None):
    s = score.dropna().sort_values(ascending=False, kind="stable")
    rank = {j: i + 1 for i, j in enumerate(s.index)}
    sell = [j for j in hold if not on or rank.get(j, float("inf")) > keep]
    n = max(k - len([j for j in hold if j not in sell]), 0) if on else 0
    cand = [j for j in s.index if j not in hold and (eligible is None or eligible.get(j, False))]
    return dict(sell=sell, buy=[dict(inst=j) for j in cand[:n]],
                backup=[dict(inst=j) for j in cand[n:n + 5]])


def execute_conc(acc, mk, d, pend, names, cfg):
    at = cfg.get("exec_at", "open")
    oc, mc, slip = cfg["open_cost"], cfg["min_cost"], cfg.get("slip", 0.001)
    pos = acc.s["positions"]
    for j in pend["sell"]:
        if j not in pos:
            continue
        q = mk.quote(d, j)
        if not allowed(q, "sell", at):
            rejected(acc, d, "卖出", j, "跌停/停牌/缺价，继续持有", names)
            continue
        px, sh = q["raw_" + at] * (1 - slip), pos[j]["shares"]
        acc.trade(d, "卖出", j, sh, px, sell_fee(sh * px, cfg, d), "清单卖出", names.get(j, ""))
    free = max(cfg["k"] - len(pos), 0)  # failed sales still occupy a slot
    backup = list(pend["backup"])
    slot = acc.total() / cfg["k"]  # marked to execution time, not today's close
    for primary in pend["buy"][:free]:
        queue = [primary]
        while queue:
            j = queue.pop(0)["inst"]
            q = mk.quote(d, j)
            px = q.get("raw_" + at, float("nan")) * (1 + slip)
            sh = affordable(j, min(slot, acc.s["cash"]), px, oc, mc) if allowed(q, "buy", at) else 0
            if j not in pos and sh > 0:
                acc.trade(d, "买入", j, sh, px, fee(sh * px, oc, mc), "清单/替补买入", names.get(j, ""))
                pos[j]["factor"] = q["factor"]
                break
            rejected(acc, d, "买入", j, "不符合交易状态/封板/资金不足", names)
            if backup:
                queue.append(backup.pop(0))


def execute_fb(acc, mk, d, pend, names):
    pos = acc.s["positions"]
    held = [j for j, p in pos.items() if p["buy_date"] < str(d.date())]
    opts = []
    if pend:
        for c in pend.get("cands", []):
            j, q = c["inst"], mk.quote(d, c["inst"])
            o = q.get("raw_open", float("nan"))
            # Check the percentage, not rounded displayed price boundaries.
            gap = o / c["ref"] - 1
            if j not in pos and allowed(q, "buy", "open") and -0.07 - 1e-9 <= gap <= -0.02 + 1e-9:
                opts.append((gap, j, o, q))
        for gap, j, o, q in sorted(opts)[:max(pend["K"] - len(pos), 0)]:
            x = mk.execution_quote(d, j)
            px = x.get("price", float("nan")) * (1 + FB_SLIP)
            # First 5-minute close is observed at 09:35. Submit then at an
            # open+0.5% limit; never assume the 09:25 auction can be revisited.
            if not x.get("valid", False) or not (px > 0 and px <= o * 1.005):
                rejected(acc, d, "买入", j, "09:35价格缺失或超过开盘价+0.5%限价", names)
                continue
            sh = affordable(j, min(acc.total() / pend["K"], acc.s["cash"]), px, FB_BUY_COMM, 5)
            if not sh:
                rejected(acc, d, "买入", j, "资金不足", names)
                continue
            acc.trade(d, "买入", j, sh, px, fee(sh * px, FB_BUY_COMM),
                      f"09:35首5分钟收盘价近似，低开 {gap:+.2%}", names.get(j, ""))
            pos[j]["factor"] = q["factor"]
    for j in held:
        q = mk.quote(d, j)
        if not allowed(q, "sell", "close"):
            rejected(acc, d, "卖出", j, "跌停/停牌/缺价，顺延", names)
            continue
        px, sh = q["raw_close"] * (1 - FB_SLIP), pos[j]["shares"]
        acc.trade(d, "卖出", j, sh, px, fee(sh * px, FB_SELL_COMM, stamp=stamp_rate(d)),
                  "买入次日收盘卖出", names.get(j, ""))


def advance_r3(state, reference_return, bench_return, d, cfg):
    date = str(d.date())
    if state.get("last_date") == date:
        return state
    if state.get("last_date", "") > date:
        raise ValueError("R3 dates must advance monotonically")
    nav = state.get("nav", 1.0) * (1 + reference_return) / (1 + bench_return)
    peak = max(state.get("peak", 1.0), nav)
    dd = nav / peak - 1
    expo = state.get("exposure", 1.0)
    if expo == 1.0 and dd < cfg["r3_in"]:
        expo = cfg["r3_exposure"]
    elif expo < 1.0 and dd > cfg["r3_out"]:
        expo = 1.0
    state.update(nav=nav, peak=peak, dd=dd, exposure=expo, last_date=date)
    return state


def signal_score(score, day):
    """Pool/ST/delisting membership known on the signal day; no future masks."""
    ok = day.get("in_pool", True)
    if not hasattr(ok, "index"):
        ok = day["buy_ok"].notna()
    for c in ("is_st", "is_delisted"):
        if c in day:
            ok = ok & ~day[c].fillna(False)
    ok = ok & ~day.index.to_series().str.startswith("BJ")
    return score.reindex(day.index).where(ok).dropna()
