"""候选实验共用：R3 回撤止损、双账户混合、打分平均"""
import numpy as np
import pandas as pd


def r3(bt, res, dd_in=-0.08, dd_out=-0.04):
    """超额回撤 < -8% 时一半仓位换成中证1000 ETF，回到 -4% 以内恢复（只用 T-1 信息）"""
    cum = (1 + res["excess"]).cumprod()
    dd = (cum / cum.cummax() - 1).shift(1).fillna(0)
    st, e = 1.0, []
    for v in dd.values:
        st = 0.5 if (st == 1.0 and v < dd_in) else (1.0 if (st == 0.5 and v > dd_out) else st)
        e.append(st)
    return bt.overlay(res, pd.Series(e, index=res.index))


def mix(a, b, w=0.5):
    m = a.copy()
    for c in ["return", "turnover", "cost"]:
        m[c] = w * a[c] + (1 - w) * b[c]
    m["excess"] = m["return"] - m["bench"]
    return m


def rank_blend(preds, weights):
    """每个模型先转当日百分位排名，再加权平均"""
    ps, ws = [], []
    for d, w in zip(preds, weights):
        s = d.set_index(["datetime", "instrument"])["score"]
        r = s.groupby(level=0).rank(pct=True)
        ps.append(r * w)
        ws.append(r.notna() * w)
    num = pd.concat(ps, axis=1).sum(axis=1)
    den = pd.concat(ws, axis=1).sum(axis=1)   # 某模型缺分时用剩下的模型（与 candidates.py 的 mean 一致）
    return (num / den.replace(0, np.nan)).rename("score").dropna().reset_index()


def ir(x, n_year=238):
    return x.mean() / (x.std() + 1e-12) * np.sqrt(n_year)
