"""
两个实盘策略的全部参数（改这里即可，live.py / backtest.py 都从这里读）

共同部分（策略 A）：
  股票池    中证1000 成分股（按历史成分，防幸存者偏差）
  因子      retail_factors.py 中 16 个"散户行为"因子
  模型      LightGBM（robust 超参），标签 = T+1 收盘买入、持有 1 日的收益（剔除次日买不进的样本）
  中性化    每日把打分转成排名后减去所属证监会行业均值 → 行业中性
  成交      T 日收盘后出信号，T+1 日尾盘集合竞价成交
  风控 R3   策略超额（相对中证1000）从高点回撤超过 8% → 一半仓位换成中证1000 ETF；回撤收窄到 4% 以内恢复
"""

COMMON = dict(
    pool="csi1000",
    bench="SH000852",
    feature_set="retail",
    model_file="models/retail_lgb.txt",
    open_cost=0.0005,     # 买入：佣金+过户等（万5，含最低 5 元）
    close_cost=0.0010,    # 卖出：佣金 + 印花税 0.05%（万10）
    min_cost=5.0,
    risk_degree=0.95,     # 最多用 95% 资金买股票
    r3_in=-0.08,          # 超额回撤低于 -8% → 半仓
    r3_out=-0.04,         # 回到 -4% 以内 → 恢复满仓
    r3_exposure=0.5,
    hedge_etf="512100",   # 中证1000ETF（南方）；也可用 159845 / 560010 等
)

STRATEGIES = {
    # 主策略：分散、稳健。2023-26 超额 9.3% / IR 1.24；样本外 2020-22 超额 16.4% / IR 2.01
    "A_top50": dict(COMMON, topk=50, n_drop=2, every=1, account=1_000_000),
    # 进取版：更集中、收益弹性更大，但波动与回撤更大。2023-26 超额 12.1% / IR 1.22；样本外 15.6% / IR 1.49
    "A_top20": dict(COMMON, topk=20, n_drop=1, every=1, account=1_000_000),
}
