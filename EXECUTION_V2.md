# 执行口径修复（v2）

本次修复统一了回测与每日模拟盘的现金、股数、替补、交易状态、持仓上限和R3执行。
README 中原有收益表和 `results/` 中原有结果属于旧口径，**尚未用 v2 全历史重跑**。
2023–2026 与反复用于比较的2020–2022均属于研究验证数据；参数冻结后的新增数据才用于前向检验。

## 已修复

- `engine/execution.py` 是 A/C、K、FB 的共同成交实现；买入重新计算实际资金和价格，最低佣金与印花税分别计算。
- 卖出失败继续占持仓名额；新买入前检查实际空位，K 最多3只、FB最多2只。
- 新买入排除历史ST、退市、北交所和上市不足30个自然日股票。ST状态按交易日期读取，不把当前ST/名称倒灌到历史。状态未知不新买；旧仓卖出仍按报价和保守涨跌停判断处理。
- A/C 回测与每日账户使用相同的替补数量和顺序。清单股数只作参考，执行时按真实余额和成交报价重算。
- R3 使用连续未控仓参考账户相对基准的净值 `参考累计净值 / 指数累计净值`，保存历史高点与状态。低于高点−8%降低暴露，恢复到−4%以内解除；当日收盘更新决定下一交易日操作。
- R3 在执行端先用现金购买ETF，仅卖出资金缺口；恢复时实际卖出ETF。回测不再用日收益线性混合代替买卖。
- K/FB 开盘阶段只用开盘价估值和先做除权折算，不使用当日收盘价计算开盘资金。
- 股票买入当日不能卖出；缺价、停牌、封板与现金不足不能伪造成交。
- C策略临时调用A工具后恢复模型配置，不污染同一进程后续A策略。
- 状态最后原子替换，重复处理同日交易记录会覆盖该日记录，不重复追加。
- 模拟盘工作流在状态/分钟数据缺失时明确记录；结果提交连续三次失败会使任务失败。

## FB 买入时点

9:25 得到开盘价，只用于筛选低开2%–7%和排序。成交回放改用独立记录的**09:35首5分钟收盘价**加千1滑点，且不得超过开盘价+0.5%的限价。
缺少该日该股的报价、时间不为09:35、成交量为0或价格超过限价，记录未成交，不回退到开盘价。
该价格是分钟级的成交估计，并不保证限价单能成交；若需要精确09:30或逐笔排队成交，需要接入对应的报价与成交数据。

每日流程会自动尝试从baostock获取候选首5分钟数据。数据源不可用时其他账户照常运行，FB继续管理已有持仓，缺价候选不买。
`experiments/fb_dip.py` 的开盘价事件研究保留为历史研究；显式设置 `RA_ALLOW_LEGACY_OPEN_RESEARCH=1` 才能重新构建该旧表。
新的可复现成交回放入口是 `strategies2/fb_backtest.py`。

## 现有账户迁移

首次处理新的交易日时在 `state.json` 写入 `execution_version=2` 和 `execution_start`，保留现有现金、持仓、净值及交易历史。
从切换前一天的账户快照建立连续参考账户；已有R3半仓状态会保留，并记录迁移回撤基线。该参考账户不冒充修复后的历史回测。
信号日参考状态、下一日参考清单、高点与相对净值都随账户保存，重复运行不会重新初始化。

## 重跑历史回测

完整历史数据不包含在仓库中，需要下载行情、历史ST状态与相应FB分钟数据。
没有足够的历史状态时A/C/K回测会报错并提示回填，避免把未确认的结果写成新收益表。

```bash
source env.sh
python data_fetch/fetch_status.py --start 2014-06-01 --end 2026-10-09
python -m engine.panel
python -m engine.features --set retail
python -m engine.features --set behavior
python -m engine.em_features
python experiments/senti.py build

python strategies/backtest.py --account 100000
python strategies/backtest.py --account 1000000
python strategies2/backtest.py --account 100000
python strategies2/backtest.py --account 1000000

python data_fetch/fetch_execution.py --start 2020-01-01 --end 2026-10-09
python strategies2/fb_backtest.py --start 2020-01-02 --end 2026-10-09 --account 100000
```

注意：后一次回测会覆盖同名结果；比较账户金额时请分别保存结果目录。旧预测文件可以用于比较执行口径，更新训练标签后应另外重训模型，不能声称已完成模型重新验证。

独立使用A/C信号工具时，可传入已更新至信号日的连续账户状态：

```bash
python strategies/live.py signal --strategy A_top50 --r3_state paper/accounts/A_top50/state.json
python strategies2/live.py signal --strategy C_top50 --r3_state paper/accounts/C_top50/state.json
```

不传状态时从固定的 `--r3_start`（默认2023-01-01）回放，不再按近365天滚动重置；需要该起点以来的完整历史ST数据。
该参数应在开始运行时固定；更换模型、参考起点或资金规模会改变参考组合，须作为新的验证版本记录。

## 仍然保留的近似

- 中证1000ETF使用指数收益近似，没有ETF价格、跟踪误差、实际份额取整或真实冲击成本。
- 除权除息仍用复权因子折算股数，属于总收益近似；不会生成实际股息税、红利现金到账或精确可卖股份。变化小于0.3%的复权噪声沿用旧处理。
- A/C尾盘及K开盘的成交报价仍是日线近似。买卖方向的封板限制已经分开，但不建模排队或实际成交量容量。
- 行业中性仅作用于信号，未强制最终行业权重中性；当前行业快照不等于完整的历史行业分类。
- FB情绪统计实际覆盖历史中证1000相关证券样本，不是A股全市场；ST/未知状态记录不参与统计。

## 验证

```bash
python -m pytest -q
python -m compileall -q engine paper strategies strategies2 data_fetch experiments
bash -n setup_env.sh paper/run_daily.sh
```

合成行情回归测试覆盖现金/份额、失败卖单占位、替补、ST状态、09:35报价时间与限价、T+1、除权处理、连续R3、账户迁移、重启和回测/每日执行一致性。
自动测试工作流独立运行这些检查；它不代表已下载全历史数据或已完成收益验证。
