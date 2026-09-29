#!/usr/bin/env bash
# 提高收益的方案所需的模型（研究期 + 样本外各一个），全部在中证1000内
set -e
source env.sh
OOS="--train 2015-01-01 2018-12-31 --valid 2019-01-01 2019-12-31 --test 2020-01-01 2022-12-31"
# 方案6：16 个散户因子 + 12 个"长期犯错"行为因子
python -m engine.model --sets retail,behavior --horizon 1 --tag rb_h1
python -m engine.model --sets retail,behavior --horizon 1 $OOS --tag rb_h1_oos
# 方案3b：开盘成交标签（T+1 开盘买入、T+2 开盘卖出）
RA_LABEL=reto1 RA_NB=nbo python -m engine.model --sets retail --horizon 1 --tag retail_h1o
RA_LABEL=reto1 RA_NB=nbo python -m engine.model --sets retail --horizon 1 $OOS --tag retail_h1o_oos
for t in rb_h1 rb_h1_oos retail_h1o retail_h1o_oos; do python experiments/neutralize.py --pred $t --mode ind; done
echo TRAIN_DONE
