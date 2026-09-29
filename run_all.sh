#!/usr/bin/env bash
# 一键复现 v2 全部实验（首次约 2~3 小时，大部分时间在下载换手率/龙虎榜）
# 内存：2GB 可跑（引擎按块读写磁盘）；磁盘：约 4GB
set -e
source env.sh

# 0) 数据
bash setup_env.sh --alpha158
source env.sh   # 激活刚建好的虚拟环境
python data_fetch/fetch_baostock.py --out "$RA_EXTRA/turn" --workers 4     # 换手率/ST（约 45 分钟）
python data_fetch/fetch_lhb.py --out "$RA_EXTRA"                           # 龙虎榜（约 1.5 小时，可断点续传）
python data_fetch/fetch_industry.py                                        # 行业分类

# 1) 面板 + 因子（分块落盘）
python -m engine.panel
python -m engine.features --set retail
python -m engine.features --set alpha158 --chunk 50
python -m engine.extra_features

# 2) 模型
python -m engine.model --sets retail --horizon 1 --tag retail_h1
python -m engine.model --sets retail --horizon 5 --tag retail_h5
python -m engine.model --sets alpha158 --horizon 1 --sample 3 --tag a158_h1
python -m engine.model --sets alpha158,retail --horizon 1 --sample 3 --tag both_h1
python -m engine.model --sets retail,extra --horizon 1 --tag retail_extra_lhb_tl
python -m engine.model --sets retail --horizon 1 --rolling 63 --fixed_rounds 100 --tag retail_h1_roll
python -m engine.model --sets retail --horizon 1 --model xgb --tag retail_h1_xgb
python -m engine.model --sets retail --horizon 1 --model lgb_rank --tag retail_h1_rank

# 3) 实验
python experiments/p1_turnover.py
python experiments/compare.py --preds a158_h1,retail_h1,both_h1 --out output/p1/features
python experiments/p1_crowding.py --pred retail_h1
python experiments/p2_factor_ic.py
python experiments/neutralize.py --pred retail_h1 --mode ind
python experiments/neutralize.py --pred retail_h1 --mode ind+mv
python experiments/compare.py --preds retail_h1,retail_h1_roll,retail_h1_xgb,retail_h1_rank,retail_h1_neu_ind,retail_h1_neu_ind_mv --out output/p3/models
python experiments/final.py
python experiments/candidates.py

# 4) 稳健性：样本外 2020-2022、权重扫描、持股数量
python -m engine.model --sets retail --horizon 1 --train 2015-01-01 2018-12-31 --valid 2019-01-01 2019-12-31 --test 2020-01-01 2022-12-31 --tag retail_h1_oos
python -m engine.model --sets alpha158 --horizon 1 --sample 3 --train 2015-01-01 2018-12-31 --valid 2019-01-01 2019-12-31 --test 2020-01-01 2022-12-31 --tag a158_h1_oos
python experiments/neutralize.py --pred a158_h1 --mode ind
python experiments/neutralize.py --pred retail_h1_oos --mode ind
python experiments/neutralize.py --pred a158_h1_oos --mode ind
python experiments/h_robust.py weights
python experiments/h_robust.py oos
python experiments/topk.py

# 5) 最终策略
python strategies/backtest.py
python strategies/live.py train
