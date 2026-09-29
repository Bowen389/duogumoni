#!/usr/bin/env bash
# 第二批策略（C_top50 / K_top3）额外需要的数据：东方财富股东户数 → 因子 feat_em
#   bash strategies2/setup_data.sh            # 第一次：抓中证1000 历史成分股的全部股东户数（约 15 分钟）
#   bash strategies2/setup_data.sh --refresh  # 每日：增量拉取最近公告（几秒）并重建因子（约 1 分钟）
# 先运行过 bash setup_env.sh（需要其中的面板数据）
set -e
cd "$(dirname "$0")/.."
source env.sh
if [ "$1" = "--refresh" ] && [ -f "$RA_EXTRA/em_holders.parquet" ]; then
  python data_fetch/fetch_em.py update
else
  python data_fetch/fetch_em.py holders
fi
python -m engine.em_features
echo DATA2_DONE
