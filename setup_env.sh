#!/usr/bin/env bash
# 一键准备环境与数据（在仓库根目录运行）
#   bash setup_env.sh              # Python 环境 + Qlib 数据 + 行业 + 面板 + 散户因子 + 长期犯错因子（三个策略所需，约 6 分钟）
#   bash setup_env.sh --refresh    # 每日收盘后：重新下载最新 Qlib 数据并重建面板/因子（实盘用）
#   bash setup_env.sh --alpha158   # 另外构建 Alpha158 因子（研究实验需要，约 20 分钟）
set -e
cd "$(dirname "$0")"
REFRESH=0; A158=0
for a in "$@"; do
  case $a in --refresh) REFRESH=1 ;; --alpha158) A158=1 ;; esac
done
source env.sh

# 1) Python 3.11 虚拟环境（pyqlib 不支持 3.13）
if [ ! -x "$RA_VENV/bin/python" ]; then
  if command -v python3.11 >/dev/null; then
    python3.11 -m venv "$RA_VENV"; . "$RA_VENV/bin/activate"; pip install -q -U pip; pip install -q -r requirements.txt
  else
    command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH=$HOME/.local/bin:$PATH; }
    uv python install 3.11 -q; uv venv -q -p 3.11 "$RA_VENV"; . "$RA_VENV/bin/activate"; uv pip install -q -r requirements.txt
  fi
fi
source env.sh

# 2) Qlib 日线数据（社区维护，每个交易日更新；约 570MB）
if [ $REFRESH = 1 ] || [ ! -d "$RA_PROVIDER/features" ]; then
  mkdir -p "$RA_PROVIDER"
  wget -q https://github.com/chenditc/investment_data/releases/latest/download/qlib_bin.tar.gz -O /tmp/qlib_bin.tar.gz
  tar -xzf /tmp/qlib_bin.tar.gz -C "$RA_PROVIDER" --strip-components=1 && rm -f /tmp/qlib_bin.tar.gz
fi
echo "Qlib 数据最新交易日：$(tail -1 "$RA_PROVIDER/calendars/day.txt")"

# 3) 行业分类 + 面板 + 散户因子
mkdir -p "$RA_EXTRA"
if [ $REFRESH = 1 ] || [ ! -f "$RA_EXTRA/industry.parquet" ]; then python data_fetch/fetch_industry.py; fi
if [ $REFRESH = 1 ] || [ ! -d "$RA_DATA/panel" ]; then python -m engine.panel; fi
if [ $REFRESH = 1 ] || [ ! -d "$RA_DATA/feat_retail" ]; then python -m engine.features --set retail; fi
if [ $REFRESH = 1 ] || [ ! -d "$RA_DATA/feat_behavior" ]; then python -m engine.features --set behavior; fi   # AE_top20 需要
if [ $A158 = 1 ]; then python -m engine.features --set alpha158 --chunk 50; fi
echo SETUP_DONE
