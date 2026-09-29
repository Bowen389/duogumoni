# 用法：source env.sh   （所有脚本都在仓库根目录下运行）
# 数据路径可以用环境变量覆盖，默认放在 ~/.qlib 下
export RA_PROVIDER=${RA_PROVIDER:-$HOME/.qlib/qlib_data/cn_data}   # Qlib 日线数据
export RA_DATA=${RA_DATA:-$HOME/.qlib/retail_alpha_data}            # 面板 / 因子（可重建）
export RA_EXTRA=${RA_EXTRA:-$HOME/.qlib/retail_extra}               # 行业 / 换手率 / 龙虎榜
export RA_VENV=${RA_VENV:-$(pwd)/.venv}
export PYTHONPATH=$(pwd)${PYTHONPATH:+:$PYTHONPATH}
if [ -f "$RA_VENV/bin/activate" ]; then . "$RA_VENV/bin/activate"; fi
