#!/usr/bin/env bash
# 下载社区维护的 A 股日线数据（Qlib 格式，约 570MB，解压后约 850MB，每日更新）
# 来源：https://github.com/chenditc/investment_data （Qlib 官方 README 推荐）
set -e
mkdir -p ~/.qlib/qlib_data/cn_data
wget https://github.com/chenditc/investment_data/releases/latest/download/qlib_bin.tar.gz
tar -zxvf qlib_bin.tar.gz -C ~/.qlib/qlib_data/cn_data --strip-components=1
rm -f qlib_bin.tar.gz
echo "数据已就绪：~/.qlib/qlib_data/cn_data"
