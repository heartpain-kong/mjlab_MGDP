#!/usr/bin/env bash
# 曲线终端启动脚本 —— 把 logs/rsl_rl 里的 TensorBoard 曲线当 K 线看。
#
#   bash kline.sh                        启动，默认 http://127.0.0.1:8766
#   bash kline.sh --port 9000
#   bash kline.sh --no-browser           不开浏览器（跑在服务器上时用）
#   bash kline.sh --logdir /data/archive/logs/rsl_rl
#
# 需要当前 Python 环境里有 tensorboard（训练环境本来就有）。
# 只读：不写 event 文件，只往 logs/kline_cache/ 里放解析缓存。

set -u

PROJ="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python}"
cd "$PROJ" || exit 1

exec "$PY" -m src.kline "$@"
