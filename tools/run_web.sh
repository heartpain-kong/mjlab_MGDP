#!/usr/bin/env bash
# 一键起 mjlab 总览：自动带起训练工作台（:8765）和曲线终端（:8766）。
#
#   bash run_web.sh                    # http://127.0.0.1:8770
#   bash run_web.sh --no-autostart     # 只起总览，服务按钮自己点
#   bash run_web.sh --host 0.0.0.0     # 让同一局域网里的手机/平板也能看
#
# Python 解释器沿用 run_lgdog.sh 的规则（LGDOG_PYTHON > .venv > /root/gpufree-data/lgdog-venv）。
set -Eeuo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${ROOT_DIR}/run_lgdog.sh" web "$@"
