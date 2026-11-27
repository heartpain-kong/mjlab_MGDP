#!/bin/bash
# 一键部署 mjlab_MGDP：创建 conda 环境 → 安装依赖 → 自检 →（可选）启动工作台
#
# 用法：
#   bash scripts/deploy.sh                 # 安装 + 自检
#   bash scripts/deploy.sh --start         # 安装后启动工作台（8088）
#   bash scripts/deploy.sh --skip-system-deps   # 无 sudo 时跳过系统依赖
#
# 其它参数会原样传给 configure_lgdog.sh（如 --skip-system-deps / --torch-index / --ui-port）。
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$(pwd)

START=0
ARGS=()
for a in "$@"; do
  case "$a" in
    --start) START=1 ;;
    *) ARGS+=("$a") ;;
  esac
done

echo "== [1/3] 安装环境（configure_lgdog.sh）"
bash configure_lgdog.sh "${ARGS[@]:-}"

echo "== [2/3] 自检"
ENV_PY=""
if [ -x /root/gpufree-data/conda/envs/lgdog/bin/python ]; then
  ENV_PY=/root/gpufree-data/conda/envs/lgdog/bin/python
elif command -v conda >/dev/null 2>&1; then
  ENV_PY=$(conda run -n lgdog which python 2>/dev/null || true)
fi
PY=${ENV_PY:-python}
"$PY" scripts/verify_install.py || { echo "[warn] 自检未通过，请查看上面的输出"; }

echo "== [3/3] 完成"
echo "仓库: $ROOT"
echo "Python: $PY"
echo ""
echo "下一步："
echo "  bash scripts/deploy.sh --start --ui-port 8088   # 启动工作台"
echo "  bash run_lgdog.sh server start --ui-port 8088   # 或手动启动"
echo "  PYTHONPATH=$ROOT $PY -m pytest -q tests/test_mgdp_wall_training.py"
if [ "$START" = "1" ]; then
  echo "== 启动工作台"
  export LGDOG_PYTHON="$PY"
  bash run_lgdog.sh server start --ui-port "${UI_PORT:-8088}"
fi
