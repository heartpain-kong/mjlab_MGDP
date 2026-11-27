#!/bin/bash
# 回放开源策略（model_12400）。要求当前代码处于 policy-v1 标签。
# 用法: bash release/play_policy.sh <策略目录>   策略目录里应有 model_12400.pt 与 perception_3cca*.pt
set -e
DIR=${1:?用法: bash release/play_policy.sh <策略目录>}
cd "$(dirname "$0")/.."
PY=${PY:-python}

if ! git describe --tags --exact-match 2>/dev/null | grep -q '^policy-v1$'; then
  echo "[warn] 当前代码不是 policy-v1，这份策略可能与代码不匹配"
  echo "       请先执行: git checkout policy-v1"
fi

$PY scripts/play.py LgDog1-Race-MGDP-WallTraining-D435i20 \
  --checkpoint-file "$DIR/model_12400.pt" \
  --perception-checkpoint "$DIR"/perception_*.pt
