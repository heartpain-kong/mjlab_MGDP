#!/usr/bin/env bash
set -Eeuo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${LGDOG_PYTHON:-}" ]]; then
  VENV_PYTHON="${LGDOG_PYTHON}"
elif [[ -x "${ROOT_DIR}/.venv/bin/python" ]]; then
  VENV_PYTHON="${ROOT_DIR}/.venv/bin/python"
else
  VENV_PYTHON="/root/gpufree-data/lgdog-venv/bin/python"
fi
[[ -x "${VENV_PYTHON}" ]] || { echo "未找到 ${VENV_PYTHON}，请先运行 bash install_ubuntu.sh" >&2; exit 1; }
exec "${VENV_PYTHON}" "${ROOT_DIR}/lgdog.py" "$@"
