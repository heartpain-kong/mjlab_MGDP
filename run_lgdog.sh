#!/usr/bin/env bash
set -Eeuo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${ROOT_DIR}/server_env.sh"
lgdog_server_env "${ROOT_DIR}"
if [[ -n "${LGDOG_PYTHON:-}" ]]; then
  VENV_PYTHON="${LGDOG_PYTHON}"
elif [[ -f "${ROOT_DIR}/logs/server/python_path.txt" ]]; then
  read -r VENV_PYTHON < "${ROOT_DIR}/logs/server/python_path.txt"
elif [[ -x "${ROOT_DIR}/.venv/bin/python" ]]; then
  VENV_PYTHON="${ROOT_DIR}/.venv/bin/python"
else
  VENV_PYTHON="/root/gpufree-data/lgdog-venv/bin/python"
fi
[[ -x "${VENV_PYTHON}" ]] || { echo "未找到 ${VENV_PYTHON}，请先运行 bash install_ubuntu.sh" >&2; exit 1; }
cd -- "${ROOT_DIR}"
exec "${VENV_PYTHON}" "${ROOT_DIR}/lgdog.py" "$@"
