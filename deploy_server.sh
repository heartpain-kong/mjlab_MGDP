#!/usr/bin/env bash
# Install, verify CUDA/Warp and start local services. Never starts full training.
set -Eeuo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "${ROOT_DIR}"
source "${ROOT_DIR}/server_env.sh"
lgdog_server_env "${ROOT_DIR}"
ENV_DIR="${LGDOG_ENV_DIR:-}"
if [[ -z "${ENV_DIR}" ]]; then
  if [[ -d /root/gpufree-data && -w /root/gpufree-data ]]; then ENV_DIR=/root/gpufree-data/lgdog-venv; else ENV_DIR="${ROOT_DIR}/.venv"; fi
fi
INSTALL=1 START=1 SMOKE=0
UI_PORT=8765 KLINE_PORT=8766 PORTAL_PORT=8770 TB_PORT=6006
INSTALL_ARGS=()
usage() {
  cat <<'EOF'
Usage: bash deploy_server.sh [options]
  --env-dir PATH         Python venv (GPUFree default: /root/gpufree-data/lgdog-venv)
  --python PATH          Python 3.10-3.13 used when creating the venv
  --torch-index URL      PyTorch wheel index; default CUDA 12.8
  --skip-system-deps    skip apt packages on preconfigured images
  --skip-install        reuse installed dependencies; still check CUDA/Warp
  --smoke               explicitly run 64 environments / 2 training iterations
  --no-start            install and verify only
  --ui-port N --kline-port N --portal-port N --tb-port N
  -h, --help
Default: install + CUDA allocation check + loopback web services; no training.
EOF
}
while (($#)); do
  case "$1" in
    --env-dir) [[ $# -ge 2 ]] || exit 2; ENV_DIR="$2"; shift 2 ;;
    --python|--torch-index) [[ $# -ge 2 ]] || exit 2; INSTALL_ARGS+=("$1" "$2"); shift 2 ;;
    --skip-system-deps) INSTALL_ARGS+=("$1"); shift ;;
    --skip-install) INSTALL=0; shift ;;
    --no-start) START=0; shift ;;
    --smoke) SMOKE=1; shift ;;
    --ui-port|--kline-port|--portal-port|--tb-port)
      [[ $# -ge 2 && "$2" =~ ^[0-9]+$ ]] || { echo 'port requires an integer' >&2; exit 2; }
      case "$1" in --ui-port) UI_PORT="$2";; --kline-port) KLINE_PORT="$2";; --portal-port) PORTAL_PORT="$2";; --tb-port) TB_PORT="$2";; esac
      shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done
[[ "${ENV_DIR}" == /* ]] || ENV_DIR="${ROOT_DIR}/${ENV_DIR}"
[[ "${UI_PORT}" != "${KLINE_PORT}" && "${UI_PORT}" != "${PORTAL_PORT}" && "${UI_PORT}" != "${TB_PORT}" && "${KLINE_PORT}" != "${PORTAL_PORT}" && "${KLINE_PORT}" != "${TB_PORT}" && "${PORTAL_PORT}" != "${TB_PORT}" ]] || { echo 'ports must be distinct' >&2; exit 2; }
for port in "${UI_PORT}" "${KLINE_PORT}" "${PORTAL_PORT}" "${TB_PORT}"; do
  [[ "${port}" -ge 1024 && "${port}" -le 65535 ]] || { echo 'ports must be 1024-65535' >&2; exit 2; }
done
mkdir -p "${ROOT_DIR}/logs/server"
if [[ -f "${ROOT_DIR}/SOURCE_MANIFEST.json" ]]; then
  python3 "${ROOT_DIR}/scripts/package_server.py" --verify "${ROOT_DIR}"
fi
find "${ROOT_DIR}" -maxdepth 1 -type f -name '*.sh' -exec chmod +x {} +
exec > >(tee -a "${ROOT_DIR}/logs/server/deploy.log") 2>&1
trap 'echo "Deployment failed at line ${LINENO}; see logs/server/deploy.log" >&2' ERR
if [[ "${INSTALL}" -eq 1 ]]; then
  # Use the known-compatible Torch 2.7 CUDA wheel line unless explicitly set.
  export LGDOG_TORCH_INDEX_URL="${LGDOG_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
  bash "${ROOT_DIR}/install_ubuntu.sh" --env-dir "${ENV_DIR}" "${INSTALL_ARGS[@]}"
fi
PYTHON_BIN="${ENV_DIR}/bin/python"
[[ -x "${PYTHON_BIN}" ]] || { echo "Missing ${PYTHON_BIN}; run without --skip-install" >&2; exit 1; }
export LGDOG_PYTHON="${PYTHON_BIN}"
"${PYTHON_BIN}" scripts/verify_install.py --require-cuda
printf '%s\n' "${PYTHON_BIN}" > "${ROOT_DIR}/logs/server/python_path.txt"
"${PYTHON_BIN}" -m pip freeze > "${ROOT_DIR}/logs/server/installed-packages.txt"
if [[ "${SMOKE}" -eq 1 ]]; then
  "${PYTHON_BIN}" scripts/train.py LgDog1-Rough-Blind-HIM-Standard \
    --env.scene.num-envs=64 --agent.max-iterations=2 --agent.logger=tensorboard \
    --agent.experiment-name=lgdog_deploy_smoke --agent.run-name="DEPLOY_SMOKE_$(date +%Y%m%d_%H%M%S)" \
    --terrain-curriculum=lp_acrl --velocity-min-abs=.3 --velocity-max-abs=.5 --enable-nan-guard True
fi
if [[ "${START}" -eq 1 ]]; then
  "${PYTHON_BIN}" lgdog.py --python "${PYTHON_BIN}" server start \
    --ui-port "${UI_PORT}" --kline-port "${KLINE_PORT}" --portal-port "${PORTAL_PORT}" --tb-port "${TB_PORT}"
fi
printf '\nReady. From your Windows machine (replace SSH_HOST and SSH_PORT):\n'
printf 'ssh -N -p SSH_PORT -L %s:127.0.0.1:%s -L %s:127.0.0.1:%s -L %s:127.0.0.1:%s -L %s:127.0.0.1:%s root@SSH_HOST\n' \
  "${UI_PORT}" "${UI_PORT}" "${KLINE_PORT}" "${KLINE_PORT}" "${PORTAL_PORT}" "${PORTAL_PORT}" "${TB_PORT}" "${TB_PORT}"
printf 'Workbench: http://127.0.0.1:%s/\nOverview: http://127.0.0.1:%s/\n' "${UI_PORT}" "${PORTAL_PORT}"
printf 'Control: LGDOG_PYTHON=%q bash run_lgdog.sh server status\n' "${PYTHON_BIN}"
