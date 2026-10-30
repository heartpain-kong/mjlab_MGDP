#!/usr/bin/env bash
# Install the LgDog mjlab environment on Ubuntu 22.04/24.04.
# Usage: bash install_ubuntu.sh [options]

set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="/root/gpufree-data"
if [[ -n "${LGDOG_ENV_DIR:-}" ]]; then
  ENV_DIR="${LGDOG_ENV_DIR}"
elif [[ -d "${DATA_ROOT}" && -w "${DATA_ROOT}" ]]; then
  ENV_DIR="${DATA_ROOT}/lgdog-venv"
else
  ENV_DIR="${ROOT_DIR}/.venv"
fi
if [[ -n "${LGDOG_PIP_CACHE_DIR:-}" ]]; then
  PIP_CACHE_DIR="${LGDOG_PIP_CACHE_DIR}"
elif [[ -d "${DATA_ROOT}" && -w "${DATA_ROOT}" ]]; then
  PIP_CACHE_DIR="${DATA_ROOT}/.cache/pip"
else
  PIP_CACHE_DIR="${ROOT_DIR}/.cache/pip"
fi
PYTHON_REQUEST="${LGDOG_PYTHON:-}"
TORCH_INDEX_URL="${LGDOG_TORCH_INDEX_URL:-}"
SKIP_SYSTEM=0
SKIP_CHECK=0
CPU_ONLY=0

log() { printf '[lgdog-install] %s\n' "$*"; }
die() { printf '[lgdog-install] ERROR: %s\n' "$*" >&2; exit 1; }

usage() {
  cat <<'EOF'
Usage: bash install_ubuntu.sh [options]

Creates an isolated .venv, installs the pinned mjlab/MuJoCo/Warp runtime,
installs this checkout in editable mode, and verifies LgDog task registration.

Options:
  --env-dir PATH       virtual environment path (default: /root/gpufree-data/lgdog-venv on GPUFree)
  --python PATH        Python 3.10-3.13 executable to use
  --torch-index URL    PyTorch index used only for torch installation
  --cpu                install the CPU PyTorch wheel (for import checks only)
  --skip-system-deps   do not run apt-get
  --skip-check         install without the final runtime check
  -h, --help           show this help

Examples:
  bash install_ubuntu.sh
  bash install_ubuntu.sh --env-dir /opt/lgdog-venv --torch-index https://download.pytorch.org/whl/cu128
  bash install_ubuntu.sh --cpu
EOF
}

while (($#)); do
  case "$1" in
    --env-dir)
      (($# >= 2)) || die "--env-dir needs a path"
      ENV_DIR="$2"; shift 2 ;;
    --python)
      (($# >= 2)) || die "--python needs an executable"
      PYTHON_REQUEST="$2"; shift 2 ;;
    --torch-index)
      (($# >= 2)) || die "--torch-index needs a URL"
      TORCH_INDEX_URL="$2"; shift 2 ;;
    --cpu) CPU_ONLY=1; shift ;;
    --skip-system-deps) SKIP_SYSTEM=1; shift ;;
    --skip-check) SKIP_CHECK=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option: $1 (use --help)" ;;
  esac
done

[[ -f "${ROOT_DIR}/setup.py" ]] || die "run this script from the project checkout"
mkdir -p "${PIP_CACHE_DIR}"
export PIP_CACHE_DIR

if [[ -r /etc/os-release ]]; then
  # shellcheck disable=SC1091
  source /etc/os-release
  if [[ "${ID:-}" != "ubuntu" ]]; then
    log "warning: detected ${PRETTY_NAME:-a non-Ubuntu system}; continuing"
  fi
fi

if [[ "${SKIP_SYSTEM}" -eq 0 ]]; then
  APT_PACKAGES=(
    build-essential cmake git pkg-config
    python3-venv python3-dev
    libegl1 libgl1 libglib2.0-0 libx11-6 libxext6 libxrender1
  )
  APT_RUNNER=()
  if [[ "${EUID}" -ne 0 ]]; then
    command -v sudo >/dev/null 2>&1 || die "sudo is required for apt packages; rerun with --skip-system-deps"
    APT_RUNNER=(sudo)
  fi
  if command -v apt-get >/dev/null 2>&1; then
    log "installing system packages"
    "${APT_RUNNER[@]}" apt-get update
    "${APT_RUNNER[@]}" env DEBIAN_FRONTEND=noninteractive apt-get install -y "${APT_PACKAGES[@]}"
  else
    die "apt-get was not found; rerun with --skip-system-deps only if the image already has build/runtime libraries"
  fi
fi

if [[ -n "${PYTHON_REQUEST}" ]]; then
  PYTHON_BIN="${PYTHON_REQUEST}"
else
  PYTHON_BIN=""
  for candidate in python3.11 python3.12 python3.10 python3; do
    if command -v "${candidate}" >/dev/null 2>&1; then
      candidate_version="$(${candidate} -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')"
      case "${candidate_version}" in
        3.10|3.11|3.12|3.13) PYTHON_BIN="${candidate}"; break ;;
      esac
    fi
  done
fi
[[ -n "${PYTHON_BIN}" ]] || die "Python 3.10-3.13 was not found; pass --python /path/to/python3.11"
"${PYTHON_BIN}" - <<'PY'
import sys
if not (3, 10) <= sys.version_info[:2] < (3, 14):
    raise SystemExit(f"Python 3.10-3.13 is required, found {sys.version.split()[0]}")
PY

if [[ ! -x "${ENV_DIR}/bin/python" ]]; then
  log "creating virtual environment at ${ENV_DIR}"
  "${PYTHON_BIN}" -m venv "${ENV_DIR}"
fi
VENV_PYTHON="${ENV_DIR}/bin/python"
[[ -x "${VENV_PYTHON}" ]] || die "virtual environment creation failed: ${VENV_PYTHON}"
"${VENV_PYTHON}" - <<'PY'
import sys
if not (3, 10) <= sys.version_info[:2] < (3, 14):
    raise SystemExit("Existing venv needs Python 3.10-3.13; choose a new --env-dir instead of replacing it")
PY

log "upgrading packaging tools"
"${VENV_PYTHON}" -m pip install --upgrade pip setuptools wheel

if [[ "${CPU_ONLY}" -eq 1 ]]; then
  TORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"
fi
if [[ -n "${TORCH_INDEX_URL}" ]]; then
  log "installing PyTorch from ${TORCH_INDEX_URL}"
  "${VENV_PYTHON}" -m pip install --index-url "${TORCH_INDEX_URL}" "torch>=2.7.0,<2.8.0"
else
  log "installing PyTorch from the default package index"
  "${VENV_PYTHON}" -m pip install "torch>=2.7.0,<2.8.0"
fi

log "installing pinned project dependencies"
"${VENV_PYTHON}" -m pip install --extra-index-url https://pypi.nvidia.com -r "${ROOT_DIR}/requirements-ubuntu.txt"

log "installing this checkout in editable mode"
"${VENV_PYTHON}" -m pip install --no-deps --editable "${ROOT_DIR}"

log "checking package dependencies"
"${VENV_PYTHON}" -m pip check

if [[ "${SKIP_CHECK}" -eq 0 ]]; then
  log "checking imports and LgDog task registration"
  (cd "${ROOT_DIR}" && "${VENV_PYTHON}" scripts/verify_install.py)
fi

cat <<EOF

Installation complete.

Activate:
  source "${ENV_DIR}/bin/activate"
Run:
  cd "${ROOT_DIR}"
  python lgdog.py tasks
  python scripts/train.py LgDog1-Rough-Blind-HIM-Standard --env.scene.num-envs=1024
EOF
