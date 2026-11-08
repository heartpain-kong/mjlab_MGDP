#!/usr/bin/env bash
# Configure the LgDog (mjlab) runtime in a Conda environment named "lgdog".
#
# Usage: bash configure_lgdog.sh [options]
#   --env-dir PATH       conda prefix (default: /root/gpufree-data/conda/envs/lgdog)
#   --python-version V   python version for a new env (default: 3.11)
#   --torch-index URL    PyTorch index used only for torch (e.g. https://download.pytorch.org/whl/cu128)
#   --cpu                install the CPU PyTorch wheel (import checks only)
#   --skip-system-deps   do not run apt-get
#   --skip-check         install without the final GPU runtime check
#   --start              start the workbench after configuration
#   --ui-port PORT       workbench port used by --start (default: 8088)
#   -h, --help           show this help
#
# The script is idempotent: an existing environment is reused, dependencies are
# refreshed in place, and the activation hooks are rewritten.
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${LGDOG_DATA_ROOT:-/root/gpufree-data}"
if [[ -n "${LGDOG_ENV_DIR:-}" ]]; then
  ENV_PREFIX="${LGDOG_ENV_DIR}"
elif [[ -d "${DATA_ROOT}" && -w "${DATA_ROOT}" ]]; then
  ENV_PREFIX="${DATA_ROOT}/conda/envs/lgdog"
else
  ENV_PREFIX="${HOME}/.conda/envs/lgdog"
fi
PYVER="${LGDOG_PYTHON_VERSION:-3.11}"
TORCH_INDEX_URL="${LGDOG_TORCH_INDEX_URL:-}"
SKIP_SYSTEM=0
SKIP_CHECK=0
CPU_ONLY=0
START=0
UI_PORT="${LGDOG_UI_PORT:-8088}"

log() { printf '[lgdog-config] %s\n' "$*"; }
die() { printf '[lgdog-config] ERROR: %s\n' "$*" >&2; exit 1; }

usage() {
  sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

while (($#)); do
  case "$1" in
    --env-dir) (($# >= 2)) || die "--env-dir needs a path"; ENV_PREFIX="$2"; shift 2 ;;
    --python-version) (($# >= 2)) || die "--python-version needs a value"; PYVER="$2"; shift 2 ;;
    --torch-index) (($# >= 2)) || die "--torch-index needs a URL"; TORCH_INDEX_URL="$2"; shift 2 ;;
    --cpu) CPU_ONLY=1; shift ;;
    --skip-system-deps) SKIP_SYSTEM=1; shift ;;
    --skip-check) SKIP_CHECK=1; shift ;;
    --start) START=1; shift ;;
    --ui-port) (($# >= 2)) || die "--ui-port needs a value"; UI_PORT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option: $1 (use --help)" ;;
  esac
done

[[ -f "${ROOT_DIR}/setup.py" ]] || die "run this script from the project checkout"
[[ -f "${ROOT_DIR}/requirements-ubuntu.txt" ]] || die "requirements-ubuntu.txt is missing"

command -v apt-get >/dev/null 2>&1 || true

# ── 1. Conda ────────────────────────────────────────────────────────────────
find_conda() {
  local candidate
  if command -v conda >/dev/null 2>&1; then command -v conda; return 0; fi
  for candidate in \
    "${CONDA_EXE:-}" \
    /root/miniconda3/bin/conda \
    /root/anaconda3/bin/conda \
    /opt/conda/bin/conda \
    "${HOME}/miniconda3/bin/conda" \
    "${HOME}/anaconda3/bin/conda"; do
    [[ -n "${candidate}" && -x "${candidate}" ]] && { printf '%s\n' "${candidate}"; return 0; }
  done
  return 1
}

CONDA_BIN="$(find_conda)" || die "conda was not found; install Miniconda or pass LGDOG_ENV_DIR to an existing prefix"
CONDA_BASE="$("${CONDA_BIN}" info --base 2>/dev/null || true)"
[[ -n "${CONDA_BASE}" && -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]] \
  || die "cannot locate conda.sh under ${CONDA_BASE:-<unknown>}"
# shellcheck disable=SC1090
source "${CONDA_BASE}/etc/profile.d/conda.sh"
log "conda ${CONDA_BASE}"

# ── 2. System packages ──────────────────────────────────────────────────────
if ((SKIP_SYSTEM == 0)); then
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
  log "installing system packages"
  if ! "${APT_RUNNER[@]}" apt-get update; then
    log "apt-get update failed; retrying with Ubuntu-only (Tsinghua) sources"
    LGDog_APT_SOURCE="$(mktemp)"
    cat > "${LGDog_APT_SOURCE}" <<'EOF'
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy-updates main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy-security main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ jammy-backports main restricted universe multiverse
EOF
    "${APT_RUNNER[@]}" apt-get update \
      -o "Dir::Etc::sourcelist=${LGDog_APT_SOURCE}" \
      -o "Dir::Etc::sourceparts=-" \
      -o "APT::Get::List-Cleanup=0" \
      || die "apt-get update still fails; install the packages manually or rerun with --skip-system-deps"
    "${APT_RUNNER[@]}" env DEBIAN_FRONTEND=noninteractive apt-get install -y \
      -o "Dir::Etc::sourcelist=${LGDog_APT_SOURCE}" \
      -o "Dir::Etc::sourceparts=-" \
      "${APT_PACKAGES[@]}" \
      || die "system package installation failed; rerun with --skip-system-deps"
    rm -f "${LGDog_APT_SOURCE}"
  else
    "${APT_RUNNER[@]}" env DEBIAN_FRONTEND=noninteractive apt-get install -y "${APT_PACKAGES[@]}" \
      || die "system package installation failed; rerun with --skip-system-deps"
  fi
fi

# ── 3. Conda environment named lgdog ────────────────────────────────────────
mkdir -p "$(dirname -- "${ENV_PREFIX}")"
if [[ -x "${ENV_PREFIX}/bin/python" ]]; then
  log "reusing conda environment ${ENV_PREFIX}"
else
  log "creating conda environment ${ENV_PREFIX} (python ${PYVER})"
  "${CONDA_BIN}" create --prefix "${ENV_PREFIX}" "python=${PYVER}" -y
fi
ENV_PYTHON="${ENV_PREFIX}/bin/python"
[[ -x "${ENV_PYTHON}" ]] || die "environment creation failed: ${ENV_PYTHON}"
"${ENV_PYTHON}" - <<'PY'
import sys
if not (3, 10) <= sys.version_info[:2] < (3, 14):
    raise SystemExit(f"Python 3.10-3.13 is required, found {sys.version.split()[0]}")
PY

# ── 4. Python dependencies ──────────────────────────────────────────────────
log "upgrading packaging tools"
"${ENV_PYTHON}" -m pip install --upgrade pip setuptools wheel

if ((CPU_ONLY == 1)); then
  TORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"
fi
if [[ -n "${TORCH_INDEX_URL}" ]]; then
  log "installing PyTorch from ${TORCH_INDEX_URL}"
  "${ENV_PYTHON}" -m pip install --index-url "${TORCH_INDEX_URL}" "torch>=2.7.0,<2.8.0"
else
  log "installing PyTorch from the default package index"
  "${ENV_PYTHON}" -m pip install "torch>=2.7.0,<2.8.0"
fi

log "installing pinned project dependencies"
"${ENV_PYTHON}" -m pip install --extra-index-url https://pypi.nvidia.com -r "${ROOT_DIR}/requirements-ubuntu.txt"

log "installing this checkout in editable mode"
"${ENV_PYTHON}" -m pip install --no-deps --editable "${ROOT_DIR}"

log "checking package dependencies"
"${ENV_PYTHON}" -m pip check

# ── 5. Activation hooks (CUDA driver fix + project path) ────────────────────
ACTIVATE_DIR="${ENV_PREFIX}/etc/conda/activate.d"
DEACTIVATE_DIR="${ENV_PREFIX}/etc/conda/deactivate.d"
mkdir -p "${ACTIVATE_DIR}" "${DEACTIVATE_DIR}"

cat > "${ACTIVATE_DIR}/lgdog_env.sh" <<EOF
# Generated by configure_lgdog.sh — do not edit by hand.
export LGDOG_PROJECT_ROOT="${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}\${PYTHONPATH:+:\${PYTHONPATH}}"
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
export _LGDOG_OLD_LD_PRELOAD="\${LD_PRELOAD-}"
export _LGDOG_OLD_LD_LIBRARY_PATH="\${LD_LIBRARY_PATH-}"
if [[ -f "${ROOT_DIR}/server_env.sh" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/server_env.sh"
  lgdog_server_env "${ROOT_DIR}" || true
fi
EOF

cat > "${DEACTIVATE_DIR}/lgdog_env.sh" <<'EOF'
# Generated by configure_lgdog.sh — do not edit by hand.
if [[ -n "${_LGDOG_OLD_LD_PRELOAD+x}" ]]; then
  if [[ -n "${_LGDOG_OLD_LD_PRELOAD}" ]]; then export LD_PRELOAD="${_LGDOG_OLD_LD_PRELOAD}"; else unset LD_PRELOAD; fi
  unset _LGDOG_OLD_LD_PRELOAD
fi
if [[ -n "${_LGDOG_OLD_LD_LIBRARY_PATH+x}" ]]; then
  if [[ -n "${_LGDOG_OLD_LD_LIBRARY_PATH}" ]]; then export LD_LIBRARY_PATH="${_LGDOG_OLD_LD_LIBRARY_PATH}"; else unset LD_LIBRARY_PATH; fi
  unset _LGDOG_OLD_LD_LIBRARY_PATH
fi
unset LGDOG_PROJECT_ROOT
EOF

# 让 run_lgdog.sh 默认使用这个解释器
mkdir -p "${ROOT_DIR}/logs/server"
printf '%s\n' "${ENV_PYTHON}" > "${ROOT_DIR}/logs/server/python_path.txt"

# ── 6. Runtime check ────────────────────────────────────────────────────────
if ((SKIP_CHECK == 0)); then
  log "checking imports, task registration and CUDA/Warp allocations"
  (
    set +u
    # shellcheck disable=SC1090
    source "${ACTIVATE_DIR}/lgdog_env.sh"
    cd "${ROOT_DIR}"
    "${ENV_PYTHON}" scripts/verify_install.py --require-cuda
  ) || die "runtime check failed; see the output above"
fi

cat <<EOF

Conda environment ready: lgdog
  prefix : ${ENV_PREFIX}
  python : ${ENV_PYTHON}
  project: ${ROOT_DIR}

Activate:
  conda activate ${ENV_PREFIX}

Workbench (optional):
  bash run_lgdog.sh server start --ui-port ${UI_PORT}
  bash run_lgdog.sh server status

Train:
  python lgdog.py tasks
  python -u lgdog.py train race_mgdp --envs 1024 --iterations 16000 \\
    --terrain-num-rows 8 --terrain-num-cols 10 --ppo-mini-batches 8
EOF

if ((START == 1)); then
  log "starting workbench on port ${UI_PORT}"
  LGDOG_PYTHON="${ENV_PYTHON}" bash "${ROOT_DIR}/run_lgdog.sh" server start --ui-port "${UI_PORT}"
fi
