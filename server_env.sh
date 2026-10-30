#!/usr/bin/env bash
# Source this file before launching Python; changes stay in this process tree.
lgdog_server_env() {
  local project_dir="$1" data_dir driver candidate driver_dir entry clean_path=""
  data_dir="${LGDOG_DATA_ROOT:-/root/gpufree-data}"
  if [[ ! -d "${data_dir}" || ! -w "${data_dir}" ]]; then data_dir="${project_dir}"; fi
  export PYTHONPATH="${project_dir}${PYTHONPATH:+:${PYTHONPATH}}"
  export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
  export MUJOCO_GL="${MUJOCO_GL:-egl}"
  export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${data_dir}/.cache}"
  export PIP_CACHE_DIR="${LGDOG_PIP_CACHE_DIR:-${PIP_CACHE_DIR:-${data_dir}/.cache/pip}}"
  [[ "${LGDOG_SKIP_CUDA_DRIVER:-0}" != 1 ]] || return 0
  driver="${LGDOG_CUDA_DRIVER:-}"
  if [[ -n "${driver}" && ! -f "${driver}" ]]; then
    printf 'LGDOG_CUDA_DRIVER not found: %s\n' "${driver}" >&2; return 1
  fi
  if [[ -z "${driver}" ]]; then
    for candidate in /lib/x86_64-linux-gnu/libcuda.so.1 /usr/lib/x86_64-linux-gnu/libcuda.so.1 /usr/local/nvidia/lib64/libcuda.so.1; do
      if [[ -f "${candidate}" ]]; then driver="${candidate}"; break; fi
    done
  fi
  if [[ -n "${driver}" ]]; then
    [[ "${driver}" != *'/stubs/'* ]] || { printf 'CUDA stub is not a driver: %s\n' "${driver}" >&2; return 1; }
    driver_dir="$(dirname -- "${driver}")"
    local -a old_entries=()
    IFS=: read -r -a old_entries <<< "${LD_LIBRARY_PATH:-}"
    for entry in "${old_entries[@]}"; do
      [[ -n "${entry}" && "${entry}" != *'/stubs'* && "${entry}" != "${driver_dir}" ]] || continue
      clean_path+="${clean_path:+:}${entry}"
    done
    export LD_LIBRARY_PATH="${driver_dir}${clean_path:+:${clean_path}}"
    case ":${LD_PRELOAD:-}:" in
      *":${driver}:"*) ;;
      *) export LD_PRELOAD="${driver}${LD_PRELOAD:+:${LD_PRELOAD}}" ;;
    esac
    export LGDOG_CUDA_DRIVER="${driver}"
  fi
}
