#!/usr/bin/env bash
# TensorBoard 启停脚本 —— 查看 V 轮次训练曲线
#
#   bash tb.sh start     启动（已在跑则忽略）
#   bash tb.sh stop      停止
#   bash tb.sh restart   重启
#   bash tb.sh status    查看状态 / 已识别的 run
#   bash tb.sh log       跟踪 tensorboard 自身日志
#
# 注意：必须用**绝对路径** --logdir。历史上踩过的坑是相对路径 + cwd 后来被删，
#       结果 TensorBoard 一直在跑但 /data/runs 返回 []，页面空白。

set -u

PROJ="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LOGDIR="$PROJ/logs/rsl_rl"
TBDIR="$PROJ/logs/tensorboard"
PORT="${TB_PORT:-6006}"
TB="${TB_BIN:-$(command -v tensorboard || true)}"

mkdir -p "$TBDIR"
PIDFILE="$TBDIR/tb.pid"

get_pid() {
  # 优先读 pidfile，并校验进程确实存在且是 tensorboard
  if [ -f "$PIDFILE" ]; then
    local p; p=$(cat "$PIDFILE" 2>/dev/null)
    if [ -n "$p" ] && [ -r "/proc/$p/cmdline" ]; then
      local command_line; command_line=$(tr '\0' ' ' < "/proc/$p/cmdline")
      if [[ "$command_line" == *tensorboard* && "$command_line" == *"$LOGDIR"* ]]; then
        echo "$p"; return 0
      fi
    fi
  fi
  # 兜底：按监听端口的 pid 找（避免 pkill -f 自杀）
  local p
  p=$(ss -ltnp 2>/dev/null | grep ":$PORT " | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2)
  if [ -n "${p:-}" ] && [ -r "/proc/$p/cmdline" ]; then
    local command_line; command_line=$(tr '\0' ' ' < "/proc/$p/cmdline")
    [[ "$command_line" == *tensorboard* && "$command_line" == *"$LOGDIR"* ]] && echo "$p"
  fi
}

cmd_start() {
  if [ -z "$TB" ]; then echo "请激活训练环境，或设置 TB_BIN 指向 tensorboard" >&2; return 1; fi
  local p; p=$(get_pid)
  if [ -n "${p:-}" ]; then
    echo "TensorBoard 已在运行 (pid $p) -> http://localhost:$PORT"
    return 0
  fi
  cd "$PROJ" || exit 1
  setsid nohup "$TB" \
    --logdir "$LOGDIR" \
    --host 0.0.0.0 --port "$PORT" \
    --reload_multifile=true --reload_interval 5 \
    --samples_per_plugin scalars=2000 \
    > "$TBDIR/tensorboard.log" 2>&1 < /dev/null &
  echo $! > "$PIDFILE"
  sleep 6
  p=$(get_pid)
  if [ -n "${p:-}" ]; then
    echo "已启动 (pid $p)"
    echo "  本机:      http://localhost:$PORT"
    ip=$(hostname -I 2>/dev/null | awk '{print $1}')
    [ -n "$ip" ] && echo "  局域网:    http://$ip:$PORT"
    echo "  工作目录:  $LOGDIR"
  else
    echo "启动失败，看日志: $TBDIR/tensorboard.log" >&2
    tail -20 "$TBDIR/tensorboard.log" >&2
    return 1
  fi
}

cmd_stop() {
  local p; p=$(get_pid)
  if [ -z "${p:-}" ]; then echo "没在运行"; rm -f "$PIDFILE"; return 0; fi
  kill "$p" 2>/dev/null
  sleep 2
  # 顺带收掉 data server 子进程（父进程退出后它有时会残留占端口）
  kill "$(get_pid)" 2>/dev/null
  rm -f "$PIDFILE"
  echo "已停止 (was pid $p)"
}

cmd_status() {
  local p; p=$(get_pid)
  if [ -n "${p:-}" ]; then
    echo "运行中: pid $p  http://localhost:$PORT"
  else
    echo "未运行"
  fi
  echo "logdir: $LOGDIR"
  echo "已识别的 run:"
  curl -s --max-time 8 "http://127.0.0.1:$PORT/data/runs" 2>/dev/null \
    | tr ',' '\n' | tr -d '[]"' | sed '/^$/d;s/^/  - /' || echo "  (取不到，可能没起来)"
}

case "${1:-status}" in
  start)   cmd_start ;;
  stop)    cmd_stop ;;
  restart) cmd_stop; cmd_start ;;
  status)  cmd_status ;;
  log)     tail -f "$TBDIR/tensorboard.log" ;;
  *) echo "用法: bash tb.sh {start|stop|restart|status|log}"; exit 1 ;;
esac
