#!/usr/bin/env bash
# qwen-stop.sh — teardown Qwen cluster deployment cleanly
set -uo pipefail

[ -f "$HOME/.eugr-nodes" ] && . "$HOME/.eugr-nodes"
NODE0="${NODE0:-192.168.10.10}"
NODE1="${NODE1:-192.168.10.11}"
NODE2="${NODE2:-192.168.10.12}"
CONTAINER="vllm_node"
PIDFILE="$HOME/.qwen-launcher.pid"

if [ -r "$PIDFILE" ]; then
  pid=$(tr -d '[:space:]' < "$PIDFILE")
  if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
    echo "qwen-stop: killing launcher pid: $pid"
    kill "$pid" 2>/dev/null || true
    sleep 3
    kill -9 "$pid" 2>/dev/null || true
  fi
  rm -f "$PIDFILE"
fi

# Stop any remaining run-recipe processes
pkill -f 'run-recipe\.py.*qwen' 2>/dev/null || true

rc=0
for n in "$NODE0" "$NODE1" "$NODE2"; do
  ssh -o ConnectTimeout=10 -o BatchMode=yes "$n" \
    "docker rm -f '$CONTAINER' >/dev/null 2>&1 || true" || true
  left=$(ssh -o ConnectTimeout=10 -o BatchMode=yes "$n" \
          "docker ps -a --filter name='^${CONTAINER}\$' --format '{{.Names}}'" 2>/dev/null || echo UNREACHABLE)
  if [ -n "$left" ]; then
    echo "qwen-stop: WARNING $n still shows: $left"
    rc=1
  else
    echo "qwen-stop: $n clean"
  fi
done

exit $rc
