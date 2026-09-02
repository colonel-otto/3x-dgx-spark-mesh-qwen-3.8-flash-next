#!/usr/bin/env bash
# qwen-next-boot-tp2.sh — launch Qwen 3.8 Flash Next (~180B MoE) on 2 nodes (TP=2: sparkmain + spark1)
set -euo pipefail

MNBT="${1:-8192}"
LOG="${2:-$HOME/qwen-next-tp2-mnbt${MNBT}.log}"

[ -f "$HOME/.eugr-nodes" ] && . "$HOME/.eugr-nodes"
NODE0="${NODE0:-192.168.10.10}"
NODE1="${NODE1:-192.168.10.11}"
NODES="$NODE0,$NODE1"

CACHE_ROOT=/opt/eugrcache
LAUNCHER="$HOME/eugr-launcher"
RECIPE="qwen3.8-next-flash-nvfp4-tp2"

echo "=== qwen-next-boot-tp2: mnbt=$MNBT nodes=$NODES ==="
echo "    recipe: $RECIPE"
echo "    log:    $LOG"

for n in $NODE0 $NODE1; do
  ssh -o ConnectTimeout=10 "$n" "
    set -e
    sudo -n mkdir -p ${CACHE_ROOT}-vllm ${CACHE_ROOT}-flashinfer ${CACHE_ROOT}-triton ${CACHE_ROOT}-tilelang
    sudo -n chmod 777 ${CACHE_ROOT}-vllm ${CACHE_ROOT}-flashinfer ${CACHE_ROOT}-triton ${CACHE_ROOT}-tilelang
  " || { echo "PRECONDITION FAILED on $n"; exit 1; }
  echo "  preconditions OK: $n"
done

cd "$LAUNCHER"
nohup python3 run-recipe.py "$RECIPE" \
  -t eugr/spark-vllm-b12x:latest \
  -n "$NODES" \
  -v ${CACHE_ROOT}-vllm:/root/.cache/vllm \
  -v ${CACHE_ROOT}-flashinfer:/root/.cache/flashinfer \
  -v ${CACHE_ROOT}-triton:/root/.triton \
  -v ${CACHE_ROOT}-tilelang:/root/.tilelang \
  --no-cache-dirs \
  --gpu-memory-utilization 0.85 \
  --port 8100 \
  -e "NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0,rocep1s0f1,roceP2p1s0f1" \
  -e NCCL_IB_SUBNET_AWARE_ROUTING=1 \
  -e NCCL_NET_PLUGIN=none \
  -e NCCL_IB_MERGE_NICS=0 \
  -e NCCL_BUFFSIZE=16777216 \
  -e NCCL_TIMEOUT=3600 \
  > "$LOG" 2>&1 &

PID=$!
echo "$PID" > "$HOME/.qwen-launcher.pid"
echo "  launched pid $PID -> $LOG"
