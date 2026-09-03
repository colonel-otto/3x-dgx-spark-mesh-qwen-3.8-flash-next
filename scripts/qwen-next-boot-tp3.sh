#!/usr/bin/env bash
# qwen-next-boot-tp3.sh — launch Qwen 3.8 Flash Next (~180B MoE) on 3 nodes (TP=3: sparkmain + spark1 + spark2)
set -e

MNBT="${1:-8192}"
LOG="${2:-$HOME/qwen-next-tp3-mnbt${MNBT}.log}"

[ -f "$HOME/.eugr-nodes" ] && . "$HOME/.eugr-nodes"
NODE0="${NODE0:-192.168.10.10}"
NODE1="${NODE1:-192.168.10.11}"
NODE2="${NODE2:-192.168.10.12}"
NODES="$NODE0,$NODE1,$NODE2"

CACHE_ROOT=/opt/eugrcache
LAUNCHER="$HOME/eugr-launcher"
RECIPE="qwen3.8-next-flash-nvfp4-tp3"

echo "=== qwen-next-boot-tp3: mnbt=$MNBT nodes=$NODES ==="
echo "    recipe: $RECIPE"
echo "    log:    $LOG"

for n in $NODE0 $NODE1 $NODE2; do
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
  -v /opt/qwen-patches/virtual_tp.py:/usr/local/lib/python3.12/dist-packages/vllm/config/virtual_tp.py \
  -v /opt/qwen-patches/registry.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/registry.py \
  -v /opt/qwen-patches/speculative.py:/usr/local/lib/python3.12/dist-packages/vllm/config/speculative.py \
  -v /opt/qwen-patches/qwen3_next.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_next.py \
  -v /opt/qwen-patches/qwen3_next_mtp.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_next_mtp.py \
  -v /opt/qwen-patches/vocab_parallel_embedding.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/vocab_parallel_embedding.py \
  -v /opt/qwen-patches/kernel_warmup.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/warmup/kernel_warmup.py \
  --no-cache-dirs \
  --gpu-memory-utilization 0.82 \
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
