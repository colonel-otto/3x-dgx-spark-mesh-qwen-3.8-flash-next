#!/usr/bin/env bash
# sglang-flashnext-boot.sh -- launch RadixArk/Qwen3.8-Flash-Next-NVFP4 on SGLang
# across 2 or 3 DGX Sparks.
#
#   ./scripts/sglang-flashnext-boot.sh <tp:2|3> [log]
#
# Written 2026-09-04. Flags are the community 2-Spark recipe
# (github.com/tonyd2wild/Qwen3.8-Flash-Next-NVFP4-DGX-Spark, verified serving
# 2026-08-26..28 on GB10 at TP=2) merged with THIS cluster's fabric settings
# from 3spark-qwen-3.8-27b-dense/scripts/sglang-boot-dspark-fp8.sh. Image is
# sglang-flashnext-tp3:local for BOTH TP sizes (built by
# docker/build-sglang-flashnext.sh from the digest-pinned official day-0
# image) so the two arms differ only in TP; the padding patch is inert unless
# SGLANG_FORCE_UNALIGNED_TP=1, which only the TP=3 arm sets.
#
# TP=2 has NEVER been booted on this cluster and TP=3 has never been booted
# ANYWHERE for this model (no public TP=3 Qwen4Exp run exists as of
# 2026-09-04). Bring up TP=2 FIRST: every Flash-Next dimension divides by 2,
# so TP=2 exercises the day-0 engine + GB10 NVFP4 MoE kernels + our fabric
# with ZERO padding involved, and isolates model-port bugs from sharding bugs.
#
# Refuses to run if another engine holds the GPUs on any target node.
# Keep gpu_idle_guard.py alongside this launcher when copying it to a node.
set -euo pipefail

TP="${1:?usage: sglang-flashnext-boot.sh <tp:2|3> [log]}"
LOG="${2:-$HOME/sglang-flashnext-tp${TP}.log}"
case "$TP" in 2|3) ;; *) echo "FATAL: tp must be 2 or 3 (got '$TP')" >&2; exit 2 ;; esac

[ -f "$HOME/.eugr-nodes" ] && . "$HOME/.eugr-nodes"
NODE0="${NODE0:?NODE0 unset -- is ~/.eugr-nodes present?}"
NODE1="${NODE1:?NODE1 unset}"
NODE2="${NODE2:-}"
if [ "$TP" = 3 ]; then
  [ -n "$NODE2" ] || { echo "FATAL: tp=3 needs NODE2" >&2; exit 2; }
  NODES=("$NODE0" "$NODE1" "$NODE2")
else
  NODES=("$NODE0" "$NODE1")
fi

IMAGE="${IMAGE:-sglang-flashnext-tp3:local}"
CONTAINER="sglang_node"
PORT=8100
MODEL="RadixArk/Qwen3.8-Flash-Next-NVFP4"
SERVED="qwen3.8-flash-next-nvfp4,qwen3.8-flash-next"

# --- Knobs that BENCHMARK-POLICY.md requires to be identical across arms and
# verified from the live process (ps -eo args), not from this file.
MAX_RUNNING="${MAX_RUNNING:-16}"           # production concurrency ceiling
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-2400000}"  # 2.4M tokens (~28.6 GB headroom per node at TP=3)
MEM_FRACTION="${MEM_FRACTION:-0.80}"
MAX_MAMBA_CACHE="${MAX_MAMBA_CACHE:-97}"  # SSM pool state count
SPEC_STEPS="${SPEC_STEPS:-3}"             # NEXTN (native MTP) draft depth; recipe's MTP4 = 3 steps / 4 draft tokens
SPEC_DRAFT_TOKENS="${SPEC_DRAFT_TOKENS:-4}"

# --- TP-specific
if [ "$TP" = 3 ]; then
  # Q 24->36, KV 2->3, GDN 16->18/48->54, MoE 640->672 need the gate lift;
  # the vision tower (16 heads) must be replicated at TP=1 via dp-encoder.
  EXTRA_ENV=(-e SGLANG_FORCE_UNALIGNED_TP=1)
  EXTRA_ARGS=(--mm-enable-dp-encoder)
  # Point-to-point triangle: NO subnet is common to all 3 nodes, so hand NCCL
  # every HCA with exact-match prefix and let SUBNET_AWARE_ROUTING pick the
  # link per peer. Narrowing per pair (correct for TP=2) is fatal here.
  HCAS='=rocep1s0f0,roceP2p1s0f0,rocep1s0f1,roceP2p1s0f1'
else
  EXTRA_ENV=()
  EXTRA_ARGS=()
  # TP=2 with a SAR-capable NCCL (2.30.7 in this image) should also be fine
  # with the full list; if "Init torch distributed begin" hangs, fall back to
  # the dense repo's per-pair narrowing (scripts/fabric-hca.sh common/for).
  HCAS='=rocep1s0f0,roceP2p1s0f0,rocep1s0f1,roceP2p1s0f1'
fi

echo "=== sglang-flashnext-boot: TP=$TP nodes=${NODES[*]} image=$IMAGE ==="

# --- Guards
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
for n in "${NODES[@]}"; do
  ssh -o ConnectTimeout=10 -o BatchMode=yes "$n" "python3 - '$PORT'" \
    < "$SCRIPT_DIR/gpu_idle_guard.py" \
    || { echo "FATAL: cannot establish idle GPUs on $n; launch aborted" >&2; exit 1; }
  ssh -o ConnectTimeout=10 -o BatchMode=yes "$n" "docker image inspect '$IMAGE' >/dev/null 2>&1" \
    || { echo "FATAL: image $IMAGE missing on $n -- run docker/build-sglang-flashnext.sh" >&2; exit 1; }
done

# --- GB10 unified-memory page-cache trap (recipe: mandatory before every
# launch). A warm page cache starves the GPU allocator ~20 min into load and
# free -g under-reports on GB10. Best-effort: needs passwordless sudo.
for n in "${NODES[@]}"; do
  ssh -o ConnectTimeout=10 -o BatchMode=yes "$n" \
    "sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null 2>&1 && echo '  drop_caches OK: $n' || echo '  drop_caches SKIPPED (no sudo -n): $n'; sudo -n swapoff -a && sudo -n swapon -a && echo '  swap clean OK: $n' || true; mkdir -p ~/.cache/flashinfer ~/.triton ~/.cache/sglang ~/.cache/huggingface"
done

DIST_INIT="${NODES[0]}:20000"
EXTRA_ENV_STR="${EXTRA_ENV[*]:-}"
EXTRA_ARGS_STR="${EXTRA_ARGS[*]:-}"

for i in "${!NODES[@]}"; do
  n="${NODES[$i]}"
  echo "  starting rank $i on $n (NCCL_IB_HCA=$HCAS)"
  # RDMA passthrough: WITHOUT --privileged + /dev/infiniband + IPC_LOCK +
  # memlock=-1, NCCL silently falls back to TCP over the 1GbE management link.
  # Persistent JIT caches (sglang/flashinfer/triton) are mounted: JIT autotuning
  # and compiled kernels persist across launches.
  ssh -o ConnectTimeout=15 -o BatchMode=yes "$n" \
    "docker rm -f '$CONTAINER' >/dev/null 2>&1 || true
     docker run -d --name '$CONTAINER' \
       --gpus all --network host --ipc host --shm-size 32g \
       --privileged --device /dev/infiniband \
       --cap-add IPC_LOCK --ulimit memlock=-1:-1 \
       -v \$HOME/.cache/huggingface:/root/.cache/huggingface \
       -v \$HOME/.cache/flashinfer:/root/.cache/flashinfer \
       -v \$HOME/.cache/sglang:/root/.cache/sglang \
       -v \$HOME/.triton:/root/.triton \
       -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
       -e TORCH_CUDA_ARCH_LIST=12.1a -e FLASHINFER_CUDA_ARCH_LIST=12.1a \
       -e NCCL_IB_HCA=$HCAS \
       -e NCCL_IB_SUBNET_AWARE_ROUTING=1 \
       -e NCCL_NET_PLUGIN=none \
       -e NCCL_IB_MERGE_NICS=0 \
       -e NCCL_BUFFSIZE=16777216 \
       -e NCCL_TIMEOUT=3600 \
       -e NCCL_CUMEM_ENABLE=0 \
       -e TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
       -e NCCL_DEBUG=WARN \
       $EXTRA_ENV_STR \
       '$IMAGE' \
       sglang serve \
         --model-path '$MODEL' \
         --served-model-name $SERVED \
         --host 0.0.0.0 --port $PORT \
         --tp-size $TP --nnodes $TP --node-rank $i \
         --dist-init-addr $DIST_INIT \
         --trust-remote-code \
         --quantization modelopt_fp4 \
         --fp4-gemm-backend flashinfer_cutlass \
         --page-size 64 \
         --mamba-scheduler-strategy extra_buffer \
         --mamba-track-interval 64 \
         --max-mamba-cache-size $MAX_MAMBA_CACHE \
         --speculative-algorithm NEXTN \
         --speculative-num-steps $SPEC_STEPS \
         --speculative-eagle-topk 1 \
         --speculative-num-draft-tokens $SPEC_DRAFT_TOKENS \
         --enable-linear-replayssm-spec \
         --speculative-attention-mode decode \
         --chunked-prefill-size 8192 \
         --max-running-requests $MAX_RUNNING \
         --context-length 262144 \
         --max-total-tokens $MAX_TOTAL_TOKENS \
         --mem-fraction-static $MEM_FRACTION \
         --allow-auto-truncate \
         --reasoning-parser auto \
         --tool-call-parser qwen3_coder \
         --default-chat-template-kwargs '{\"enable_thinking\": false}' \
         --ple-offload-embedding \
         --cuda-graph-max-bs 8 \
         --disable-cuda-graph-padding \
         --disable-prefill-cuda-graph \
         --enable-metrics \
         --sampling-backend pytorch $EXTRA_ARGS_STR" \
    || { echo "FATAL: rank $i failed to start on $n" >&2; exit 1; }
done

echo
echo "  All ranks launched. Streaming rank-0 log -> $LOG"
ssh -o ConnectTimeout=15 -o BatchMode=yes "${NODES[0]}" "docker logs -f '$CONTAINER'" > "$LOG" 2>&1 &
echo "$!" > "$HOME/.sglang-flashnext-logtail.pid"

cat <<MSG

  Startup is SLOW: 126 GiB of NVFP4 weights + first-time JIT of GB10 kernels
  (TP=3 adds never-compiled padded shapes). Frozen output is not a hang;
  check RSS growth on a worker before calling it one.
      tail -f $LOG
  Ready when this returns 200 -- but HTTP 200 IS NOT HEALTH on this model:
      curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:$PORT/v1/models

  NEXT -- do not skip:
      ./scripts/sglang-flashnext-validate.sh http://127.0.0.1:$PORT/v1 qwen3.8-flash-next
      (TP=3 additionally needs the logprob gate against the TP=2 arm before any
       tok/s is recorded -- see docs/HANDOFF-2026-09-04-SGLANG-FLASH-NEXT-TP3.md)
MSG
