#!/usr/bin/env bash
# docker/build-sglang-flashnext.sh -- build sglang-flashnext-tp3:local on all
# 3 Sparks from the digest-pinned day-0 base and verify the ENGINE is
# byte-identical across nodes (by content, not image ID).
#
# Ported 2026-09-04 from 3spark-qwen-3.8-27b-dense/docker/build.sh. Differences:
#   - base is pinned to a fixed digest constant (override with SGLANG_BASE);
#   - no NCCL donor stage (the base's NCCL 2.30.7 already has SUBNET_AWARE_ROUTING);
#   - fingerprint list covers the 9 files the Flash-Next patch rewrites;
#   - libnccl lives in the /opt/sglang venv, not /usr/local/lib.
#
# Run FROM sparkmain (NODE0). Builds locally, then syncs patch+scripts to the
# workers and builds there too (no registry on the LAN).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

[ -f "$HOME/.eugr-nodes" ] && . "$HOME/.eugr-nodes"
NODE0="${NODE0:?NODE0 unset -- is ~/.eugr-nodes present?}"
NODE1="${NODE1:?NODE1 unset}"
NODE2="${NODE2:?NODE2 unset}"

IMAGE_TAG="sglang-flashnext-tp3:local"
BUILD_DIR_REMOTE="~/3spark-qwen-flash-build"

# Digest verified 2026-09-04 (== lmsysorg/sglang:qwen38flashnext and
# :dev-qwen38flashnext on that date). The tag is MUTABLE and moved once already
# between the community recipe's 08-28 build and 09-03; three ranks on three
# different day-0 builds can disagree on shard shapes and present as a hang or
# as silently wrong output, so every node builds from THIS digest.
SGLANG_BASE="${SGLANG_BASE:-lmsysorg/sglang@sha256:5ae5816783d58e2e56e84d2e863f5441425056f500b7fbd7448c4aae017a2521}"
echo "=== Pinned base (identical on every node) ==="
echo "  SGLANG_BASE=$SGLANG_BASE"

BUILD_ARGS=(--build-arg "SGLANG_BASE=$SGLANG_BASE")

echo "=== Building $IMAGE_TAG on $NODE0 (local) ==="
cd "$REPO_ROOT"
docker build "${BUILD_ARGS[@]}" -t "$IMAGE_TAG" -f docker/Dockerfile.sglang-flashnext-tp3 .

echo "=== Syncing and building on worker nodes ==="
for n in "$NODE1" "$NODE2"; do
  echo "  Syncing to $n..."
  ssh -o ConnectTimeout=10 -o BatchMode=yes "$n" "mkdir -p $BUILD_DIR_REMOTE/patches/sglang_upstream $BUILD_DIR_REMOTE/scripts $BUILD_DIR_REMOTE/docker"
  scp -q "$REPO_ROOT/patches/sglang_upstream/enable_unaligned_tp_on_cuda.py" "$n:$BUILD_DIR_REMOTE/patches/sglang_upstream/"
  scp -q "$REPO_ROOT/scripts/audit_unaligned_tp_gates.py" "$REPO_ROOT/scripts/verify_unaligned_tp_shims.py" "$n:$BUILD_DIR_REMOTE/scripts/"
  scp -q "$REPO_ROOT/docker/Dockerfile.sglang-flashnext-tp3" "$n:$BUILD_DIR_REMOTE/docker/"
  ssh -o ConnectTimeout=10 -o BatchMode=yes "$n" \
    "cd $BUILD_DIR_REMOTE && docker build --build-arg SGLANG_BASE='$SGLANG_BASE' -t '$IMAGE_TAG' -f docker/Dockerfile.sglang-flashnext-tp3 ."
done

# Every rank must run the SAME ENGINE. Compare by CONTENT (image IDs differ per
# node from build metadata alone). Probe goes over STDIN, never inline -- see the
# dense repo's build.sh for the empty-md5 trap that motivated this.
echo "=== Verifying engine identity across nodes (by content) ==="
PROBE=$(mktemp)
trap 'rm -f "$PROBE"' EXIT
cat > "$PROBE" <<'INNER'
set -eu
SGL=/sgl-workspace/sglang/python/sglang/srt
md5sum \
  $SGL/model_executor/model_runner.py \
  $SGL/layers/vocab_parallel_embedding.py \
  $SGL/configs/update_config.py \
  $SGL/models/qwen3_5.py \
  $SGL/model_loader/weight_utils.py \
  $SGL/layers/attention/mamba/mamba.py \
  $SGL/layers/parameter.py \
  $SGL/layers/moe/fused_moe_triton/layer.py \
  $SGL/models/qwen3_vl.py \
  $SGL/layers/linear.py \
  $SGL/models/qwen4_exp.py \
  $SGL/models/qwen4_exp_mtp.py | awk '{print $1}'
python3 -c 'import sglang; print(sglang.__version__)'
readlink -f /opt/sglang/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2
cat /etc/sglang-flashnext-tp3-provenance | grep -v built_utc
INNER

fingerprint_of() {  # fingerprint_of <host|LOCAL>
  local host="$1" out
  if [ "$host" = LOCAL ]; then
    out=$(docker run --rm -i --entrypoint sh "$IMAGE_TAG" < "$PROBE" 2>/dev/null)
  else
    out=$(ssh -o ConnectTimeout=10 -o BatchMode=yes "$host" "docker run --rm -i --entrypoint sh '$IMAGE_TAG'" < "$PROBE" 2>/dev/null)
  fi
  [ -n "$out" ] || return 1
  printf '%s' "$out" | md5sum | awk '{print $1}'
}

ref_fp=$(fingerprint_of LOCAL) || { echo "FATAL: could not fingerprint $IMAGE_TAG on $NODE0 (probe returned nothing)" >&2; exit 1; }
echo "  $NODE0: $ref_fp"
mismatch=0
for n in "$NODE1" "$NODE2"; do
  if fp=$(fingerprint_of "$n"); then
    echo "  $n: $fp"; [ "$fp" = "$ref_fp" ] || mismatch=1
  else
    echo "  $n: UNREACHABLE or image missing"; mismatch=1
  fi
done
if [ "$mismatch" -ne 0 ]; then
  echo "FATAL: the engine differs across nodes. Do NOT launch or benchmark; rebuild with the pinned digest." >&2
  exit 1
fi
echo "=== $IMAGE_TAG READY and IDENTICAL on all nodes (fingerprint $ref_fp) ==="
