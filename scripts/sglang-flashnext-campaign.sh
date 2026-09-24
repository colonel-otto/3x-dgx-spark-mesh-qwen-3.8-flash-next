#!/usr/bin/env bash
# sglang-flashnext-campaign.sh -- UNATTENDED TP=2 vs TP=3 campaign for
# RadixArk/Qwen3.8-Flash-Next-NVFP4 on SGLang, 3x DGX Spark.
#
# Run on sparkmain under nohup (a dropped SSH session must not kill it):
#   nohup ~/3spark-qwen-flash-build/scripts/sglang-flashnext-campaign.sh \
#       > ~/campaign-flashnext-$(date -u +%Y%m%dT%H%MZ).log 2>&1 &
#
# For each arm in $ARMS (default "2 3", leaving the 3-node engine serving at
# the end): stop any engine, boot, wait for readiness (container exit = arm
# FAILED, logs collected, next arm), capture the LIVE config (policy req 4),
# run the 6/6 correctness gate (fail = no numbers, arm GATE_FAILED), run the
# policy-compliant sweep (256-token window, exclusivity, min/max), and write
# a bundle under ~/results-flash-next-sglang/<ts>-tp<N>/.
#
# Progress for a disconnected operator: ~/campaign-flashnext-status.txt is
# rewritten at every state change (one line per arm + a CURRENT line).
set -uo pipefail

BUILD="$HOME/3spark-qwen-flash-build"
ARMS="${ARMS:-2 3}"
TS="$(date -u +%Y%m%dT%H%MZ)"
ROOT="$HOME/results-flash-next-sglang"
STATUS="$HOME/campaign-flashnext-status.txt"
READY_TIMEOUT_S="${READY_TIMEOUT_S:-3000}"   # 50 min: 126 GiB load + cold JIT of padded shapes
# Both arms MUST share these (BENCHMARK-POLICY.md req 4). c=16 needs >=16 slots.
export MAX_RUNNING="${MAX_RUNNING:-16}"
export MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-600000}"
export MEM_FRACTION="${MEM_FRACTION:-0.80}"
export MAX_MAMBA_CACHE="${MAX_MAMBA_CACHE:-97}"
export OUTPUT_TOKENS=256

[ -f "$HOME/.eugr-nodes" ] && . "$HOME/.eugr-nodes"
NODE0="${NODE0:?}"; NODE1="${NODE1:?}"; NODE2="${NODE2:?}"
mkdir -p "$ROOT"
declare -A STATE

log()   { echo "[$(date -u +%H:%M:%SZ)] $*"; }
status() {
  {
    echo "campaign=$TS updated=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    for a in $ARMS; do echo "tp$a=${STATE[$a]:-PENDING}"; done
    echo "CURRENT=$*"
  } > "$STATUS"
}

stop_engine() {
  if [ -r "$HOME/.sglang-flashnext-logtail.pid" ]; then
    kill "$(cat "$HOME/.sglang-flashnext-logtail.pid")" 2>/dev/null || true
    rm -f "$HOME/.sglang-flashnext-logtail.pid"
  fi
  for n in $NODE0 $NODE1 $NODE2; do
    ssh -o ConnectTimeout=10 -o BatchMode=yes "$n" "docker rm -f sglang_node >/dev/null 2>&1 || true"
  done
  sleep 5
}

nodes_for() { if [ "$1" = 3 ]; then echo "$NODE0 $NODE1 $NODE2"; else echo "$NODE0 $NODE1"; fi; }

collect_logs() {  # collect_logs <tp> <bundle>
  for n in $(nodes_for "$1"); do
    ssh -o ConnectTimeout=10 -o BatchMode=yes "$n" \
      "docker logs sglang_node 2>&1 | grep -v -E 'rope_parameters|UserWarning|warnings.warn' | tail -400" \
      > "$2/rank-log-$n.txt" 2>/dev/null || true
  done
}

wait_ready() {  # wait_ready <tp> <bundle> ; returns 0 ready, 1 failed, 2 timeout
  local tp="$1" bundle="$2" t0=$(date +%s)
  while :; do
    for n in $(nodes_for "$tp"); do
      st=$(ssh -o ConnectTimeout=10 -o BatchMode=yes "$n" "docker ps -a --filter name=^sglang_node\$ --format '{{.Status}}'" 2>/dev/null)
      case "$st" in
        Exited*|"") log "rank on $n is '$st' -> arm FAILED"; return 1 ;;
      esac
    done
    code=$(curl -s -m 5 -o /dev/null -w '%{http_code}' http://127.0.0.1:8100/v1/models || true)
    if [ "$code" = 200 ]; then
      # HTTP 200 is not health on this model: demand a real token.
      out=$(curl -s -m 300 http://127.0.0.1:8100/v1/chat/completions -H 'Content-Type: application/json' \
        -d '{"model":"qwen3.8-flash-next","messages":[{"role":"user","content":"Say OK."}],"max_tokens":8,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}' \
        | python3 -c 'import json,sys; d=json.load(sys.stdin); m=d["choices"][0]["message"]; print((m.get("content") or m.get("reasoning") or "").strip())' 2>/dev/null || true)
      if [ -n "$out" ]; then log "first completion: '$out'"; return 0; fi
    fi
    if [ $(( $(date +%s) - t0 )) -ge "$READY_TIMEOUT_S" ]; then return 2; fi
    sleep 20
  done
}

capture_live_config() {  # capture_live_config <tp> <bundle>
  local tp="$1" bundle="$2"
  ssh -o ConnectTimeout=10 -o BatchMode=yes "$NODE0" "docker inspect sglang_node --format '{{json .Config.Cmd}}'" > "$bundle/live-cmd.json" 2>/dev/null
  ssh -o ConnectTimeout=10 -o BatchMode=yes "$NODE0" "docker inspect sglang_node --format '{{json .Config.Env}}'" > "$bundle/live-env.json" 2>/dev/null
  ssh -o ConnectTimeout=10 -o BatchMode=yes "$NODE0" "docker exec sglang_node cat /etc/sglang-flashnext-tp3-provenance" > "$bundle/image-provenance.txt" 2>/dev/null
  ssh -o ConnectTimeout=10 -o BatchMode=yes "$NODE0" "docker logs sglang_node 2>&1 | grep -E 'max_total_num_tokens|Mamba Cache|KV Cache|max_running_requests|ZERO-PADDING|Updating .*(heads|intermediate)|sglang is using nccl|Capture cuda graph end|Load weight end'" > "$bundle/boot-key-lines.txt" 2>/dev/null
  for n in $(nodes_for "$tp"); do
    ssh -o ConnectTimeout=10 -o BatchMode=yes "$n" "echo '== $n'; free -g | sed -n 1,2p" >> "$bundle/memory-after-boot.txt" 2>/dev/null
  done
  # Live-process check of the shared knobs (policy req 4).
  grep -o -E '"--max-running-requests","[0-9]+"|"--max-total-tokens","[0-9]+"|"--mem-fraction-static","[0-9.]+"|"--max-mamba-cache-size","[0-9]+"|"--tp-size","[0-9]+"' "$bundle/live-cmd.json" | tr '\n' ' ' > "$bundle/live-knobs.txt"; echo >> "$bundle/live-knobs.txt"
  log "live knobs: $(cat "$bundle/live-knobs.txt")"
}

run_arm() {  # run_arm <tp>
  local tp="$1"
  local bundle="$ROOT/$TS-tp$tp"
  mkdir -p "$bundle"
  STATE[$tp]=BOOTING; status "tp$tp booting (bundle $bundle)"
  stop_engine
  log "=== ARM TP=$tp: boot ==="
  "$BUILD/scripts/sglang-flashnext-boot.sh" "$tp" "$bundle/rank0-boot.log" > "$bundle/boot-launcher.log" 2>&1 \
    || { STATE[$tp]=LAUNCH_FAILED; status "tp$tp launch failed"; cat "$bundle/boot-launcher.log"; return; }
  wait_ready "$tp" "$bundle"; rc=$?
  collect_logs "$tp" "$bundle"
  if [ $rc -ne 0 ]; then
    STATE[$tp]=$([ $rc = 1 ] && echo BOOT_FAILED || echo BOOT_TIMEOUT); status "tp$tp $rc"
    log "arm TP=$tp did not come up (rc=$rc). Last error lines:"
    grep -h -E "RuntimeError|AssertionError|Error:|unrecognized" "$bundle"/rank-log-*.txt | tail -6
    return
  fi
  STATE[$tp]=READY; status "tp$tp ready, capturing config"
  capture_live_config "$tp" "$bundle"

  log "=== ARM TP=$tp: correctness gate ==="
  STATE[$tp]=GATING; status "tp$tp gate"
  if "$BUILD/scripts/sglang-flashnext-validate.sh" http://127.0.0.1:8100/v1 qwen3.8-flash-next > "$bundle/gate.log" 2>&1; then
    log "gate PASSED"; tail -2 "$bundle/gate.log"
  else
    STATE[$tp]=GATE_FAILED; status "tp$tp GATE FAILED -- no numbers"; cat "$bundle/gate.log"; return
  fi

  log "=== ARM TP=$tp: sweep (c=1,4,8,16 x5, 256-tok window) ==="
  STATE[$tp]=SWEEPING; status "tp$tp sweeping"
  if "$BUILD/scripts/qwen-next-sweep.sh" "$tp" 8192 "$bundle" > "$bundle/sweep.log" 2>&1; then
    STATE[$tp]=DONE; status "tp$tp done"
    log "sweep DONE:"; cat "$bundle/rows.tsv"
  else
    STATE[$tp]=SWEEP_FAILED; status "tp$tp sweep failed"; tail -20 "$bundle/sweep.log"
  fi
  # Post-sweep gate: the engine must still be correct after load (MTP-decay,
  # mamba-pool, token-0 loop classes all show up here, not before).
  "$BUILD/scripts/sglang-flashnext-validate.sh" http://127.0.0.1:8100/v1 qwen3.8-flash-next > "$bundle/gate-after.log" 2>&1 \
    && log "post-sweep gate PASSED" || { log "post-sweep gate FAILED -- see gate-after.log"; STATE[$tp]="${STATE[$tp]}+POSTGATE_FAILED"; }
  collect_logs "$tp" "$bundle"
  status "tp$tp finished: ${STATE[$tp]}"
}

log "campaign $TS arms=[$ARMS] MAX_RUNNING=$MAX_RUNNING MAX_TOTAL_TOKENS=$MAX_TOTAL_TOKENS"
for a in $ARMS; do STATE[$a]=PENDING; done
status "starting"
for a in $ARMS; do run_arm "$a"; done
log "=== campaign complete ==="
for a in $ARMS; do log "  tp$a: ${STATE[$a]}  ($ROOT/$TS-tp$a)"; done
status "COMPLETE"
