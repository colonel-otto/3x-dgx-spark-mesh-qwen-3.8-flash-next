#!/usr/bin/env bash
# G1 no-build probe: vLLM TP=2 (production recipe) with the two regressions TP=3 would force.
# Arms: hc0 (HC replicated), roce0 (NCCL all-reduce), both. Then relaunch SGLang TP=3.
set -u
cd "$HOME"
L=$HOME/g1-probe-20260929.log
exec >> "$L" 2>&1
NODES="192.168.10.10 192.168.10.11 192.168.10.12"
say() { echo "[$(date -u +%H:%M:%S)] $*"; }

stop_all() {
  for n in $NODES; do ssh -o BatchMode=yes -o ConnectTimeout=10 $n "docker rm -f sglang_node vllm_node >/dev/null 2>&1" </dev/null; done
  pid=$(cat ~/.qwen-launcher.pid 2>/dev/null); [ -n "$pid" ] && ps -p "$pid" -o args= 2>/dev/null | grep -q run-recipe && kill "$pid" 2>/dev/null
  sleep 5
}

R=$HOME/eugr-launcher/recipes
BASE=$R/colonel-qwen3.8-flash-next-nvfp4-tp2-pinned.yaml
mk() {  # name, python-snippet-free sed expr list
  name=$1; shift
  cp "$BASE" "$R/g1-$name.yaml"
  for e in "$@"; do sed -i "$e" "$R/g1-$name.yaml"; done
}
mk hc0    's/^  VLLM_ENABLE_ROCE_ALLREDUCE: "1"/  VLLM_ENABLE_ROCE_ALLREDUCE: "1"\n  VLLM_QWEN3_8_FLASH_NEXT_HC_TP: "0"/'
mk roce0  's/^  VLLM_ENABLE_ROCE_ALLREDUCE: "1"/  VLLM_ENABLE_ROCE_ALLREDUCE: "0"/'
mk both   's/^  VLLM_ENABLE_ROCE_ALLREDUCE: "1"/  VLLM_ENABLE_ROCE_ALLREDUCE: "0"\n  VLLM_QWEN3_8_FLASH_NEXT_HC_TP: "0"/'
for a in hc0 roce0 both; do say "recipe g1-$a env:"; grep -E "ROCE_ALLREDUCE:|HC_TP" "$R/g1-$a.yaml"; done

run_arm() {
  arm=$1
  say "=== ARM $arm: stop everything, launch"
  stop_all
  ( cd "$HOME/eugr-launcher" && setsid nohup python3 run-recipe.py "g1-$arm" \
      -t vllm-node-b12x -n 192.168.10.10,192.168.10.11 \
      -v /etc/nccl-ib-hca:/etc/nccl-ib-hca \
      > "$HOME/g1-$arm-launch.log" 2>&1 < /dev/null & echo $! > ~/.qwen-launcher.pid )
  ok=0
  for i in $(seq 1 120); do
    if curl -fsS -m 4 http://127.0.0.1:8100/v1/models 2>/dev/null | grep -q qwen3.8; then ok=1; say "ARM $arm serving after ~$((i*10))s"; break; fi
    pid=$(cat ~/.qwen-launcher.pid); kill -0 "$pid" 2>/dev/null || { say "ARM $arm launcher exited"; break; }
    sleep 10
  done
  if [ "$ok" = 1 ]; then
    # quick correctness gate before timing
    ans=$(curl -s -m 60 http://127.0.0.1:8100/v1/chat/completions -H "Content-Type: application/json" \
      -d '{"model":"qwen3.8-flash-next","messages":[{"role":"user","content":"17*23? number only"}],"max_tokens":16}' \
      | python3 -c "import sys,json;print(json.load(sys.stdin)['choices'][0]['message']['content'].strip())" 2>&1)
    say "ARM $arm sanity answer: [$ans]"
    "$HOME/3spark-qwen-flash-build/scripts/qwen-next-sweep.sh" 2 8192 "$HOME/results-g1-$arm-20260929" > "$HOME/g1-$arm-sweep.log" 2>&1 < /dev/null
    grep -E "IDLE|EXCLUSIVITY" "$HOME/g1-$arm-sweep.log"
    say "ARM $arm rows:"; cat "$HOME/results-g1-$arm-20260929/rows.tsv"
  else
    say "ARM $arm FAILED to serve; tail of launch log:"; grep -nE "Error|Traceback|error:" "$HOME/g1-$arm-launch.log" | tail -8 | cut -c1-260
  fi
  stop_all
}

for arm in hc0 roce0 both; do run_arm "$arm"; done

say "=== restoring SGLang TP=3 (cg16, 3/4, 600000)"
cd "$HOME/3spark-qwen-flash-build" && MAX_TOTAL_TOKENS=600000 setsid ./scripts/sglang-flashnext-boot-cg16.sh 3 "$HOME/sglang-flashnext-tp3-restored-20260929.log" < /dev/null > "$HOME/sglang-boot-restored-wrapper.out" 2>&1
for i in $(seq 1 150); do
  curl -fsS -m 4 http://127.0.0.1:8100/v1/models 2>/dev/null | grep -q qwen3.8 && { say "SGLang serving again after ~$((i*10))s"; break; }
  sleep 10
done
say "G1 DONE"
