#!/usr/bin/env bash
# sglang-flashnext-validate.sh — correctness gate for the SGLang Flash-Next arm.
# Ported 2026-09-04 from 3spark-qwen-3.8-27b-dense/scripts/sglang-validate.sh.
#
# MANDATORY before recording any SGLang throughput number, and doubly so at
# TP=3: an unpadded 3-way shard can serve fluent nonsense rather than failing
# loudly. Mirrors scripts/qwen-quick-validate.sh so both engines face the
# same battery.
set -uo pipefail

BASE="${1:-http://127.0.0.1:8100/v1}"
MODEL="${2:-qwen3.8-flash-next}"
pass=0; fail=0

# Budgets are deliberately generous (512). This model emits a reasoning
# preamble even with thinking:false, so a 16-token budget truncates before
# the answer and every check fails with an empty string.
ask() {  # ask <prompt> <max_tokens>
  curl -s -m 180 "$BASE/chat/completions" -H 'Content-Type: application/json' \
    -d "$(python3 -c '
import json,sys
print(json.dumps({"model":sys.argv[1],"messages":[{"role":"user","content":sys.argv[2]}],
"max_tokens":int(sys.argv[3]),"temperature":0.0,
"chat_template_kwargs":{"enable_thinking":False}}))' "$MODEL" "$1" "$2")" \
  | python3 -c '
import json,sys
try:
    d=json.load(sys.stdin)
    m=d["choices"][0]["message"]
    # A DSv4-class trap: reading "content" alone makes a working reply look
    # empty when the budget went to reasoning. Note the field is "reasoning"
    # on this stack -- "reasoning_content" is the name that does NOT exist,
    # and reading only it yields a silent empty string. Verified live
    # 2026-09-03 against vllm-0.1.dev20133.
    parts=[m.get("content"), m.get("reasoning"), m.get("reasoning_content")]
    print(" ".join(x for x in parts if x).strip())
except Exception as e:
    print(f"__ERROR__ {e}")'
}

check() {  # check <label> <regex> <output>
  if printf '%s' "$3" | grep -qiE "$2"; then
    echo "PASS  $1"; pass=$((pass+1))
  else
    echo "FAIL  $1"
    echo "      got: $(printf '%s' "$3" | head -c 200)"
    fail=$((fail+1))
  fi
}

echo "=== SGLang Validation Battery ($BASE) ==="
echo "== 1. Models endpoint =="
ids=$(curl -s -m 20 "$BASE/models" | python3 -c '
import json,sys
try: print(",".join(m["id"] for m in json.load(sys.stdin)["data"]))
except Exception: print("__ERROR__")')
check "models endpoint serves $ids" "qwen" "$ids"

echo "== 2. Core reasoning & correctness =="
check "capital lookup (Paris)"   "paris" "$(ask 'What is the capital of France? Answer with one word.' 512)"
check "17 x 23 = 391"            "391"   "$(ask 'What is 17 multiplied by 23? Reply with only the number.' 512)"
check "logical deduction (No)"   "\bno\b" "$(ask 'All cats are mammals. Some mammals are dogs. Does it follow that some cats are dogs? Answer Yes or No.' 512)"

echo "== 3. Needle in haystack =="
needle=$(python3 -c '
filler = "The quarterly logistics review covered routine warehouse metrics. "
print(filler*30 + "The access code is OPAL-4482. " + filler*30 +
      "\nWhat is the access code? Reply with only the code.")')
check "needle ~1.5k tok (OPAL-4482)" "OPAL-4482" "$(ask "$needle" 512)"

echo "== 4. Text quality & degeneration =="
out=$(ask 'Write three sentences about the ocean.' 120)
ratio=$(printf '%s' "$out" | python3 -c '
import sys,re
w=re.findall(r"[a-z]+", sys.stdin.read().lower())
print(f"{len(set(w))/len(w):.3f}" if w else "0")')
# A shard serving nonsense still produces high lexical variety, so pair the
# ratio with a real-word check.
english=$(printf '%s' "$out" | grep -ciE '\b(the|and|of|is|are|to|in|it)\b' || true)
if awk "BEGIN{exit !($ratio > 0.5)}" && [ "$english" -gt 0 ]; then
  echo "PASS  no degeneration (unique-word ratio: $ratio)"; pass=$((pass+1))
else
  echo "FAIL  degeneration or non-English output (ratio: $ratio, english-hits: $english)"
  echo "      got: $(printf '%s' "$out" | head -c 200)"
  fail=$((fail+1))
fi

echo
if [ "$fail" -eq 0 ]; then
  echo "=== Result: ALL ACCEPTANCE TESTS PASSED ($pass PASS / 0 FAIL) ==="
  exit 0
fi
echo "=== Result: GATE FAILED ($pass PASS / $fail FAIL) ==="
echo "Do NOT record throughput numbers from this deployment."
exit 1
