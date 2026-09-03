#!/usr/bin/env bash
# qwen-next-sweep.sh — benchmark Qwen 3.8 Flash Next across concurrency grid c ∈ {1, 4, 8, 16}
#
# Policy-compliant per docs/BENCHMARK-POLICY.md (ported 2026-09-03):
#   - req 2: forces --output-tokens 256 explicitly. The harness default is
#     128 -- silently relying on it is the exact defect that voided the
#     SGLang TP2/TP3 campaign in the sibling dense-model repo. bench-miaai.py
#     itself asserts completion_tokens == output_tokens and aborts on
#     collapse, so this script only has to stop passing the short default.
#   - req 3: rows.tsv now carries decode_min/decode_max alongside the median.
#   - req 5: exclusivity.py brackets the whole sweep (idle check before,
#     delta verify after) and per-c requests are counted as repeat*c.
# req 1 (fabric gate) is NOT yet wired -- fabric_gate.sh has not been ported
# to this repo (see docs/BENCHMARK-POLICY.md). req 4 (live-process config)
# is the caller's responsibility -- capture `docker inspect`/`ps -eo args`
# separately and commit it in the bundle README.
set -euo pipefail

NODES_COUNT="${1:-3}"
MNBT="${2:-8192}"
OUT="${3:-$HOME/results-flash-next-tp${NODES_COUNT}}"
OUTPUT_TOKENS="${OUTPUT_TOKENS:-256}"

BASE_URL="http://127.0.0.1:8100/v1"
METRICS_URL="http://127.0.0.1:8100/metrics"
MODEL=$(curl -s http://127.0.0.1:8100/v1/models | jq -r ".data[0].id // \"qwen3.8-flash-next-nvfp4\"")
HARNESS="$HOME/bench-miaai.py"
EXCL="$(dirname "$0")/exclusivity.py"
mkdir -p "$OUT"

[ -r "$HARNESS" ] || { echo "FATAL: harness not found at $HARNESS" >&2; exit 2; }

echo "=== Qwen Flash Next Benchmark Sweep: model=$MODEL nodes=$NODES_COUNT mnbt=$MNBT out_tokens=$OUTPUT_TOKENS -> $OUT ==="

# req 5: refuse to start on a cluster already serving someone else's traffic.
echo "--- exclusivity: idle check ---"
IDLE_LINE=$(python3 "$EXCL" --url "$METRICS_URL" --check-idle --timeout 30) \
  || { echo "FATAL: cluster not idle -- $IDLE_LINE" >&2; exit 1; }
echo "  $IDLE_LINE"
START_TOTAL=$(printf '%s\n' "$IDLE_LINE" | sed -E 's/.*start_request_success_total=([0-9.]+).*/\1/')
TOTAL_EXPECTED_REQUESTS=0

# Warmup. Cold-cache numbers decay monotonically as kernels JIT; without this
# the first cells are lower bounds and the arm reads as a loss.
#
# Warmup requests count toward request_success_total exactly like measured
# ones -- the exclusivity baseline was captured before warmup, so warmup's
# own traffic must be added to the expected total or it reads as "foreign"
# contamination. Caught the hard way: 2*1 + 2*4 = 10 warmup requests showed
# up as exactly 10 unexplained foreign requests on the first policy-wired run.
echo "--- warmup (discarded) ---"
for c in 1 4; do
  python3 "$HARNESS" --base-url "$BASE_URL" --model "$MODEL" \
    --prompt 256 --concurrency "$c" --repeat 2 --output-tokens "$OUTPUT_TOKENS" \
    > "$OUT/warmup-c${c}.log" 2>&1 || true
  TOTAL_EXPECTED_REQUESTS=$((TOTAL_EXPECTED_REQUESTS + 2 * c))
done
sleep 3

: > "$OUT/rows.tsv"
printf 'nodes\tmnbt\tc\tmedian_decode_tok_s\tagg_tok_s\tttft_ms\tdecode_min\tdecode_max\n' >> "$OUT/rows.tsv"

med() { sort -n | awk '{v[NR]=$1} END{if(NR)print (NR%2)?v[(NR+1)/2]:(v[NR/2]+v[NR/2+1])/2; else print 0}'; }

for c in 1 4 8 16; do
  echo "--- measuring c=$c (5 trials)"
  python3 "$HARNESS" --base-url "$BASE_URL" --model "$MODEL" \
    --prompt 256 --concurrency "$c" --repeat 5 --output-tokens "$OUTPUT_TOKENS" \
    > "$OUT/bench-c${c}.log" 2>&1
  TOTAL_EXPECTED_REQUESTS=$((TOTAL_EXPECTED_REQUESTS + 5 * c))

  dec=$(grep '^FINAL' "$OUT/bench-c${c}.log" | sed -E 's/.*= ([0-9.]+) tok.*/\1/' || echo "0")
  per=$(grep '^trial' "$OUT/bench-c${c}.log" | sed -E 's/.*median_decode=([0-9.]+).*/\1/' | grep -E '^[0-9.]+$' || true)
  agg=$(grep '^trial' "$OUT/bench-c${c}.log" | sed -E 's/.*agg=([0-9.]+).*/\1/' | med)
  ttft=$(grep '^trial' "$OUT/bench-c${c}.log" | sed -E 's/.*ttft=([0-9]+)ms.*/\1/' | med)
  lo=$(printf '%s\n' "$per" | sort -n | head -1); hi=$(printf '%s\n' "$per" | sort -n | tail -1)

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$NODES_COUNT" "$MNBT" "$c" "${dec:-0}" "${agg:-0}" "${ttft:-0}" "${lo:-NA}" "${hi:-NA}" >> "$OUT/rows.tsv"
  echo "    c=$c decode=$dec agg=$agg ttft=${ttft}ms spread=[${lo:-NA}, ${hi:-NA}]"
done

# req 5: verify nothing else touched the engine during the whole sweep.
echo "--- exclusivity: verify ---"
python3 "$EXCL" --url "$METRICS_URL" --verify \
  --start-total "$START_TOTAL" --expected "$TOTAL_EXPECTED_REQUESTS" \
  --out "$OUT/exclusivity.json" \
  || { echo "FATAL: exclusivity check failed -- see $OUT/exclusivity.json" >&2; exit 1; }
echo "  -> $OUT/exclusivity.json"

echo "=== Benchmark Complete: $OUT/rows.tsv ==="
cat "$OUT/rows.tsv"
