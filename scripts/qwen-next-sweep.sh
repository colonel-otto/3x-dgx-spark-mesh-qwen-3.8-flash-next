#!/usr/bin/env bash
# qwen-sweep.sh — benchmark Qwen 3.8 across concurrency grid c ∈ {1, 4, 8, 16}
set -euo pipefail

NODES_COUNT="${1:-3}"
MNBT="${2:-8192}"
OUT="${3:-$HOME/qwen-sweep-tp${NODES_COUNT}-mnbt${MNBT}}"

BASE_URL="http://127.0.0.1:8100/v1"
MODEL="qwen3.8-27b-nvfp4"
HARNESS="$HOME/bench-miaai.py"
mkdir -p "$OUT"

echo "=== Qwen Benchmark Sweep: nodes=$NODES_COUNT mnbt=$MNBT -> $OUT ==="

# Warmup
echo "--- warmup ---"
for c in 1 4; do
  python3 "$HARNESS" --base-url "$BASE_URL" --model "$MODEL" \
    --prompt 256 --concurrency "$c" --repeat 2 > "$OUT/warmup-c${c}.log" 2>&1 || true
done
sleep 3

# Measure
: > "$OUT/rows.tsv"
printf 'nodes\tmnbt\tc\tmedian_decode_tok_s\tagg_tok_s\tttft_ms\n' >> "$OUT/rows.tsv"

for c in 1 4 8 16; do
  echo "--- measuring c=$c (5 trials)"
  python3 "$HARNESS" --base-url "$BASE_URL" --model "$MODEL" \
    --prompt 256 --concurrency "$c" --repeat 5 > "$OUT/bench-c${c}.log" 2>&1

  dec=$(grep '^FINAL' "$OUT/bench-c${c}.log" | sed -E 's/.*= ([0-9.]+) tok.*/\1/' || echo "0")
  agg=$(grep '^trial' "$OUT/bench-c${c}.log" | sed -E 's/.*agg=([0-9.]+).*/\1/' \
        | sort -n | awk '{v[NR]=$1} END{print (NR%2)?v[(NR+1)/2]:(v[NR/2]+v[NR/2+1])/2}' || echo "0")
  ttft=$(grep '^trial' "$OUT/bench-c${c}.log" | sed -E 's/.*ttft=([0-9]+)ms.*/\1/' \
        | sort -n | awk '{v[NR]=$1} END{print (NR%2)?v[(NR+1)/2]:(v[NR/2]+v[NR/2+1])/2}' || echo "0")
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$NODES_COUNT" "$MNBT" "$c" "$dec" "$agg" "$ttft" >> "$OUT/rows.tsv"
  echo "    c=$c decode=$dec agg=$agg ttft=${ttft}ms"
done

echo "=== Benchmark Complete: $OUT/rows.tsv ==="
cat "$OUT/rows.tsv"
