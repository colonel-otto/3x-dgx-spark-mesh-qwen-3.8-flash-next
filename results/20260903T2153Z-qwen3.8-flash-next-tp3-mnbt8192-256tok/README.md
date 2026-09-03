# Qwen 3.8 Flash Next — TP=3, mnbt=8192, 256-token window (2026-09-03)

## What this is

A re-run of the 3-node TP=3 Flash-Next concurrency sweep, with two changes
from the earlier `20260902-tp3-mnbt8192` bundle:

1. **Config adjustment** (the "adjust" half of the requested re-run):
   `max_model_len` 32,768 → **262,144** (full native context), `gpu-memory-
   utilization` 0.80 → **0.82**, plus a new `kernel_warmup.py` patch mount
   (`/opt/qwen-patches/kernel_warmup.py`) not present in the prior boot.
2. **Policy compliance** (found missing while preparing this run — see
   `docs/BENCHMARK-POLICY.md`, ported into this repo for the first time
   today): the harness (`~/bench-miaai.py` on sparkmain) was silently running
   its **128-token default** completion window with no flag to change it and
   no assertion that the window held — the exact defect class that voided
   the SGLang TP2/TP3 campaign in the sibling dense-model repo the same week.
   Deployed the DSv4-fixed harness (forces `min_tokens==max_tokens`,
   `ignore_eos`, aborts on `WindowCollapse`) and made `qwen-next-sweep.sh`
   pass `--output-tokens 256` explicitly. Also wired in `exclusivity.py`
   (ported from DSv4/dense repo) so the sweep asserts nobody else touched the
   engine during the measurement window.

**Consequence: this bundle is NOT directly comparable to
`20260902-tp3-mnbt8192`.** That run's actual decode-window length was never
recorded (no `output_tokens_actual` field, no flag existed to set it) — it
almost certainly used the same silent 128-token default this run explicitly
overrides to 256. Decode tok/s is not invariant to window length. Treat the
two bundles as separate data points, not a matched before/after.

## A false exclusivity failure, and why it doesn't affect the published numbers

The first attempt at this sweep (`results-flash-next-tp3`, not committed)
**failed its own new exclusivity check**: `expected 145, got delta 155, 10
foreign requests detected`. Investigation traced this to a script bug, not
real contamination — `qwen-next-sweep.sh` captured the idle/start counter
*before* the warmup phase but only accumulated `TOTAL_EXPECTED_REQUESTS`
during the *measured* c=1/4/8/16 loop. Warmup itself issues real requests
against the same counter (`repeat=2` × c∈{1,4} = 2+8 = 10), which is exactly
the "foreign" delta observed. Fixed in `scripts/qwen-next-sweep.sh` (warmup
requests now counted) and re-run clean: `EXCLUSIVITY_PASS delta=155
expected=155`. The failed run's throughput numbers were consistent with this
clean re-run's (c=1: 40.4 vs 40.9, c=4: 30.5 vs 30.8, c=8: 26.2 vs 25.8,
c=16: 20.2 vs 20.2) — strong evidence the false exclusivity failure did not
correspond to any actual measurement contamination — but per
`docs/BENCHMARK-POLICY.md` an unverified run is not published, hence the
clean re-run rather than retroactively trusting the failed one.

## Configuration (captured from the live process, not the recipe file)

```
vllm serve RadixArk/Qwen3.8-Flash-Next-NVFP4 \
  --host 0.0.0.0 --port 8100 \
  --served-model-name qwen3.8-flash-next-nvfp4 qwen3.8-flash-next \
  --tensor-parallel-size 3 --trust-remote-code \
  --kv-cache-dtype fp8 --moe-backend b12x --attention-backend flashinfer \
  --gpu-memory-utilization 0.82 --max-model-len 262144 \
  --max-num-seqs 16 --max-num-batched-tokens 8192 \
  --enable-chunked-prefill --async-scheduling --enable-prefix-caching \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
  --load-format safetensors \
  --reasoning-parser qwen3 --tool-call-parser qwen3_xml --enable-auto-tool-choice \
  --max-cudagraph-capture-size 32 \
  --nnodes 3 --node-rank 0 --master-addr 192.168.10.1 --master-port 29501
```

`GPU KV cache size: 9,401,066 tokens` (from boot log; concurrency headroom
35.86x at 262,144 tokens/request) — up substantially from the 32K-context
prior run due to the smaller per-request context reservation ratio, not a
memory-fraction change.

## Correctness gate

5/5 acceptance tests passed both before the sweep (`gate.log`, part of
`qwen-next-sweep.sh`'s own pre-flight) and again standalone after
(this bundle's `gate.log`), confirming the engine stayed healthy through the
full measurement window.

## Results

| Nodes / TP | MNBT | Out tokens | c | Median decode (tok/s/user) | [min, max] | Aggregate (tok/s) | TTFT (ms) |
|---|---|---|---|---|---|---|---|
| 3 / TP=3 | 8192 | 256 | 1  | 40.9 | [38.1, 41.9] | 39.5  | 220  |
| 3 / TP=3 | 8192 | 256 | 4  | 30.8 | [29.9, 31.4] | 113.6 | 600  |
| 3 / TP=3 | 8192 | 256 | 8  | 25.8 | [25.4, 26.3] | 185.0 | 912  |
| 3 / TP=3 | 8192 | 256 | 16 | 20.2 | [20.1, 20.3] | 279.8 | 1631 |

Spread is tight at every cell (widest is c=1 at ~9% of median) — no sign of
the JIT/cold-cache decay pattern that has previously required a warmup pass
to avoid understating throughput; the 2-trial warmup already discards that.

Exclusivity: `EXCLUSIVITY_PASS delta=155 expected=155` (`exclusivity.json`).
No fabric gate was run (`fabric_gate.sh` not yet ported to this repo — see
`docs/BENCHMARK-POLICY.md` requirement 1, still open).

## Honest read

- Aggregate throughput scales up through c=16 (39.5 → 279.8 tok/s), same
  shape as the prior TP=3 result, consistent with a healthy, non-degraded
  fabric under this workload.
- Per-stream decode degrades monotonically with concurrency (40.9 → 20.2),
  as expected for shared GPU compute across streams.
- No TP=2 counterpart exists yet in this repo at the 256-token window /
  adjusted config — `qwen-next-boot-tp2.sh` still runs the old 32K/0.80
  settings by deliberate choice (this re-run's scope was TP=3 only). A
  TP=2-vs-TP=3 comparison at matched settings is not available from this
  bundle alone.
- This is a single sweep, not yet repeat-day confirmed (`reps_verified:
  false` in `results/index.yaml`).

## Provenance

- Harness: `scripts/qwen-next-sweep.sh` (fixed for `--output-tokens`,
  exclusivity, and the warmup-counting bug described above).
- Raw files: `bench-c{1,4,8,16}.log`, `warmup-c{1,4}.log`, `rows.tsv`,
  `exclusivity.json`, `gate.log`.
- Boot: `scripts/qwen-next-boot-tp3.sh` (long-context + kernel_warmup patch
  edits already staged in the working tree at the start of this task).
