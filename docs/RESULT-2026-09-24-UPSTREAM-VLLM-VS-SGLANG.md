# RESULT: upstream vLLM (native Qwen4Exp) vs SGLang TP=3 vs our old vLLM port

**Date:** 2026-09-23/24. **Status:** CURRENT -- production changed to the
"rung 2" config below.

## Harness (identical for every arm)
- Single-stream: `scripts/bench/sp.py`/`sp_code.py` -- 3 sequential streaming requests,
  512 max tokens, thinking off, model-default sampling; prose = "400-word
  essay on tides", code = "thread-safe LRU cache with TTL + pytest".
- Concurrency: `scripts/bench/conc.py` -- c simultaneous streaming 400-word essays with
  unique nonce+topic prompts, 512 max tokens, thinking off. Aggregate =
  total completion tokens / wall span.
- Correctness: `scripts/sglang-flashnext-validate.sh` (6-prompt battery).
- Tools: `scripts/test_tool_calling_regression.py` (3 trials x 2 turns).
- One run per cell; no trial spread -- treat <10% deltas as ties.

## Numbers

| arm | nodes | prose tok/s | code tok/s | c1 | c4 agg | c16 agg | battery |
|---|---|---|---|---|---|---|---|
| old vLLM port (Qwen3-Next patches, MTP=1) | 3 | 34-39.5 | -- | -- | -- | -- | -- |
| old vLLM port, MTP=4 | 3 | crashed in warmup (illegal memory access in `_multi_step_decode`) | | | | | |
| SGLang TP=3 (`sglang-flashnext-tp3:local`, NEXTN 3/4) | 3 | 48-51 | 57-67 | 48.3 | 125.3 | 219.7 | 6/6 |
| rung 1: upstream `qwen3.8-flash-next-nvfp4-cluster` unmodified | 2 | 47.5-51 | 69-71 | 46.3 | 111.6 | 211.6 | 6/6 |
| **rung 2: `colonel-qwen3.8-flash-next-nvfp4-tp2` (production)** | 2 | 48.6-50.8 | 71-74 | 51.2 | 116.7 | 221.0 | 6/6 |
| upstream solo recipe (spark-sep) | 1 | 37-41 | 51-56 | 38.6 | 78.5 | 112.3 (c8, max_num_seqs 8) | 6/6 |

Rung 2 extra points: c6 125.2 agg (per-req 21.9-23.6), c8 156.0 agg
(20.0-20.9) -- no batch 5-7 straggler.

Upstream vLLM stack: image `eugr/spark-vllm-b12x:latest` 688d2769
(vLLM `local-inference-lab/vllm@57fdda71` dev/karmic-kraken, b12x 1.3.0,
flashinfer 0.7.0), checkpoint `local-inference-lab/Qwen3.8-Flash-Next-NVFP4`
@ `7c4f1bc1`. B12X autotune is ON by default in b12x 1.3.0.

Rung 2 = rung 1 + served names/port 8100, pinned revision, probabilistic MTP
drafts, `VLLM_MXFP8_LM_HEAD=1`, decode-aware prefill (4 parallel),
8192 batched tokens, `B12X_ROCE_SPIN_LIMIT`. Recipe lives in
`colonel-otto/spark-vllm-docker` branch `colonel/recipes`.

## Reading
- On 2 nodes the upstream stack ties SGLang-on-3 for prose and aggregate
  throughput and is faster for code (+5-20 tok/s) with lower warm TTFT.
- Our per-type single-stream numbers match or beat the best public 2-Spark
  recipe (ursuciprian: prose 49.0, code 61.3).
- TP=3 is not available on the native model (KV heads 2, GDN key heads 16,
  MoE intermediate 640, vocab 248320 do not divide by 3; native code
  enforces divisibility). See `docs/PATCH-INVENTORY.md` section 4.
- The third Spark therefore serves an independent solo instance.

## Tool calling
Rung 1/2 leave thinking on (model default). Turn 1 often returns a short
plan sentence in `content` alongside correct parallel tool calls, e.g.
"I'll start by reading the canon doc and the two design-system files in
parallel." Our harness fails that turn (`content` must be empty), but no
turn was contaminated with the echoed compaction block. The strict rule is
stricter than the OpenAI schema; whether OpenCode mis-handles a preamble is
untested.

## Production layout (2026-09-24)
- `sparkmain:8100` -- rung 2, TP=2 on sparkmain + spark1 (primary).
- `spark-sep:8100` (192.168.10.12) -- solo, upstream solo recipe + served names.
- SGLang TP=3 image and boot script kept as fallback (`scripts/sglang-flashnext-boot.sh 3`, `MAX_TOTAL_TOKENS=600000`).
- Old vLLM image kept as `eugr/spark-vllm-b12x:local-20260823`.
