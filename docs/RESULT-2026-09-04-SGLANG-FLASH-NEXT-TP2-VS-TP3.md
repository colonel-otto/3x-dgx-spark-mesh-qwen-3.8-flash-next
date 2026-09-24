# RESULT: Qwen3.8-Flash-Next on SGLang, 2 nodes vs 3 nodes (and vs the vLLM 3-node arm)

**Date:** 2026-09-04. **Status:** CURRENT, single sweep per arm, same day,
same image, same harness, same prompt, same 256-token window, same knobs.
**Bundles:** `results/20260904T1308Z-sglang-tp2/`, `results/20260904T1355Z-sglang-tp3/`,
vLLM reference `results/20260903T2153Z-qwen3.8-flash-next-tp3-mnbt8192-256tok/`.

## 1. Headline

SGLang serves this model on all three Sparks at TP=3, correctly (6/6 text
battery before and after the sweep, both arms), and the third node pays for
itself on this engine: **+14% single-stream, +43% at c=4, and roughly +45%
per stream at c=8/c=16 over the 2-node arm**, with the c=1 and c=4 wins
outside the trial-to-trial spread and the c=8/c=16 wins inside it. Against
the vLLM 3-node bundle from 2026-09-03 the SGLang 3-node arm is **+85%
single-stream and +37% aggregate at c=16**, but that is a stack comparison,
not an engine comparison (section 4).

Nobody had run this model at TP=3 on SGLang before today; every published
config is a power of two.

## 2. Numbers (median of 5 trials, [min, max] across trials, 256-token decode window, 256-token unique cold prompts)

| arm | c=1 tok/s | c=4 tok/s | c=8 tok/s | c=16 tok/s | c=16 aggregate | c=1 TTFT |
|---|---|---|---|---|---|---|
| **SGLang TP=3** (3 nodes) | **75.7** [73.8, 76.2] | **58.5** [50.5, 64.0] | **46.9** [36.4, 49.2] | **33.3** [28.3, 34.9] | **383.2** (trials 291-450) | 259 ms |
| **SGLang TP=2** (2 nodes) | 66.5 [59.3, 68.8] | 40.9 [39.2, 44.6] | 32.5 [30.6, 42.1] | 22.9 [22.5, 29.6] | 289.2 (trials 208-390) | 266 ms |
| vLLM TP=3 (3 nodes, 09-03) | 40.9 [38.1, 41.9] | 30.8 [29.9, 31.4] | 25.8 [25.4, 26.3] | 20.2 [20.1, 20.3] | 279.8 | 220 ms |

Aggregate tok/s per cell (median of trial aggregates):

| arm | c=1 | c=4 | c=8 | c=16 |
|---|---|---|---|---|
| SGLang TP=3 | 70.2 | 207.3 | 322.4 | 383.2 |
| SGLang TP=2 | 62.2 | 142.9 | 213.1 | 289.2 |
| vLLM TP=3 | 39.5 | 113.6 | 185.0 | 279.8 |

## 3. Noise-floor reading (BENCHMARK-POLICY.md req 3: carry the spread)

| cell | TP=3 vs TP=2 delta | spreads | verdict |
|---|---|---|---|
| c=1 | +14% | [73.8, 76.2] vs [59.3, 68.8], disjoint | real |
| c=4 | +43% | [50.5, 64.0] vs [39.2, 44.6], disjoint | real |
| c=8 | +44% | [36.4, 49.2] vs [30.6, 42.1], overlap | median-only; TP=3 trial 0 (36.4) sits inside TP=2's range |
| c=16 | +45% | [28.3, 34.9] vs [22.5, 29.6], overlap | median-only; both arms bimodal (see below) |

Bimodality: at c=16 both arms alternate between a fast mode (TP=3 ~34 tok/s
/ 445 agg; TP=2 ~29.5 / 390) and a slow mode (TP=3 ~32 / 291-297; TP=2
~22.7 / 208-289). Trial 0 is the slowest at every c>1 on both arms. Two
plausible mechanisms, neither isolated: `--cuda-graph-max-bs 8` means c=16
decodes without CUDA graphs, and NEXTN acceptance varies with the per-nonce
prompt. A repeat with `--cuda-graph-max-bs 16` on both arms is the next
experiment if the c=16 cell matters.

Both SGLang arms are cold-cache lower bounds (fresh containers, persistent
FlashInfer/Triton caches mounted but first-seen padded shapes at TP=3).

## 4. What is and is not being compared

**SGLang TP=2 vs TP=3 is a clean single-variable comparison**: identical
image, flags and knobs, verified from the live process on each arm
(`live-knobs.txt`): `--max-running-requests 16`, `--max-total-tokens 600000`
(TP=2 was capped by memory to 517,248; TP=3 got the full 600,000),
`--mem-fraction-static 0.80`, `--max-mamba-cache-size 97`, NEXTN 3 steps /
4 draft tokens, `--cuda-graph-max-bs 8`, radix cache off, BF16 KV,
flashinfer_cutlass NVFP4 MoE. The only differences are `--tp-size`,
`SGLANG_FORCE_UNALIGNED_TP=1` and `--mm-enable-dp-encoder` on TP=3.

**SGLang vs vLLM is a stack comparison.** Same checkpoint, same harness,
same window, same node count, same `max_num_seqs`/`max_running_requests`
(16), but: drafter (SGLang NEXTN 3-step native MTP vs vLLM MTP K=1), KV
dtype (BF16 vs FP8), MoE kernel (flashinfer_cutlass vs B12X), CUDA-graph
policy (bs<=8, prefill graphs off vs capture size 32), memory fraction
(0.80 vs 0.82), padding plan (MoE 768 vs 672). Quote it as "SGLang stack
vs vLLM stack", never as an engine delta.

## 5. Correctness evidence, and what is still missing

- 6/6 text battery (models endpoint, Paris, 17x23, deduction, 1.5k-token
  needle, degeneration) passed on both arms before AND after the sweep.
- Exclusivity: delta 155 = expected 155 on both arms (SGLang metric names
  now auto-detected by `scripts/exclusivity.py`).
- TP=3 padding confirmed live: 36 `[ZERO-PADDING] 10240 -> 11520` lines
  (GDN in_proj per layer), Q 24->36, KV 2->3, MoE 640->768.
- **NOT done: the logprob gate** (TP=3 vs TP=2 shared-prefix |dlogprob|,
  the dense repo's standard for any padded shard). A padded shard can serve
  fluent, slightly wrong text that a 6-prompt battery passes. Until it runs,
  the TP=3 arm is "correct on the battery", not "numerically gated".
- Fabric gate not ported to this repo (policy req 1 still open).

## 6. Memory and capacity

| arm | weights/rank after load | KV tokens | free after pools |
|---|---|---|---|
| TP=2 | ~72.6 GB (+2 GB MTP) | 517,248 (BF16 K/V) | ~16 GB |
| TP=3 | ~49 GB | 600,000 (pin) | ~36 GB |

The 3-node arm has ~20 GB more headroom per node; the 600K pin could be
raised there.

## 7. Repro

```bash
ssh sparkmain '~/3spark-qwen-flash-build/docker/build-sglang-flashnext.sh'   # digest-pinned base + patch, all 3 nodes
ssh sparkmain 'cd ~/3spark-qwen-flash-build && ARMS="2 3" nohup ./scripts/sglang-flashnext-campaign.sh > ~/campaign-flashnext-$(date -u +%Y%m%dT%H%MZ).log 2>&1 < /dev/null & disown'
# status: ~/campaign-flashnext-status.txt ; bundles: ~/results-flash-next-sglang/<ts>-tp<N>/
```

The TP=3 engine was left serving on `sparkmain:8100` as
`qwen3.8-flash-next-nvfp4` / `qwen3.8-flash-next` after this run.
