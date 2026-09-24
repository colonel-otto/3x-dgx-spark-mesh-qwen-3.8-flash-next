# Handoff: SGLang for Qwen3.8-Flash-Next on 3 Sparks (TP=2 first, then TP=3)

**Written:** 2026-09-04. **Status:** image BUILT and identical on all 3 nodes;
**NEVER BOOTED** (the cluster was serving the dense Qwen3.8-27B SGLang TP=3
engine all day, 2 GB free RAM per node, and stopping it was not this
session's call). Nothing below is a performance claim.

## 1. The one-paragraph story

The morning plan was "port Qwen4Exp to SGLang by hand, then layer the dense
repo's TP=3 padding patch on top." That plan was wrong by one day of
research: SGLang has **native** `qwen4_exp` support on the unreleased
`qwen4-main` branch, published as the official day-0 image
`lmsysorg/sglang:qwen38flashnext` (== `:dev-qwen38flashnext`, digest
`sha256:5ae5816783d5...`, pushed 2026-09-03). A community recipe runs this
exact checkpoint on 2 Sparks (GB10, TP=2) from a derivative of that image at
~70 tok/s peak with native MTP. So the architecture port is gone; what
remains is exactly the TP=3 work the dense repo already paid for, plus two
new loader gates, and it is all built and verified statically. What is not
done is the only thing that matters: a boot.

## 2. What was verified on sparkmain (not taken from the research)

| item | result |
|---|---|
| `lmsysorg/sglang:v0.5.18` (our dense image) on the Flash-Next config | `ValueError: model type qwen4_exp ... not recognized` (transformers 5.12.1) |
| `lmsysorg/sglang:dev-qwen38flashnext` | sglang `0.0.0.dev1+g593134d17`, branch `qwen4-main-squashed-rebased`, commit 2026-09-03; transformers 5.12.1 but SGLang ships its own `configs/qwen4_exp.py`; `models/qwen4_exp.py` 2131 lines + `qwen4_exp_mtp.py` |
| NCCL in that image | wheel `2.30.7` in `/opt/sglang/lib/python3.12/site-packages/nvidia/nccl/lib/` has **1** `SUBNET_AWARE_ROUTING` symbol and is what `pynccl` loads; system lib 2.28.3 has 0. **No donor swap needed.** |
| SM121 QSA fixes | `_resolve_trtllm_sparse_decode` already gated `is_sm100_supported() or is_sm120()` (#36806) and `is_sm121()` routes to the Triton packed-varlen kernel (#36845). The recipe's `pr36845.diff` **does not apply** because it is already merged. |
| Our 7-anchor patch (`enable_unaligned_tp_on_cuda.py`) | all 7 anchors apply cleanly in a throwaway container |
| Gate audit over the real `qwen4_exp` import graph | 607 modules, 10 `is_cpu` sites: 7 covered, 2 new but runtime dispatch (`dp_attention.py` memcpy, `forward_batch_info.py` pin_memory), 1 already CUDA-safe |
| Dry-run of `adjust_config_with_unaligned_cpu_tp(tp=3)` on the real config | see table in section 3 |
| `docker/build-sglang-flashnext.sh` | builds on all 3 nodes, engine fingerprint `51946eb75a4d...` identical on .70/.6/.66, all 9 patched files carry the sentinel |

## 3. TP=3 padding plan (dry-run on the real config, no GPU)

| axis | original | padded | per rank | mechanism |
|---|---:|---:|---:|---|
| Q heads | 24 | 36 | 12 | `adjust_config_with_unaligned_cpu_tp` (model_runner gate lift) |
| KV heads | 2 | 3 | **1** | same; rank 2 holds an all-zero KV head |
| GDN key / value heads | 16 / 48 | 18 / 54 | 6 / 18 | `update_config` plain-attr lift (stock only writes `_cpu` attrs) |
| moe_intermediate_size | 640 | 672 | 224 | tp x NVFP4 group 16 = 48 alignment; rank 2 real slice 192 |
| shared_expert_intermediate_size | 640 | 672 | 224 | same |
| vision heads | 16 | 18 (computed) / 16 (used) | 16 | `original_num_heads` fix + `--mm-enable-dp-encoder` |
| vision intermediate | 4304 | 4320 | | derived |
| vocab 248320, PLE n-gram table | | padded inside `VocabParallelEmbedding` | | vocab-embedding gate lift |
| QSA indexer heads | 4 | 4 | 4 | `ReplicatedLinear`, not sharded |

This is the same plan, axis for axis, that vLLM's `virtual_tp.py` printed
when it booted the served engine (`~/qwen-next-tp3-mnbt8192.log`). The
padding **math** is therefore already validated on this checkpoint by the
engine serving it today. The SGLang **loaders** are the new ground.

## 4. What the dense-repo work transferred, and what was new

Transferred unchanged (all 7 anchors): model_runner gate, vocab-embedding
gate, update_config plain attrs, `qwen3_5.py` packed GDN loader (Flash-Next
reuses `qwen3_5.py`'s GDN/attention classes), `weight_utils` shim, `mamba`
shim, `parameter.py` cached bool. Also the audit script, the shim verifier,
the build-by-content fingerprint, the 6/6 gate, and the boot-script guards.

New in `patches/sglang_upstream/enable_unaligned_tp_on_cuda.py`:

- `patch_fused_moe_layer` -- `fused_moe_triton/layer.py::use_padded_loading`
  is `_is_cpu or trtllm or aiter`. On GB10 the NVFP4 MoE runner is
  `flashinfer_cutlass` (trtllm-gen has no sm_121 kernels), so CUDA takes the
  strict `narrow()` path and rank 2's 192-wide slice of a 640-wide expert
  would fail against a 224-wide shard, for all 294,914 expert tensors. Same
  cached-bool rewrite as `parameter.py`. **The static audit did not find
  this** -- its regex sees `is_cpu(` call sites, not a bool consumed 500
  lines later. Found by reading the loader.
- `patch_qwen3_vl_vision_heads` -- the dense repo's vision fix as two
  targeted edits instead of a whole-file copy (the file drifted on
  `qwen4-main`). Needed: the dry-run pads vision heads 16->18 and
  `Qwen4ExpForConditionalGeneration` subclasses
  `Qwen3VLForConditionalGeneration`, so the rotary head_dim corruption
  (1152//18 vs 1152//16) would crash the first forward exactly as before.

What did NOT transfer: the NCCL donor swap (unneeded), the `/usr/local/lib`
site-packages paths (`/opt/sglang` venv here), the DSpark/FP8 drafter
choices (Flash-Next uses its own MTP head via `--speculative-algorithm
NEXTN`), and the quantized-lm_head blocker (`lm_head` is BF16 in this
checkpoint, so it does not fire).

## 5. Research (30-day window, three parallel agents) -- what changed the plan

1. **SGLang support exists, in a branch not a release.** Cookbook pins
   `qwen4-main @ e17062a1d`; LMSYS day-0 blog 2026-08-26; latest release
   v0.5.18 (2026-08-22) has none of it. PR #36585 "native Qwen4-Exp" is open
   with failing CI -- ignore it, the branch/image is the path.
2. **Community 2-Spark recipe** (github.com/tonyd2wild/Qwen3.8-Flash-Next-NVFP4-DGX-Spark,
   2026-08-26..28; cloned to `sparkmain:~/Qwen3.8-Flash-Next-NVFP4-DGX-Spark`):
   TP=2, `--quantization modelopt_fp4 --fp4-gemm-backend flashinfer_cutlass`,
   `TORCH_CUDA_ARCH_LIST=12.1a FLASHINFER_CUDA_ARCH_LIST=12.1a`, NEXTN
   3 steps / 4 draft tokens, `--ple-offload-embedding`, `--max-mamba-cache-size 97`,
   `--max-total-tokens 600000 --mem-fraction-static 0.80`, radix off,
   pytorch sampler, drop_caches before every launch. ~76 GB/rank at TP=2,
   69.7 tok/s peak, ~50 typical, ~20 without MTP. Its image is a local build
   (`radixark/...` is not on Docker Hub) from the same day-0 base.
3. **sglang #36796 (2026-08-28):** 2x Spark TP=2, 58.5 tok/s single-stream,
   profile says **NCCL is ~7% of GPU time**; the model is kernel-bound on
   SM121 (34% NVFP4 GEMM, 16% grouped MoE, 12% QSA gather). A third node
   buys little decode per stream if that holds; it buys KV and batch.
4. **No TP=3 Qwen4Exp run exists anywhere.** Every published config is a
   power of two. We would be first.
5. **FP8:** `Qwen/Qwen3.8-Flash-Next-FP8` (official, ~173 GiB) fits at TP=3
   (~65 GiB/rank, ~55 free). BF16 (335 GiB) does not. Quality NVFP4 == FP8
   == BF16 within +-1 on a 1,370-item suite (primitive-ai, 2026-09-03). FP8
   forces a BF16 KV cache (QSA), 5-8x fewer KV tokens. **Verdict: no quality
   reason to leave NVFP4; FP8 is a speed A/B with a KV penalty, run it later
   per BENCHMARK-POLICY.md if at all.**
6. **Landmines:** MTP acceptance decays toward zero over server uptime on
   qwen4_exp (sglang #37326) -- a long sweep will drift down and read as a
   regression; restart between arms. `--disable-radix-cache` collapses the
   mamba pool without `--max-mamba-cache-size` (recipe incident). The GB10
   unified pool: page cache starves the GPU allocator ~20 min into load.
   vLLM users saw non-deterministic greedy decode from the QSA indexer's
   `persistent_topk` on GB10 -- do not assume byte-identical replays.

## 6. Run order (nothing here has been executed)

```bash
# 0. Precondition: the cluster must be FREE. Today it was not.
ssh sparkmain 'for n in 192.168.10.10 192.168.10.11 192.168.10.12; do ssh $n "hostname; docker ps --format \"{{.Names}}\" | grep -E \"vllm_node|sglang_node\""; done'
#    Stopping whatever is there is the operator's decision, not the script's.

# 1. TP=2 FIRST (zero padding: proves the day-0 engine, the GB10 NVFP4 MoE
#    kernels, our fabric env, and the recipe flags on THIS cluster).
ssh sparkmain '~/3spark-qwen-flash-build/scripts/sglang-flashnext-boot.sh 2'
#    Cold boot: 126 GiB of weights + first-time JIT. Watch RSS, not the log.
ssh sparkmain '~/3spark-qwen-flash-build/scripts/sglang-flashnext-validate.sh http://127.0.0.1:8100/v1 qwen3.8-flash-next'
#    Then capture a TP=2 logprob reference (dense repo: scripts/tp-logprob-capture.py).

# 2. TP=3 (SGLANG_FORCE_UNALIGNED_TP=1 + --mm-enable-dp-encoder are set by the script).
ssh sparkmain '~/3spark-qwen-flash-build/scripts/sglang-flashnext-boot.sh 3'
#    Expected first-boot failure class if anything: a rank-2 loader shape
#    error in an NVFP4 expert tensor (w13/w2 or their block scales). That is
#    LOUD and means patch_fused_moe_layer needs a scale-tensor sibling.
#    A boot that "works" proves nothing -- see 3.

# 3. Correctness BEFORE any tok/s: 6/6 battery, then the logprob gate
#    against the TP=2 arm (TP=1 is impossible: 126 GiB > one node). Same
#    design as docs in the dense repo (REVIEW-2026-09-03-SGLANG-TP3-GATE.md):
#    deterministic boots, TP=3 within 2x of the TP=2-vs-itself drift.
#    Deterministic mode at TP=3 needs SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM=0
#    again (GDN in_proj_ba is N=36 here too).

# 4. Only then: policy-compliant sweep (256-tok window, exclusivity,
#    fabric gate, min/max) against results/20260903T2153Z-...-256tok
#    (vLLM TP=3: 40.9 / 30.8 / 25.8 / 20.2 tok/s at c=1/4/8/16).
```

## 7. Files

| path | what |
|---|---|
| `docker/Dockerfile.sglang-flashnext-tp3` | digest-pinned day-0 base + patch + audit + shim check + provenance |
| `docker/build-sglang-flashnext.sh` | builds on 3 nodes, verifies engine by content |
| `patches/sglang_upstream/enable_unaligned_tp_on_cuda.py` | 7 dense anchors + 2 Flash-Next additions |
| `scripts/audit_unaligned_tp_gates.py` | entrypoints = `qwen4_exp`, `qwen4_exp_mtp` |
| `scripts/verify_unaligned_tp_shims.py` | unchanged from dense |
| `scripts/sglang-flashnext-boot.sh <2\|3>` | recipe flags + our fabric; refuses to evict another engine |
| `scripts/sglang-flashnext-validate.sh` | 6/6 gate, `enable_thinking:false` |
| sparkmain `~/3spark-qwen-flash-build/` | synced copy the build ran from |
| sparkmain `~/Qwen3.8-Flash-Next-NVFP4-DGX-Spark/` | community recipe clone (README, launcher, incident notes) |

Image on all three nodes: `sglang-flashnext-tp3:local`, base
`lmsysorg/sglang@sha256:5ae5816783d58e2e56e84d2e863f5441425056f500b7fbd7448c4aae017a2521`.
`docker run --rm --entrypoint cat sglang-flashnext-tp3:local /etc/sglang-flashnext-tp3-provenance`
tells you what it is.

## 7b. Live boot log (added 2026-09-04 13:10 UTC, supersedes "NEVER BOOTED")

The operator authorized evicting the dense engine and asked for an
unattended TP=2 vs TP=3 campaign. Each boot below is the SAME image tag
rebuilt in place on all 3 nodes (fingerprint changes recorded).

| boot | time (UTC) | got to | failure | fix |
|---|---|---|---|---|
| 1 | 12:52 | NCCL group formed (nccl 2.30.7), padding applied, weight load | ranks 1+2: `linear.py:452 start (0) + length (4320) exceeds dimension size (4304)` via `qwen4_exp.py:2089` | vision MLP built from padded `vision_config.intermediate_size`; TP=1 (dp-encoder) tower must use `original_intermediate_size`. Third edit in `patch_qwen3_vl_vision_heads`. |
| 2 | 12:59 | past the vision tower | rank 2: `linear.py:1394 (QKVParallelLinear.weight_loader) start (512) + length (256) exceeds dimension size (512)` via `qwen4_exp.py:2008` | 512 = 2 KV heads x 256; NVFP4 checkpoint leaves qkv_proj UNQUANTIZED so it takes the v1 loader in `linear.py`, whose 8 `_is_cpu` branches are all padding branches. New `patch_linear_layer` (cached-bool rewrite). The audit's "linear.py already pads on CUDA" classification was wrong for v1 loaders; corrected. |
| 3 | 13:08 | ALL weights loaded on all ranks (every loader fix held), post-load processing | rank 2: `modelopt_quant.py:2751 AssertionError: The intermediate size required padding, but padding is also implemented for gated activations` | cutlass NVFP4 MoE swizzles `w13_weight_scale` in 128-row tiles: per-rank 2 x 224 = 448 rows pads to 512 and the code refuses to split-pad gated w1/w3. Need `intermediate_per_rank % 64 == 0`. Second edit inside `patch_update_config`: under the flag, MoE/shared alignment becomes tp x 64 (`SGLANG_UNALIGNED_TP_MOE_ALIGN`), so 640 -> **768** (256/rank; rank 2 = 128 real + 128 zero rows). Dry-run confirmed 768/768, GDN 18/54, heads 36/3. TP=2 (320/rank = 5 x 128) never sees this. The campaign moved on to TP=2 automatically; TP=3 re-run queued after it. |
| 4 | 13:55 | **SERVING.** All ranks loaded, post-load OK, KV 600K, 6/6 gate PASSED, sweep done, 6/6 gate PASSED again (campaign `20260904T1355Z`) | none | Numbers in `docs/RESULT-2026-09-04-SGLANG-FLASH-NEXT-TP2-VS-TP3.md`: c=1 75.7 tok/s vs TP=2 66.5 vs vLLM TP=3 40.9. Logprob gate vs TP=2 still owed. |

TP=2 arm (campaign `20260904T1308Z`, 13:12-13:26): booted first try, 6/6
before and after, sweep clean. Bundles imported into `results/`.

Lesson for the audit: a cached module-level `_is_cpu` bool is invisible to a
call-site regex, and "already pads on CUDA" must be checked per LOADER, not
per file. Both v1 (`linear.py`) and v2 (`parameter.py`) paths exist and which
one a projection takes depends on whether the checkpoint quantized it.

Campaign mechanics: `scripts/sglang-flashnext-campaign.sh` under nohup on
sparkmain; status in `~/campaign-flashnext-status.txt`; per-arm bundles in
`~/results-flash-next-sglang/<ts>-tp<N>/` (rank logs, live cmd/env, image
provenance, gate logs, sweep rows). Shared knobs for BOTH arms:
`--max-running-requests 16` (c=16 needs 16 slots; the recipe's 6 would have
queued), `--max-total-tokens 600000`, `--mem-fraction-static 0.80`,
`--max-mamba-cache-size 97`, `--cuda-graph-max-bs 8`, `--enable-metrics`
(exclusivity.py now auto-detects SGLang metric names). Harness is the
policy-compliant `~/bench-miaai.py` (WindowCollapse assertion, 256-token
window); the repo copy was stale and has been re-synced from sparkmain.

## 8. Open questions this session could not close

- Does the NVFP4 expert **block-scale** loader path also need padded
  loading, or is `narrow_padded_param_and_loaded_weight` reached for
  `w13_weight_scale` / `w2_weight_scale` through the same `_load_w13/_load_w2`?
  Only a rank-2 boot answers this.
- Whether `--enable-linear-replayssm-spec` and `--mamba-track-interval`
  still exist on the 09-03 build (recipe used the 08-26 build). SGLang
  rejects unknown flags loudly at start, so this costs one attempt.
- `--max-total-tokens 600000` was the recipe's TP=2 OOM pin; at TP=3 the
  per-rank KV footprint is smaller and the pin can probably rise. Leave it
  for the first boot.
- Whether TP=2 on this fabric is happy with the full 4-HCA list under a
  SAR-capable NCCL (the dense TP=2 boots narrowed per pair on a non-SAR
  NCCL). If "Init torch distributed begin" hangs at TP=2, narrow.
