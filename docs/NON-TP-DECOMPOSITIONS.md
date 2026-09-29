# Non-tensor-parallel decompositions of Qwen3.8-Flash-Next-NVFP4 on 3 DGX Sparks

Date: 2026-09-29. Read-only analysis; nothing was booted, stopped or modified. The live SGLang TP=3 server was not touched.

Tags: **[V]** = verified (read in code, ran a command, or read a primary source this session). **[I]** = inferred (reasoning or arithmetic; not tested).
Paths: `vllm/...` = `/usr/local/lib/python3.12/dist-packages/vllm/...` in image `vllm-node-b12x:latest` (vLLM `0.1.dev21546+g502d6cb5a`, b12x 1.3.0, flashinfer 0.7.0, torch 2.13, NCCL 2.29.7). `sgl/...` = `/sgl-workspace/sglang/python/sglang/srt/...` in `sglang-flashnext-tp3:local` (sglang `g593134d17`, 2026-09-03, NCCL 2.30.7, mooncake 0.3.13 and nixl 1.4.1 installed).

## 0. Baselines and the numbers everything is compared against

Measured (from `docs/TRIAL-LOG-2026-09-29.md`, `bench-miaai.py`, 256 tok, n=5) [V]:

| config | c=1 tok/s | c=16 aggregate |
|---|---|---|
| vLLM TP=2, b12x kernels, MTP 4 | 86.2 | 408 |
| SGLang TP=3 (padded), NEXTN 3/4 | 72-74 | 427-434 |
| vLLM TP=3 old port, MTP 1 | 40.9 | 227 |

Other reference points [V, other harness `sp.py`, do not mix]: solo Spark (upstream solo recipe) 38.6 tok/s c1 vs TP=2 51.2 in the same harness, i.e. solo = 0.75 x TP=2 (`docs/RESULT-2026-09-24-UPSTREAM-VLLM-VS-SGLANG.md`). Third-party TP=1 baseline 30.8 tok/s and TP=4+EP 31.0 tok/s on the official day-0 image (tsw2k README); TP=2+EP with the same image 55.8 tok/s c1 (getrefined README, code prompt, their harness).

### Model facts used (all [V] unless noted)
- Config (`local-inference-lab/...` snapshot `7c4f1bc1`): 48 layers, hidden 2560, 24 Q heads, 2 KV heads, head_dim 256, GDN 16 key / 48 value heads, 512 routed experts top-10, moe_intermediate 640, shared expert 640, `ple_layer_ids=[2]`, `hc_count=4`, MTP 1 layer, `ple_embedding_dtype=nvfp4`.
- The PLE (N-gram) module exists in exactly one decoder layer, index 1 (`(layer_idx+1) in ple_layer_ids`, `vllm/models/qwen4_exp/nvidia/model.py` decoder-layer init). The MTP head has no PLE (`sgl/models/qwen4_exp_mtp.py:46` sets `config.ple_layer_ids = []`).
- Weight bytes by tensor-header scan (GiB): vLLM checkpoint (98.5 total): routed experts 63.3, PLE 26.9 (nvfp4), GDN 2.0, MTP 1.5, hyper-connection 1.2, lm_head 1.2, embed 1.2, full attn 0.6, shared expert 0.2, other 0.5. SGLang/RadixArk checkpoint (126 total): routed experts 63.3, PLE 47.8 (fp8), other dense 6.8, MTP 4.9, lm_head 1.2, embed 1.2, visual 0.8.
- Consequence: the non-expert, non-PLE ("dense") weights are only ~5.7 GiB (incl. lm_head) but are read in full every step; routed experts are 63 GiB but only ~9% are touched by a 5-token MTP verify batch (~5.9 GiB) [I, arithmetic: 512*(1-(1-10/512)^5) = 47.6 experts of 512].
- Bytes/step model [I], GB10 273 GB/s spec: TP=2 single stream = 2.84 dense + 2.95 experts = 5.8 GiB (~22 ms) vs measured step 3.75/86 = 44 ms; TP=3 = 1.9 + 2.0 = 3.9 GiB (~15 ms) vs measured 52 ms. At c=16 (nearly all experts touched): TP=2 34.5 GiB/step predicts 429 tok/s vs measured 408 (good fit); TP=3 predicts 650 vs measured 430 (poor fit: TP=3 is overhead-bound, not byte-bound). So per-step non-byte overhead (collectives, launches, draft passes) is ~20 ms at TP=2 and ~35 ms at TP=3. This is the calibration used below.

### Fabric
Triangle, one cable per pair, ~109 Gb/s measured per cable (`dsv4-3node-ep3-RESULT.md`), NCCL busbw ~4.6 GB/s/pair expected (`HANDOFF-TP3-PENALTY-ISOLATION.md`). 3-node NCCL needs `NCCL_IB_SUBNET_AWARE_ROUTING=1`, `NCCL_NET_PLUGIN=none`, all HCAs listed (as in the live SGLang boot script). vLLM's `VLLM_ENABLE_ROCE_ALLREDUCE` (b12x one-shot RoCE all-reduce) is wired only into TP groups (`vllm/distributed/device_communicators/b12x_roce_all_reduce.py:6-8`, `cuda_communicator.py:90`); DP/EP all-gather/reduce-scatter and PP send/recv use ordinary NCCL [V read, I for latency consequence].

## 1. Candidate 1: DP-attention x3 + EP3 on SGLang (same image, flags only)

**(a) Runs today?** Plausibly yes, no new image. Evidence [V]:
- The model file already carries the DP-attention plumbing: MoE gather/scatter around the expert block (`sgl/models/qwen4_exp.py:1311-1370`), attention-TP-group embeddings (`:1579`), N-gram/PLE token gather across DP ranks (`:476-480`, env `SGLANG_USE_ATTN_TP_NGRAM` at `:363-364`).
- GDN and full attention shard by `attn_tp_size`, not `tp_size` (`sgl/models/qwen3_5.py:332`, `:1093-1107`), so with `--dp-size 3` attn_tp_size = 1 and the head asserts (which fail for KV heads 2 at 3-way) are not reached. The stock-image padding hook only fires if `SGLANG_FORCE_UNALIGNED_TP=1` (`sgl/model_executor/model_runner.py:1169-1178`), so leave that env unset.
- DP-attention is rejected only for DFLASH, DSpark-with-non-`none`-a2a, STANDALONE and NGRAM speculation (`sgl/arg_groups/speculative_hook.py:195,347,646,905`). NEXTN/EAGLE is not blocked. Caveat from the GLM handoff: check for accept-length regression under dp-attention (PR #16310 class bug) before trusting numbers.
- Expert divisibility: `sgl/layers/moe/fused_moe_triton/layer.py:372` asserts `num_global_routed % ep_size == 0`, so 512/3 fails as-is. `--ep-num-redundant-experts 1` makes the MoE block allocate 512+1 = 513 physical experts (`sgl/models/qwen2_moe.py:339-343`), 171 per rank; trivial placement appends the extra slot as a duplicate of a logical expert (`sgl/eplb/expert_location.py` `init_trivial`). Whether Qwen4Exp's MoE block goes through that `qwen2_moe.py` path with redundant experts was not run [I].
- Suggested flags (not run): `--tp-size 3 --nnodes 3 --enable-dp-attention --dp-size 3 --ep-size 3 --ep-num-redundant-experts 1 --enable-dp-lm-head`, keep `--speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4`, drop `SGLANG_FORCE_UNALIGNED_TP` and keep `--mm-enable-dp-encoder`.
- Traps [V]: dp-attention divides `chunked_prefill_size` by `dp_size` (`sgl/arg_groups/parallel_hook.py:191-206`; pass 24576 to get 8192); `max_running_requests` is divided across DP ranks for CUDA-graph sizing (`sgl/arg_groups/cuda_graph_hook.py`, `per_rank_pool_bs = max_running_requests // attn_dp_size`), so pass 48 to give each rank 16; dp-attention disables piecewise prefill graphs (`cuda_graph_hook.py:181`, already `--disable-prefill-cuda-graph` in the live command).
- MoE kernel: SGLang TP=3 already runs the generic NVFP4 path (`--fp4-gemm-backend flashinfer_cutlass`) at 73 tok/s, so switching to EP does not lose a Spark-tuned kernel the way vLLM does (see candidate 2) [I]. The NVFP4 flashinfer-cutlass MoE passes `moe_ep_size/rank` (`sgl/layers/quantization/modelopt_quant.py:1384-1385`); an EP-equal assertion exists at `:2559` on a different path (check which path boots).

**(b) Memory per rank** [I from [V] sizes]: routed experts 63.3/3 = 21.1; PLE 47.8 fp8: default (env unset) stays TP-vocab-sharded across the 3 ranks = 15.9, or 47.8 if `SGLANG_USE_ATTN_TP_NGRAM=1` (replicated per node; do not, too tight); dense 6.8 replicated; embed+lm_head 2.4 replicated with `--enable-dp-lm-head`; MTP ~3; visual 0.8. Total ~50 GiB, leaving ~45 GiB of the 0.8 x 121 budget for KV and GDN state. KV is per-request on one node: 12 full-attention layers x 2 KV heads x 256 x K,V ~ 30 KiB/token bf16 (tsw2k: 30 GiB per 1M tokens), i.e. ~8 GiB for a 262k request. Each node holds ~1M tokens of KV. Fits.

**(c) Communication** [I]: per MoE layer one token all-gather (payload 5 KiB/token, bf16) plus one reduce-scatter; 48 target layers + 4 draft MTP layers = ~104 collectives per MTP step. Single stream: tens of KiB each, latency-bound (~40-60 us each over the triangle) = ~4-6 ms/step. c=16: ~400 KiB per collective, ~28 MB/node/step, ~3 ms bandwidth + latency = ~8 ms/step. Same collective count as TP=2/TP=3 (2 per layer), but NCCL instead of the b12x one-shot. PLE layer adds one token gather + all-reduce (single layer). MTP/NEXTN stays available [V above].

**(d) Speed** [I]: c=1 dense weights (5.7 GiB) are read in full on the active rank while two ranks idle: bytes 7.7 GiB (~30 ms) + overhead like TP=3 (~30-35 ms) = ~60-65 ms/step -> **~58 tok/s central, 48-75 range**, i.e. below SGLang TP=3 (73) and below TP=2 (86). c=16: bytes 26.8 GiB (107 ms) with TP=2's efficiency gives 530, with TP=3's gives 370, **~430 central (350-530)**, i.e. parity with both. What it buys: no padding at all, no numerics risk from padded/zero heads, per-request KV without replication. Upstream's own guidance is that dp-attention is not for low-latency small batch (`HANDOFF-TP3-PENALTY-ISOLATION.md`).

## 2. Candidate 2: TP=1 x DP=3 + `--enable-expert-parallel` on native vLLM

**(a) Runs today?** Launches, but not on the fast kernel. Evidence [V]:
- Config gates: PP=1 satisfied; `use_sequence_parallel_moe` requires `tp_size>1 and dp_size>1` (`vllm/config/parallel.py:715-731`), so at TP=1 the model's "no sequence-parallel MoE" raises (`vllm/models/qwen4_exp/nvidia/model.py:187,214`) are not hit. Default all2all `allgather_reducescatter` (`parallel.py:197`); `pplx`/`naive` removed (`:501-508`).
- **The b12x MoE kernel refuses EP**: `_supports_parallel_config` requires `not use_ep and ep_size == 1 and not use_all2all_kernels and not enable_eplb` (`vllm/model_executor/layers/fused_moe/b12x.py:627-635`), and raises "b12x TP MoE does not support expert maps" (`:859-860`); the flashinfer b12x path is the same (`experts/flashinfer_b12x_moe.py:234-241`: "does not yet support expert parallelism"). So EP forces `--moe-backend flashinfer_cutlass` (parallel config accepted, `experts/flashinfer_cutlass_moe.py:195-200`) or `marlin` (`experts/marlin_moe.py:596-600`); vLLM's own CUTLASS FP4 is `ep_size == 1` only (`experts/cutlass_moe.py:734-737`).
- Non-MoE b12x stays possible: `uses_b12x` is true if `linear_backend=="b12x"` OR `moe_backend=="b12x"` (`vllm/models/qwen4_exp/nvidia/backend.py:8-10`), so `--linear-backend b12x --moe-backend flashinfer_cutlass` keeps b12x PLE/QSA/GDN/linear [I that the mixture boots].
- Uneven expert maps are supported by vLLM (`fused_moe/expert_map_manager.py:22-70`, 171/171/170) but the FlashInfer cutlass kernel is invoked with `ep_size/ep_rank` only (`flashinfer_cutlass_moe.py:384-385`); it derives per-rank expert count from equal splits, so 171/171/170 is likely wrong [I, not verified in FlashInfer source]. Use 513 via `--enable-eplb --eplb-config '{"num_redundant_experts":1,...}'` (EPLB is accepted with TP*DP > 1, `parallel.py:527-540`), untested with CUDA graphs + MTP.
- PLE: b12x PLE refuses cross-DP storage (`vllm/models/qwen4_exp/nvidia/ple_layer.py:130-136`) and shards over TP only (`b12x_ple.py:620-621`), so each DP rank holds the whole 26.9 GiB nvfp4 table. Non-b12x path supports ETP spanning DP (`ngram_embedding.py:55-175`) but nvfp4 PLE "requires the b12x execution backend" (`ngram_embedding.py` `from_quant_config`).
- Launch shape (from our DSv4 EP3 run, `dsv4-3node-ep3-RESULT.md`): head without a rank flag, workers `--data-parallel-start-rank N --headless`; `--data-parallel-size 3 --data-parallel-size-local 1 --data-parallel-address <head> --data-parallel-rpc-port <p>`. Do not use `--data-parallel-rank` (forces external-LB).
- `prefill_compute_share` is refused with DP (`vllm/config/vllm.py:1449-1455`); our rung-2 `decode-aware` prefill flags may need checking.

**(b) Memory per rank** [I]: experts (63.3+1.3 MTP)/3 = 21.5 + PLE 26.9 + dense/embed/lm_head/MTP-dense 7.0 = ~55 GiB; ~30 GiB KV at gmu 0.8 = ~1M tokens.

**(c) Comm/MTP**: same as candidate 1 (allgather+reducescatter per MoE layer, ~100/step) over NCCL; MTP stays available (no MTP-specific DP gate found) [V: no gate; I: works]. DP lockstep means idle ranks run dummy batches; c=1 latency is unaffected but idle ranks burn power.

**(d) Speed** [I]: The lesson of the DSv4 EP3 run is kernel-bound: EP2 = EP3 = 19-22 tok/s vs TP=2 b12x 49-55 (2.5x). Qwen differs in that the generic path is cutlass FP4, and SGLang TP=3 (also generic) does 73, so I expect **c=1 ~45-60 tok/s** (below SGLang candidate 1 because NCCL collectives replace nothing and b12x is lost), c=16 **~300-420**. Strictly dominated by candidate 1 unless we specifically need vLLM.

## 3. Candidate 3: PP=3 x TP=1

### 3a. vLLM native (b12x kernels survive)
- B12X gates ignore PP (`use_ep` needs `dp*pcp*tp>1`; DSv4 PP3 run confirmed the B12X backend loads under TP=1/PP=3, `dsv4-3node-pp3-RESULT.md`). So PP is the only 3-way option that keeps the b12x MoE kernel [V from DSv4, [I] for Qwen].
- Blockers, all small and located [V]:
  1. `vllm/model_executor/models/config.py:1010-1015` raises NotImplementedError before load (this is our T3 failure).
  2. `vllm/models/qwen4_exp/nvidia/model_state.py:138-142` raises RuntimeError again per rank.
  3. `vllm/models/qwen4_exp/nvidia/mtp.py:635-646`: drafter forward branches on the global `get_pp_group().is_first_rank`; the drafter lives on the last rank, so it takes the `intermediate_tensors` branch and fails (vllm issue #54709 item 2). Fix: `if intermediate_tensors is None` (issue's proposed one-liner). Also check `mtp.py:659,779`.
  4. Possible: draft-token sync across PP ranks (vllm PR #52295, open). `vllm/v1/worker/gpu/pp_utils.py` already has PP output broadcast for the V2 runner (`:63-85`), so the base is there; MTP-under-PP core (#46994) is merged (2026-09-11) but whether it is in this image was not confirmed.
- Already fine [V]: the model implements PP (`model.py:476-478` intermediate tensors of width `hidden*hc_count` = 10240 bf16 = 20 KiB/token; `:536-548`, `:589-596` materialize the delayed HC combine before send; `:630-643` skips `hyper_connection_mixer.*` on non-last ranks, which was 1Cat-vLLM #479 wall 3). PLE sits on rank 0 (layer index 1), which has raw `input_ids`; the "later ranks lack input_ids" reason in the gate does not apply as long as layer 1 is on stage 0 (default 16/16/16 partition puts it there; issue #54709 proposes exactly this placement check and reports PP4 at ~61 tok/s single stream / 163 at 8-way on 4x SM80 with MTP after the same fixes; a ROCm TP2xPP3 reproduction also exists in that thread).
- Difficulty: **low to moderate**. Three bind-mounted whole-file patches (our proven mechanism, `PATCH-INVENTORY.md` section 1) + existing recipe `configs/colonel-qwen3.8-flash-next-nvfp4-pp3.yaml`. Unknown: further per-rank assumptions in b12x preparation/PLE hooks on ranks that own `PPMissingLayer`s (`b12x_startup.py` is world-collective).
- Memory [I]: ~24 GiB layers per stage, + PLE 26.9 and embed on stage 0 (~52 GiB), + lm_head 1.2 and MTP 1.5 on stage 2 (~27 GiB). KV per stage covers 4 full-attention layers, so per-token KV per node is 1/3: total pool several M tokens.
- Comm [I]: no per-layer collectives. Per step 2 boundaries x tokens x 20 KiB: c=1 100 KiB, c=16 1.6 MB, plus tiny sampled-token/draft broadcasts. <1 ms/step. MTP stays available if fixes 3/4 hold; DSv4's "PP forbids MTP" (`DeepSeekMTP` lacks `SupportsPP`) does not transfer, Qwen4Exp MTP has explicit PP branches.
- Speed [I]: single stream is sequential across stages = a solo Spark that never has to offload PLE: bytes 11.6 GiB (~46 ms) + ~12 ms overhead + hops = ~58-62 ms -> **~62 tok/s central (50-72)**. Cross-check: solo/TP=2 = 0.75 x 86 = 65. Will not beat TP=2 (86) or SGLang TP=3 (73) at c=1. Aggregate: 3 micro-batches in flight (async scheduling, `max_concurrent_batches = pp_size` [I]) with b12x kernels and no per-layer collectives: **~350-480 at c=16, possibly higher at c>=32**; zebgop's PP=3 74 tok/s is on SM80 cards and not comparable. Prefill should scale well (chunks pipeline across stages, no collectives; SGLang PR #40501 reports TP1xPP4 prefill 1.27x DEP4 per GPU on GB300) [I; worth measuring TTFT at 32k/128k].

### 3b. SGLang PP
- Our image (2026-09-03) cannot: `Qwen4ExpVLModel.forward` accepts `pp_proxy_tensors` but never uses it (`sgl/models/qwen4_exp.py:1685`, only occurrence) and there is no `check_pipeline_parallel_compat` / `SGLANG_ENABLE_PP_SPEC` in the tree (grep empty) [V].
- PR #40501 (merged 2026-09-22) adds it, validated on GB300 with TP2xPP2 + MTP at GSM8K 0.975-0.980, accept length 3.1-3.2 [V from PR body]. The aggregate-with-MTP path also needs #39643 (merged 2026-09-28) and #40001 (still open: hybrid GDN recurrent-state commit under PP+spec; without it GSM8K drops to 0.955, accept 3.02). So PP+MTP on SGLang is a correctness risk for a GDN hybrid until #40001 lands.
- Cost: rebuild `sglang-flashnext-tp3` from a post-09-28 base, re-check SM121 QSA fixes and our padding patch (moot at TP=1), push image to 3 nodes. **High effort compared to 3a**, so 3a first.

## 4. Candidate 4: hybrid, TP=2 pair plus a job for node 3

Node 3 is not needed for memory: TP=2 already fits (99 GiB checkpoint, ~50 GiB weights per rank). So the third node must earn its keep by throughput or latency:
- 4a. **Third node hosts the PLE table or remote experts / MTP head**: kill. PLE is one layer, gather is on the critical path of layer 1 and adds a network RTT per step for ~27 GiB you do not need to move; MTP head is 1/48 of compute but is called 4x sequentially per step (4 extra RTTs); remote experts need EP, which the b12x kernel refuses (candidate 2 evidence). [I]
- 4b. **Independent solo instance on node 3 + router** (this is what production does today, `RESULT-2026-09-24...` "Production layout"): TP=2 endpoint 86 tok/s / ~408 agg plus a solo endpoint (measured solo c=8 aggregate 112 vs TP=2 c=16 221 in the sp harness, so ~50% of a TP=2 pair). Summed aggregate ~550-600 in bench-miaai units [I from ratios], single-stream 86 on one and ~65 on the other. Best honest use of node 3 for throughput, but it is two endpoints (needs an LB, no shared prefix cache).
- 4c. **Disaggregated prefill/decode: node 3 = prefill (TP=1), pair = decode (TP=2)**. KV to move for a 100k prompt: ~3 GiB (12 layers x ~30 KiB) + ~0.12 GiB GDN state [I], ~0.15 s over one cable. SGLang image already has mooncake 0.3.13 + nixl 1.4.1 + `disaggregation/` and PR #40501 exercised PD-prefill with MTP for Qwen4Exp (on GB300, Mooncake) [V]; our image predates that PR, so hybrid GDN+QSA state transfer on this build is unproven [I]. vLLM has NIXL/Mooncake connectors (`vllm/distributed/kv_transfer/kv_connector/v1/*`) but hybrid GDN+QSA transfer is untested. Benefit: decode latency is isolated from long agent prompts; no gain in c=1 tok/s (stays 86 or 73). Effort high.
- Verdict: 4b is free and real; 4c is a latency-isolation project, not a tok/s project.

## 5. Candidate 5: uneven TP with targeted padding (baseline)

Where the divisibility checks live [V]: `vllm/models/qwen4_exp/nvidia/qsa.py:207-220`, `b12x_qsa.py:903-911` (Q heads 24 divisible by 3; KV heads 2 -> raises), SGLang `qwen3_5.py:1100-1107` (assert), GDN key heads 16 (`qwen3_5.py:850-854` already pads via `triton.cdiv` in some paths), MoE 640 and vocab 248320 (padding hooks in `patches/sglang_upstream/enable_unaligned_tp_on_cuda.py`).
- 5a. **Zero-head KV 2->3 (what the live SGLang does)**: requires Q 24->36 so the GQA group stays 12 (Q/KV ratio preserved), so each rank does 12 Q heads, the same as TP=2 (24/2). Attention flops per rank do not fall; GDN 16->18 key / 48->54 value heads (+12.5%); MoE 640->672 (+5%) [numbers from `TRIAL-LOG` and patch]. Padded heads are zero weights, so numerics rest on zero-pad exactness and the logprob gate that is still owed.
- 5b. **KV replication instead of zero heads** (each rank stores both KV heads, Q = 8/rank): rank 0 needs KV0 for its Q heads 0-7, rank 1 needs KV0 and KV1 (Q 8-15 straddle the GQA boundary at head 12), rank 2 needs KV1. So replicating both KV heads on every rank is exact, wastes 1 extra KV head of compute/memory per rank (KV proj is tiny; KV cache per token per rank 12 KiB fp8 vs 6 KiB at TP=2, negligible), and cuts Q work 12->8 heads per rank versus 5a. GDN: contiguous split of 48 value heads = 16/rank, needing 6-7 key heads/rank with boundary key heads (5, 10) duplicated on two ranks, versus 18/54 padding. This is the minimal-waste TP=3 but needs new sharding code in both engines (neither supports non-divisible KV/key head replication; vLLM asserts at `qsa.py:219-220`); large patch, and the wins are second-order because the measured TP=3 shortfall is overhead not per-rank arithmetic (section 0 calibration).
- Expected: same as the measured SGLang TP=3 (72-74, ~430); 5b might recover a few ms of attention time, +0-8% [I].

## 6. Candidate 6: "TP=2 x 1.5" uneven splits
Not justified. No engine here supports uneven TP shards; a 2:1 head split (16/8 or 8/8/8 with KV homing) is equivalent to 5b with more code. The only uneven split with a clean justification is uneven PP layer partition (`VLLM_PP_LAYER_PARTITION`, already supported and used in issue #54709 as `2,16,16,14`) to offload stage 0 (which also holds the 26.9 GiB PLE and the embed) and stage 2 (lm_head + MTP); e.g. `14,17,17` if a stage is memory-heavy or slower. Only a tuning knob for candidate 3.

## 7. Ranking and the first experiment

| # | option | c=1 tok/s (bench-miaai units) | c=16 agg | needs | risk |
|---|---|---|---|---|---|
| 3a | vLLM PP3xTP1 (b12x kept) | ~62 (50-72) | ~350-480 | 3 file patches + existing recipe | medium (unknown later-rank gates, MTP-under-PP) |
| 1 | SGLang DP-attn3 + EP3 (513 experts) | ~58 (48-75) | ~430 (350-530) | flags only | medium (EP/redundant expert path, accept length under dp-attn) |
| 2 | vLLM DP3 + EP (cutlass) | ~50 (45-60) | ~300-420 | flags, EPLB for 513 | high, dominated by 1 |
| 5a | SGLang TP3 padded (baseline) | 73 (measured) | 430 (measured) | live | n/a |
| 5b | TP3 replicate KV | ~73-78 | ~430-460 | big patch | high effort |
| 4b | 2x endpoints (TP2 + solo) | 86 / ~65 | ~550-600 summed | LB | none |
| 4c | PD split | 86 (decode unchanged) | n/a | mooncake + rebuild | high effort |

None of the clean-by-3 options is expected to beat 86 (TP=2) at c=1; the honest wins are memory/KV headroom, no padding, and possibly aggregate/prefill.

### FIRST: vLLM PP=3 x TP=1 (option 3a)
Why first: it is the only 3-way decomposition that keeps the b12x kernels that make TP=2 the fastest, has no per-layer collectives (the thing that makes TP=3 overhead-bound), needs no new image, and the code is already PP-aware (three verified edit sites).

Minimal experiment (one boot, ~12 min with warm caches), all changes are bind-mounted copies, no image or source edits:
1. Copy out and patch three files: `vllm/model_executor/models/config.py` (replace the unconditional PP gate at :1010-1015 with "every PLE layer index must lie inside PP rank 0's layer range", per issue #54709), `vllm/models/qwen4_exp/nvidia/model_state.py` (drop or gate the raise at :138-142), `vllm/models/qwen4_exp/nvidia/mtp.py` (:635 `if get_pp_group().is_first_rank:` to `if intermediate_tensors is None:` and the matching last-rank test at :659/:779 if it hits).
2. Boot `configs/colonel-qwen3.8-flash-next-nvfp4-pp3.yaml` unchanged (TP=1, PP=3, MTP 4, b12x MoE + linear), default 16/16/16 partition. Check the log for `Using 'B12X' ... MoE backend` on all three PP ranks and for MTP accept length.
3. Correctness first (greedy prompt diff against the TP=2 reference), then `bench-miaai.py` c=1/4/8/16 (warm sweep, discard the first), plus TTFT at 32k and 128k prompts.

Confirms if: c=1 >= 65 with accept length near TP=2's, and c=16 aggregate >= 430 (or c=32 > TP=2's). Falsified if: (i) it dies on a further gate on a non-first rank (record which; that sizes the real patch); (ii) c=1 < 55 (pure sequential-stage cost, no chance to beat SGLang TP=3); (iii) MTP accept length collapses (drafter/hidden-state relay broken) or output diverges from the TP=2 reference; (iv) c=16 < 350 (microbatch overlap not happening). If (ii) with good aggregate, PP3 remains the best long-context/KV option and is worth keeping as a second production lane.

### SECOND (zero-code): SGLang DP-attn3 + EP3 (option 1), one boot with the flags in section 1, correctness = greedy diff vs TP=2 reference plus `sglang:spec_accept_length` >= 3.5 (falsified if < 3, which is the dp-attention accept-length regression class, or if the 513-expert path fails at load).

## 8. Not verified (explicit)
- No engine was booted; every speed number is [I] from a bytes model calibrated on two measured configs (TP=2, TP=3) at two concurrencies, with the caveat that TP=3 fits poorly at c=16.
- Whether Qwen4Exp's SGLang MoE block actually applies `ep_num_redundant_experts`, whether flashinfer-cutlass NVFP4 MoE accepts 171-expert shards on SM121, and whether NEXTN accept length holds under dp-attention.
- Whether this vLLM image already contains vllm PR #46994 (MTP under PP) and needs PR #52295.
- FlashInfer cutlass MoE behavior with uneven EP splits (source not read).
- Host-side GB10 memory behavior (unified 121 GiB, page-cache and wedge traps documented in the DSv4 notes) when a node holds 52 GiB (PP stage 0) vs 27 GiB (stage 2).
