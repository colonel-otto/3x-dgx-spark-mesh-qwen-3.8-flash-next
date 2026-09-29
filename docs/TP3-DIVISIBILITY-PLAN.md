# TP=3 divisibility plan for Qwen3.8-Flash-Next-NVFP4 on the NATIVE vLLM image

**Date:** 2026-09-29. **Status:** design + CPU-only verification. Nothing here was run on a GPU,
nothing was started/stopped, no existing file was modified. The SGLang TP=3 server that is live
was left alone.

Evidence labels used throughout:

- **[V]** verified by reading code or running a read-only command. File:line refer to the image
  `vllm-node-b12x:latest` (vLLM `0.1.dev21546`, b12x 1.3.0), package root
  `/usr/local/lib/python3.12/dist-packages/`, abbreviated `vllm/...` and `b12x/...`.
- **[V-CPU]** verified by the CPU exactness test in Appendix A, run on real checkpoint tensors
  (`~/agent-scratch/tp3_cpu_check.py` on sparkmain, `docker run --rm --entrypoint python3`, no `--gpus`).
- **[I]** inferred. Needs a GPU or a measurement to confirm.

Checkpoint audited: `local-inference-lab/Qwen3.8-Flash-Next-NVFP4` @ `7c4f1bc1` (36 safetensors
shards, 98.5 GiB, all 394 tensor patterns enumerated from the shard headers).

---

## 0. Bottom line (read this first)

1. **TP=3 is buildable on the native image with zero Python patches to vLLM or b12x.** Every divide
   site is driven by `config.json` (verified below), so the whole job is: an offline-padded
   checkpoint directory + an edited `config.json` + env/flag changes. BHCC2025's `model.diff` /
   `mtp.diff` / `install()` are not needed and one of them (load-time transform) cannot work with
   our `--load-format b12x` loader.
2. **Use PLAN-PAD (DSv4-style group-preserving pad), not PLAN-COPY.** PLAN-COPY's GDN half is
   *rejected by the b12x GDN kernels* (they require V:K heads = 3:1; COPY gives 1:1). PAD keeps
   exactly the per-rank geometries production TP=2 already runs (12 q : 1 kv, 3:1 GDN).
3. **Two blockers nobody had listed:** (a) the b12x PLE (N-gram) table raises at TP=3 because its
   padded row count is not divisible by 3 (`b12x/sequence/ple_embedding/_contracts.py:381`); fix is a
   config edit (`make_ngram_vocab_size_divisible_by` 128 -> 96) plus trimming 32 unused rows from
   one PLE shard. (b) The b12x RoCE one-shot all-reduce/all-gather (`VLLM_ENABLE_ROCE_ALLREDUCE=1`
   in our production TP=2 recipe) pairs HCA index h with peer HCA index h and stripes every peer over
   every HCA, which cannot work on our switchless triangle. TP=3 must fall back to NCCL.
4. **Speed: do not expect a win.** Weight-streaming bound says at most ~+12% per stream; hyper-
   connection weights become replicated (they cannot be sharded 3 ways), attention work per rank does
   not shrink, and comm goes from RoCE one-shot to NCCL ring. Honest single-stream estimate
   **~ +3% (range -10% .. +15%)** over native TP=2's 86.2 tok/s (our `bench-miaai` number).
   KV capacity is not a reason either: the model has only 12 full-attention layers.
5. **Go/no-go:** run a no-engineering probe first (TP=2 with the two TP=3-forced regressions
   applied, Section 6 step G1). Only build the converter if that probe stays >= ~80 tok/s.

---

## 1. Per-axis decision table (every tensor family)

Legend for the decision column: **EXACT** = divides by 3 as-is; **REPL** = replicated on every rank;
**PAD** = zero-pad the sharded axis so it divides (extra lanes are exactly zero);
**COPY** = head-replicate-copy. Shapes are `[out, in]` as stored. "Global" = after conversion.

### 1.1 Language model, main 48 layers (12 full-attention + 36 GDN, all 48 MLPs are MoE)

No dense MLP exists: the shard headers contain no non-expert `mlp.gate_proj/up_proj/down_proj`
(only `mlp.experts.*`, `mlp.shared_expert.*`, `mlp.gate`, `mlp.shared_expert_gate`) [V]. The config's
`intermediate_size=12288` is unused (it divides by 3 anyway).

| Family (dtype) | Original global | TP=2 per rank (prod) | TP=3 decision | TP=3 global after conversion | TP=3 per rank | Why exact / notes |
|---|---|---|---|---|---|---|
| **q_proj** incl. sigmoid gate (MXFP8 w `[12288,2560]` F8_E4M3 + scale `[12288,80]` U8) | 24 heads x (q256 + gate256) | `[6144,2560]` (12 heads) | **PAD heads 24->36** | `[18432,2560]`, scale `[18432,80]` | `[6144,2560]` (12 heads) | New heads are appended zero rows. q=0 -> uniform softmax over a zero V head -> 0, then `o_proj` pad columns are zero anyway. Group size (q per kv) stays 12, exactly the DSv4 "hold heads-per-group, pad the group count" move. [V-CPU] rel err 2.6e-6. |
| **k_proj / v_proj** (MXFP8 `[512,2560]` + scale `[512,80]`) | 2 kv heads | `[256,2560]` (1 head) | **PAD kv 2->3** (3rd head all zero) | `[768,2560]`, scale `[768,80]` | `[256,2560]` (1 head) | kv 2->3 alone is WRONG (rank 1 would map q heads 8-11 to the wrong kv): negative control in test C, rel err 0.83. q must go to 36 together. Zero K,V rows -> `k_norm` (Gemma RMSNorm) of zeros stays finite and cache holds zeros, never garbage. |
| **o_proj** (MXFP8 `[2560,6144]` + scale `[2560,192]`) | 24x256 in | `[2560,3072]` | **PAD in-dim 6144->9216** | `[2560,9216]`, scale `[2560,288]` | `[2560,3072]` | RowParallel; pad columns zero. b12x MXFP8 needs K%128 per rank: 3072 ok. |
| q_norm / k_norm (BF16 `[256]`) | shared over heads | replicated | REPL (unchanged) | same | same | per-head-dim, not per head. |
| **QSA indexer** `index_qk_proj` (MXFP8 `[640,2560]`) + q/k layernorm `[128]` | 4 q heads + 1 kv head x 128 | replicated | **REPL** (already `ReplicatedLinear`) | unchanged | full copy | `indexer_qsa.py:132`, `b12x_indexer_qsa.py:45` [V]. Indexer heads 4/1 are never divided. Work is redundant per rank exactly as at TP=2. |
| **GDN in_proj_qkv** (MXFP8 `[10240,2560]` + scale `[10240,80]`) = q2048 + k2048 + v6144 rows | Hk16 / Hv48 x 128 | `[5120,2560]` | **PAD Hk 16->18, Hv 48->54** | `[11520,2560]` = q2304 + k2304 + v6912, scale `[11520,80]` | `[3840,2560]` (q768,k768,v2304) | V:K stays 3:1 (mandatory for b12x GDN, see 1.3). Zero q/k heads: kernel L2-norm has `+1e-6` inside rsqrt (`b12x/sequence/gdn_decode/_cute_kernels.py:416-421`, prefill `_shared/delta_prefill/_cute_kernels.py:752`), so zero heads give 0, not NaN [V]. [V-CPU] rel err 2.3e-6. Row layout matters: pad heads go at the END of each of the q, k, v blocks. |
| **GDN in_proj_z** (MXFP8 `[6144,2560]`) | Hv48 x128 | `[3072,2560]` | PAD 6144->6912 | `[6912,2560]` | `[2304,2560]` | zero gate lanes. |
| **GDN in_proj_b / in_proj_a** (MXFP8 `[48,2560]` each) | Hv48 | `[24,2560]` each (merged N=48) | PAD 48->54 | `[54,2560]` each | `[18,2560]` each (**merged N=36**) | **Alignment flag:** merged N=36 is not a multiple of 8. b12x dense linear defaults to the slower quantized path when `out_features % 8 != 0` (`b12x/gemm/blockscaled/_tuning.py:94,139,162`). Loud or slow, not silent [I]. See risk R3. |
| **GDN conv1d** (BF16 `[10240,1,4]`) | q,k,v channels | `[5120,1,4]` | PAD, same row layout as in_proj_qkv | `[11520,1,4]` | `[3840,1,4]` | pad channels zero -> silu(0)=0. |
| **GDN out_proj** (MXFP8 `[2560,6144]` + scale `[2560,192]`) | Hv48x128 in | `[2560,3072]` | PAD in-dim 6144->6912 | `[2560,6912]`, scale `[2560,216]` | `[2560,2304]` | K per rank 2304 = 18x128, ok for K%128. |
| GDN `A_log`, `dt_bias` (BF16 `[48]`), norm `[128]` | per v head | `[24]` | PAD 48->54 with **finite 0.0** | `[54]` | `[18]` | finite pad avoids exp overflow in inactive recurrent lanes (GLM lesson, `SHARDED-TP3-CANDIDATE.md`). norm weight replicated. |
| **Routed experts w13** (NVFP4: gate/up `[640,1280]` U8 packed + scale `[640,160]` F8_E4M3 + per-expert scalar `weight_scale_2`, `input_scale`) x 512 x 48 | I=640 | I=320 (`[640,1280]` fused) | **PAD I 640->768** | gate/up `[768,1280]`, scale `[768,160]` | I=256 (`[512,1280]` fused) | silu(0)*0=0. Per-rank 256 = 16 NVFP4 groups/row... 256 is a multiple of 64 (vLLM pads gated NVFP4 to 64 at `vllm/model_executor/layers/quantization/utils/b12x_moe.py:33`) and of 128 (b12x has a special n64 repack path for `I%128==64`, which TP=2's 320 uses; 256 is the plain path) [V]. [V-CPU] 3-way partial sums == unpadded, rel err 8e-7. |
| **Routed experts w2** (down `[2560,320]` U8 + scale `[2560,40]`) | I=640 | `[2560,160]` | PAD in-dim I 640->768 | `[2560,384]`, scale `[2560,48]` | `[2560,128]`, scale `[2560,16]` | pad columns zero (packed 2 fp4/byte -> 128 zero bytes per row). |
| NVFP4 scale tensors for the pad region | | | fill **0x00** (E4M3 +0) | | | Matches vLLM's own auto-pad (`F.pad` zeros, `b12x_moe.py:37-48`). Weight nibbles 0 x any finite scale = exactly 0 (also true for 0x38=1.0, tested). Padding is done on disk, so it precedes `swizzle_blockscale` (`b12x_moe.py:98,107`; the code raises if padding is needed after, line 102). Per-rank swizzle tiles: w13 scale rows 512 = 4x128 tiles, w2 scale cols 16 = 4x4. `weight_scale_2` and `input_scale` (scalars per expert/projection) are untouched. |
| **Shared expert** (MXFP8 gate/up `[640,2560]` + scale `[640,80]` U8; down `[2560,640]` + scale `[2560,20]`) | I=640 | I=320 | **PAD I 640->768** | gate/up `[768,2560]`, scale `[768,80]`; down `[2560,768]`, scale `[2560,24]` | I=256 | Native vLLM has a "replicate misaligned shared expert" path (`vllm/model_executor/models/qwen3_next.py:96-125,161`) but it only fires for Quark MX configs (`group_size` comes from Quark), so under `modelopt_mixed` it is inactive [V]. Fallback if pad misbehaves: force `replicate_shared_expert=True` (0.23 GiB replicated, +~0.6 ms/step of redundant streaming [I]). |
| MXFP8 scale tensors for the pad region | | | fill **0x7F** (E8M0 = 1.0) | | | E8M0 has no zero and no sign: 0x00 = 2^-127 (fp32 subnormal), **0xFF = NaN, never use it**. Weight bytes 0 x scale = exactly 0 for every byte except 0xFF, so 0x00 is arithmetically safe [V-CPU], but vLLM's own quantizer comment documents byte 0 flushing to zero on some hardware and producing 0/0 (`vllm/model_executor/layers/quantization/utils/mxfp8_utils.py`, "sb == 0"), and BHCC hit a 0-scale re-encode trip on DSv4.1. 0x7F costs nothing. |
| **Router gate** `mlp.gate` (BF16 `[512,2560]`), `shared_expert_gate` (BF16 `[1,2560]`) | | replicated | **REPL** (`ReplicatedLinear`) | unchanged | full copy | `qwen3_next.py:183-197` [V]. 512 experts are never split (TP shards I, not experts). |
| **Hyper-connection mixers** (BF16: down `[320,10240]`, up `[10240,320]`, block_inject `[4,10240]`, hc_norm `[10240]`; 2 per layer + final mixer = 97) | hidden 2560 x 4 streams, low-rank 320 | **sharded** by low-rank/hidden (HC_TP) | **REPL, automatic** | unchanged | full copy (1.19 GiB/rank vs 0.60) | `hyperconnection.py:207-216`: `tp_size` falls back to 1 when `hc_lowrank % tp` or `hidden % tp` != 0 (320%3=2, 2560%3=1). No code change. Consequence: **weights streamed per step double** for the HC set, but the sharded path's 2 all-gathers per HC mix (`hyperconnection.py:690,703`, ~194 per step) disappear. Cost/benefit unknown, see G1. [V] code, [I] cost. |
| **PLE / N-gram table** (NVFP4-group16: 128 shard tensors `[2500012,80]` U8 + `[2500012,10]` scale; one PLE layer) | 320,001,536 padded rows (26.8 GiB) | rows/2 per rank (13.4 GiB) | **EXACT after config fix** (row-shard by TP) | padded rows 320,001,504, shard 127 trimmed to 2,499,980 rows | 106,667,168 rows/rank (8.94 GiB) | b12x requires `padded_vocab_size % tp == 0` (`_contracts.py:381-386`). Padded = `align_up(total, table_alignment)` (`ple_hash/geometry.py:97`); `total`=320,001,446 (sum of the 16 checkpoint primes, read from the checkpoint). 320,001,536 % 3 = 2 -> boot error. With `make_ngram_vocab_size_divisible_by=96` (any of 3,6,12,24,48,96): padded=320,001,504, %3=0. Prime sizes derive only from `ngram_vocab_size_base` and layer ordinal (`reference.py:93-123`), NOT from the alignment, so the hash is unchanged [V]. The loader sizes checkpoint shards as `ceil(padded/128)=2,500,012` (unchanged for 96, so shards 0..126 load untouched) and expects the last shard to have `padded-127*2,500,012 = 2,499,980` rows (`b12x_ple.py:636-639,743-757`), hence trim 32 alignment-only rows (indices >= total, never addressed). Do NOT use 384 (=lcm(128,3)): shard rows become 2,500,014 and every shard mismatches. Lookup result is all-reduced across TP (`b12x_ple.py:620-621`). Geometry buffers are compared to the checkpoint (`b12x_ple.py:634-650`) so a wrong geometry fails loudly. |
| PLE conv1d `[10240,1,4]`, key/value proj (BF16 `[10240,2560]`,`[2560,2560]`), norms | | replicated | **REPL** | unchanged | full copy | `ple_layer.py:161-168` `disable_tp=True`; PLE conv state `tp_replicated` (`model.py:854-859`) [V]. |
| **embed_tokens / lm_head** (BF16 `[248320,2560]` each; lm_head is MXFP8 online quantized with `VLLM_MXFP8_LM_HEAD=1`) | vocab 248320 | 124160 rows | **PAD, native** | 248448 = lcm(64,3)=192 x 1294 | `[82816,2560]` (=647x128) | **No patch needed on this image**: `vllm/model_executor/layers/vocab_parallel_embedding.py:355` already does `padding_size = lcm(padding_size, tp)`, and trailing pad rows are zero-filled (`:646`). The old patch V6 is obsolete. BHCC `install()` would just re-pad 64x3=192 to the same 192. `LogitsProcessor` trims back to the real vocab [I: standard vLLM behavior, not re-read]. lm_head rows/rank %128 ok for MXFP8. |
| **Vision tower** (16 heads, hidden 1152, MLP 4304, depth 27; merger MXFP8) | 16 / 4304 not divisible | replicated (data mode) | **REPL** via `--mm-encoder-tp-mode data` (already in recipe) | unchanged | full copy 0.38 GiB | `qwen3_vl.py:412-429,515-540` use `disable_tp=is_vit_use_data_parallel()` [V]. Text-only serving can use `--language-model-only` to skip it. |

### 1.2 MTP head (`mtp.*`, 1 layer, `mtp_num_hidden_layers=1`)

Same axes as the main model, but the dtypes differ from BHCC's assumptions [V from headers]:

| Family | Dtype in OUR checkpoint | TP=3 decision |
|---|---|---|
| attention q/k/v/o, indexer | **BF16, no scale tensors** | PAD exactly as 1.1 (q 24->36, kv 2->3, o cols), no scale handling |
| routed experts (512) | **NVFP4** (U8 + E4M3 scale + scalars), same as main | PAD I 640->768, same rules as main |
| shared expert | **BF16** (`mtp.layers.0.mlp.shared_expert.*`) | PAD 640->768 zero rows/cols, no scales |
| router gate, shared_expert_gate, HC, indexer | BF16 | REPL |
| `fc_embedding`, `fc_hidden` `[2560,2560]` (2560 out not divisible) | BF16 | REPL: **already `ReplicatedLinear` when b12x is used** (`mtp.py:226-245`), so `mtp.diff` hunk 1-2 is unnecessary and its other hunks fail on our `mtp.py` |
| `pre_fc_norm_embedding [2560]`, `pre_fc_norm_hidden [10240]` | BF16 | REPL |
| MTP embed / lm_head | shared with target | native vocab pad |

The draft model reads the same `config.json` (`--hf-overrides` dict overrides apply to the target
only, BHCC learned this the hard way), so the config edit must be on disk in the model dir.

### 1.3 Why GDN must be PAD (verified kernel contracts)

- `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:354-367`: with b12x selected (default for
  `qwen4_exp_text` when `moe_backend` or `linear_backend` is b12x, `:307-317`), the layer raises
  `ValueError("b12x GDN prefill requires SM12x, BF16, 128-wide heads, and V:K heads=3:1")` unless
  `linear_num_value_heads == 3 * linear_num_key_heads` **on the global config** [V].
  Also `b12x/sequence/gdn_prefill/_impl.py:54-55`. Decode only needs `Hv % Hk == 0`
  (`gdn_decode/_impl.py:86`).
- COPY (Hk 16->48, Hv 48) is 1:1 -> ValueError, unless you override both GDN backends away from
  b12x (`gdn_prefill_backend` flashinfer/triton, decode cuda/triton), giving up the kernels that
  production TP=2 uses on 36 of 48 layers. The math of COPY is exact [V-CPU] but the kernel
  contract is not.
- PAD (Hk 18, Hv 54, per rank 6:18) keeps 3:1, per-rank packed qkv width 3840.
- b12x QSA only needs `q_heads % kv_heads == 0`, `head_dim==256` (`b12x/attention/qsa/_contract.py:284-306`,
  `_sparse_gqa_cute_config.py:49`), so both (12,1) and (8,2) are legal there; (12,1) is the geometry
  already exercised at TP=2.

### 1.4 Divide-site audit for the padded config (all read, none need patching)

| Site | File:line | Check with padded config (heads 36, kv 3, Hk 18, Hv 54, I 768) at tp=3 | Result |
|---|---|---|---|
| Model-level head check | `vllm/config/model.py:1443` | 36 % 3 | ok |
| Full-attn (non-b12x) | `qsa.py:207-221` | heads 36%3, kv 3>=3 and 3%3 | ok |
| Full-attn (b12x, production path) | `b12x_qsa.py:903-912` | same | ok |
| GDN state shape | `mamba_utils.py:274-295`, `model.py:787-801` | divide(conv_dim=11520,3), divide(54,3) | ok |
| GDN layer | `qwen_gdn_linear_attn.py:735-743,841-842` | divide(54,3), divide(18,3) | ok |
| MoE / shared expert | `qwen3_next.py:217-237`, FusedMoE | 768/3, `_should_replicate...` inactive | ok |
| b12x MoE alignment | `b12x_moe.py:33` | round_up(256,64)=256, no extra pad | ok |
| HC workspace | `hyperconnection.py:207-216` | falls back to replicated | ok (cost, see 3) |
| PLE | `_contracts.py:381` | needs config edit | **blocker fixed by config** |
| Vocab | `vocab_parallel_embedding.py:355` | native lcm | ok |
| Vision | `qwen3_vl.py` | data mode | ok |
| MTP fc_* | `mtp.py:226` | ReplicatedLinear | ok |

---

## 2. PLAN-COPY vs PLAN-PAD

PLAN-COPY = KV 2->6 by 3x head replication, GDN q/k 16->48 by 3x (BHCC2025).
PLAN-PAD = KV 2->3 with a zero head + q 24->36, GDN 16/48->18/54 (our SGLang and old virtual-TP plans).
PLAN-H (variant worth an A/B later) = attention COPY + GDN PAD.

| Metric | TP=2 (prod) | **PLAN-PAD** | PLAN-COPY | PLAN-H |
|---|---|---|---|---|
| b12x kernel contract | ok | **ok** (12q:1kv, GDN 6:18) | **GDN ValueError** (1:1) unless GDN falls off b12x | ok GDN; attention (8q:2kv) untested geometry |
| Weights per rank, all-in (GiB, incl. 0.38 vision, MTP; computed from headers) | 49.6 | **38.6** | ~38.7 | ~38.7 |
| Checkpoint on disk (GiB) | 98.5 | ~112 | ~112 | ~112 |
| KV bytes per token per rank, 12 full-attn layers, fp8 (prod recipe) | 6 KiB | **6 KiB** | 12 KiB | 12 KiB |
| KV bytes per token per rank if BF16 | 12 KiB | 12 KiB | 24 KiB | 24 KiB |
| KV tokens (illustrative: 0.7 x 121 GiB budget, 15 GiB overhead assumed [I], fp8) | ~2.8 M | ~4.3 M | ~2.4 M | ~2.4 M |
| Attention weights read per rank per layer (MB) | 24.9 | 24.9 | 18.9 | 18.9 |
| QSA kv gather bytes per rank (relative) | 1.0 | 1.0 | 2.0 | 2.0 |
| Idle attention rank | none | **rank 2 does all-zero attention (1/3 of attention lanes wasted)** | none | none |
| GDN weights read per rank per layer (MB) | 28.8 | **21.6** (0.75x) | 27.5 (0.95x) | 21.6 |
| GDN state per sequence per rank (36 layers) | ~55 MiB | ~41 MiB | ~41 MiB (state depends on Hv only) | ~41 MiB |
| MoE weights read per rank per layer (I) | 320 | 256 (0.8x) | 256 | 256 |
| All-reduce payload per layer | 2 x T x 2560 x 2 B | identical (2 x T x 5 KiB) | identical | identical |
| Ring all-reduce factor 2(P-1)/P; steps | 1.0; 2 | 1.33; 4 | 1.33; 4 | 1.33; 4 |
| FLOPs per rank per token (GFLOP, decode/prefill GEMM only, excl. lm_head) | 5.9 | **5.5** | ~5.6 | ~5.4 |
| Conversion work | n/a | q/k/v/o + GDN (5 proj + conv + A/dt) + experts + shared + PLE trim | k/v(+scale) x3, GDN q/k x3 (+scale), experts, shared, PLE trim | k/v x3, GDN pad, experts, shared, PLE |

FLOPs per token (GFLOP): experts 4.7, GDN 4.1, full-attn 1.2, shared 0.47, HC 1.28 (replicated at TP=3),
lm_head 1.27 sampled rows only. At TP=3 the prefill compute win is only ~7% per rank because HC
becomes replicated and attention does not shrink.

**Which is better for decode speed: PLAN-PAD.** (1) It is the only one that keeps the b12x GDN
decode+prefill kernels (verified contract). (2) Its attention geometry equals production TP=2's, so
kernel tuning tables and captured graphs are known-good; COPY doubles per-rank KV gather (the QSA
gather bucket is ~12% of kernel time in sglang#36796) and cuts only q/o GEMM bytes (~1% of step).
(3) Memory and KV are non-issues in both. PAD's costs are real but small: rank 2's attention is
pure padding (no speedup on that bucket) and ~14 GiB more on disk. Unbalanced real work also means
"PAD attention" at TP=3 is as slow as TP=2 attention, not faster.

---

## 3. What a third node buys per stream (honest estimate)

### 3.1 First-principles bound: weights streamed per rank per verify step

Model: batch-1 decode with MTP=4, so 5 rows are verified per step; each step also runs 4 MTP draft
passes, each with a full lm_head read. Experts touched per layer per step are estimated as
512 x (1-(1-10/512)^B) (independent routing; real routing is correlated, so treat as an upper
bound). MXFP8 lm_head (prod uses `VLLM_MXFP8_LM_HEAD=1`). Computed from the shard headers
(`/tmp/bw.py`, reproduced in Appendix B):

| Component (GB per rank per step, B=5) | TP=2 | TP=3 PAD | ratio |
|---|---|---|---|
| routed experts (48.1 distinct/layer) | 3.19 | 2.55 | 0.80 |
| GDN | 1.08 | 0.81 | 0.75 |
| full attention + indexer | 0.33 | 0.33 | 1.00 |
| shared expert | 0.12 | 0.10 | 0.80 |
| hyper-connection mixers | 0.64 (sharded) | 1.28 (replicated) | 2.00 |
| router gates | 0.13 | 0.13 | 1.00 |
| lm_head x5 (MXFP8) | 1.69 | 1.13 | 0.67 |
| MTP layer x4 (crude) | 0.78 | 0.77 | 0.99 |
| **total** | **7.95** | **7.09** | **0.89** |

At 273 GB/s that is 29.1 ms vs 26.0 ms; at B=1 the ratio is 0.93. Observed steps are ~35-40 ms
(86 tok/s at ~3.3-3.5 accepted tokens/step [I]), so the step is not bandwidth-bound and a
3-way split can recover at most ~11% of the bandwidth part. If the HC set could be sharded 3 ways
the ratio would be 0.81, but it cannot (hidden 2560 and low-rank 320 are both not divisible by 3).

### 3.2 Kernel-time composition (sglang#36796: NVFP4 GEMM 34%, MoE 16%, QSA gather 12%, NCCL 7%, other 31%)

| Bucket | Share at TP=2 | TP=3 factor: low / central / high | Reason |
|---|---|---|---|
| NVFP4/dense GEMM (attn, GDN, shared, lm_head) | 34% | 0.72 / 0.77 / 0.82 | attn 1.0, GDN 0.75, shared 0.8, lm_head 0.67, weighted by bytes |
| routed MoE | 16% | 0.78 / 0.80 / 0.85 | per-rank I 256 vs 320; kernels stream the padded tiles |
| QSA gather + attention | 12% | 0.95 / 1.00 / 1.05 | PAD geometry identical per rank |
| Comm | 7-9% | 1.3 / 1.8 / 2.4 | RoCE one-shot AR+AG (TP=2 prod) -> NCCL 3-rank ring, 4 steps, per-op latency ~45-70 us vs ~12-20 us [I]; HC all-gathers vanish (-194 ops), partly offsetting |
| Other (HC compute, GDN recurrent, norms, sampling, launch gaps) | 30-31% | 0.95 / 1.08 / 1.20 | GDN kernels x0.75; HC replicated x2 weights; fixed overheads unchanged |
| **Total step time vs TP=2** | 1.00 | **0.87 / 0.98 / 1.12** | |

Speedup: **+15% optimistic, +2% central, -10% pessimistic**. Applied to native TP=2's 86.2 tok/s
(`bench-miaai`, c=1): **~76 .. 99 tok/s, central ~88**.

Anchors: (a) the 3.1 bound (+12% max); (b) the only measured 3-vs-2 on this model, SGLang padded
TP=3 vs TP=2 at c=1, +14% (75.7 vs 66.5), c=16 aggregate +33% (383 vs 289) -
`RESULT-2026-09-04-SGLANG-FLASH-NEXT-TP2-VS-TP3.md`. That is the optimistic end: SGLang TP=2 is a
slower baseline and pays NCCL at both sizes. (c) DSv4 matched test: +8-17% per stream at long context.
The native TP=2 is already the fastest single-stream configuration we own (86.2 vs 72-74 for SGLang
TP=3), so the third node has to overcome a better baseline.

Uncertainty is dominated by two unmeasured items, both testable **without building anything**:
HC replicated vs sharded, and NCCL vs RoCE one-shot collectives (Section 6, G1).

Aggregate throughput (c>=8) is where 3 nodes should help more (weight streaming dominates as batches
grow; SGLang saw +33% at c=16): plausible +10-25% for vLLM [I]. Prefill/TTFT: ~7% fewer FLOPs per
rank, so no meaningful TTFT change.

---

## 4. Patch skeleton for the native image

### 4.1 What is NOT needed (verified)

| Community/old piece | Verdict on `vllm-node-b12x:latest` |
|---|---|
| BHCC `model.diff` (hook `Qwen4ExpForConditionalGeneration.load_weights`, `model.py:1086`) | Applies (our dry run), but **its transform cannot run under `--load-format b12x`**: the b12x loader yields meta tensors and registers their storage as checkpoint sources (`b12x/loader/_checkpoint.py:348-355`); any tensor op that creates new data (`repeat_interleave`, `zeros`+`copy_`, `cat`) produces a storage that is not registered and raises `NotImplementedError("checkpoint routing performed an unsupported data transformation")` (`:397-400`). Loud, not silent. Only `narrow`/view slices are supported (byte offsets are recorded, `:405-437`). |
| BHCC `mtp.diff` (3 of 4 hunks fail) | Not needed: `fc_embedding`/`fc_hidden` are `ReplicatedLinear` under b12x (`mtp.py:226-245`); MTP experts in our checkpoint are NVFP4, not FP8-block (their `weight_scale_inv` branch is dead for us). |
| BHCC `install()` vocab padding | Redundant (native lcm at `vocab_parallel_embedding.py:355`). |
| BHCC `tp_pad._transform` | Transforms only names ending `.weight`; our MXFP8 `k_proj/v_proj/in_proj_qkv/in_proj_z/...` carry `weight_scale` companions (U8 `[N,80]`), which it leaves at original size. COPY GDN also violates 3:1. Reuse its `_pad_dim`/`_rep_heads` ideas only. |
| Old virtual-TP layer (22 files, `eugr/spark-vllm-b12x:local-20260823`) | Not portable: `virtual_tp.py` does not exist in the new image, and the native model class replaced the Qwen3-Next port. Its *plan* (Q 24->36, KV 2->3, GDN 18/54, MoE 768) is what we reuse. |
| DSv4 `apply_tp3_patch.py` | Targets `deepseek_v4` attention; different anchors. Only the technique transfers. |

### 4.2 What is needed: offline padded checkpoint + config + env (no code patch)

**Why offline:** the b12x loader streams checkpoint bytes straight from O_DIRECT reads into the
destination parameter slices, so the conversion must already be in the safetensors on disk.
Hardlink every untouched shard, write new shards for modified tensors, write a new
`model.safetensors.index.json`. The loader honors the index (`_checkpoint.py:328-337` skips a
tensor if the index maps it to a different file), so stale unpadded copies inside hardlinked
shards are ignored.

**config.json edits** (in `text_config`; the model dir must be local, `--hf-overrides` does not reach the MTP draft):

| Key | Old | New |
|---|---|---|
| `num_attention_heads` | 24 | 36 |
| `num_key_value_heads` | 2 | 3 |
| `linear_num_key_heads` | 16 | 18 |
| `linear_num_value_heads` | 48 | 54 |
| `moe_intermediate_size` | 640 | 768 |
| `shared_expert_intermediate_size` | 640 | 768 |
| `make_ngram_vocab_size_divisible_by` | 128 | 96 |

`vocab_size`, `hc_*`, `indexer_*`, `head_dim` stay. `quantization_config.quantized_layers` is keyed by
layer names only (no shapes), so it stays. `hf_quant_config.json` stays. (`export-manifest.json` is
not read by vLLM [I]; regenerate or drop it.)

**Converter rules** (`scripts/tp3_pad_checkpoint.py`, new, ~200 lines; UNTESTED skeleton):

```python
# name suffix -> (axis, old, new, fill_for_weight, fill_for_scale_or_None)
# All appended at the END of the axis. GDN q/k/v blocks are padded per block.
ATTN = {   # layers.{3,7,...,47}.self_attn.* and mtp.layers.0.self_attn.*
  "q_proj.weight":       (0, 12288, 18432, 0,    None),  # rows = heads*512
  "k_proj.weight":       (0,   512,   768, 0,    None),
  "v_proj.weight":       (0,   512,   768, 0,    None),
  "o_proj.weight":       (1,  6144,  9216, 0,    None),
  # MXFP8 scales (main only; MTP attention is BF16 with no scale tensors)
  "q_proj.weight_scale": (0, 12288, 18432, 0x7F, None),
  "k_proj.weight_scale": (0,   512,   768, 0x7F, None),
  "v_proj.weight_scale": (0,   512,   768, 0x7F, None),
  "o_proj.weight_scale": (1,   192,   288, 0x7F, None),  # cols = K/32
}
GDN = {    # layers.{0,1,2,4,...}.linear_attn.*  (36 layers)
  "in_proj_qkv.weight": blockpad([2048, 2048, 6144] -> [2304, 2304, 6912], axis=0, fill=0),
  "in_proj_qkv.weight_scale": same blocks, fill 0x7F,
  "conv1d.weight": same blocks on axis 0, fill 0,          # [10240,1,4] -> [11520,1,4]
  "in_proj_z.weight": (0, 6144, 6912, 0), "in_proj_z.weight_scale": (0, 6144, 6912, 0x7F),
  "in_proj_b.weight": (0, 48, 54, 0),     "in_proj_b.weight_scale": (0, 48, 54, 0x7F),
  "in_proj_a.weight": (0, 48, 54, 0),     "in_proj_a.weight_scale": (0, 48, 54, 0x7F),
  "out_proj.weight": (1, 6144, 6912, 0),  "out_proj.weight_scale": (1, 192, 216, 0x7F),
  "A_log": (0, 48, 54, 0.0), "dt_bias": (0, 48, 54, 0.0),  # finite pad
}
EXPERT = {  # layers.*.mlp.experts.E.* and mtp.layers.0.mlp.experts.E.*  (NVFP4)
  "gate_proj.weight": (0, 640, 768, 0),  "up_proj.weight": (0, 640, 768, 0),   # U8 [640,1280]
  "gate_proj.weight_scale": (0, 640, 768, 0x00), "up_proj.weight_scale": (0, 640, 768, 0x00),  # E4M3 [640,160]
  "down_proj.weight": (1, 320, 384, 0), "down_proj.weight_scale": (1, 40, 48, 0x00),
  # weight_scale_2, input_scale: scalars, unchanged
}
SHARED = {  # layers.*.mlp.shared_expert.*  MXFP8 main; mtp.* BF16 (no scales)
  "gate_proj.weight": (0, 640, 768, 0), "up_proj.weight": (0, 640, 768, 0),
  "gate_proj.weight_scale": (0, 640, 768, 0x7F), "up_proj.weight_scale": (0, 640, 768, 0x7F),
  "down_proj.weight": (1, 640, 768, 0), "down_proj.weight_scale": (1, 20, 24, 0x7F),
}
PLE = { "layers.1.ple.ple_embedding.ngram_embedding.shard_127.weight": trim rows 2500012 -> 2499980,
        "...shard_127.weight_scale": trim rows 2500012 -> 2499980 }   # only for alignment=96
```

Implementation notes: view dtype as `uint8` for fp8/U8/E4M3 so fills are byte-exact; write
with `safetensors.torch.save_file` (same format as the originals); process one tensor at a time
(no whole-shard loads); drop page cache afterwards (unified memory, see R7). Idempotent and
`--check` mode = header-only audit (Section 5, T-B).

**Launch delta vs the TP=2 recipe** (`configs/colonel-qwen3.8-flash-next-nvfp4-tp2-pinned.yaml`):

| Item | TP=2 | TP=3 |
|---|---|---|
| model path | HF repo + `--revision` | local padded dir (all 3 nodes) |
| `--tensor-parallel-size` | 2 | 3 |
| `VLLM_ENABLE_ROCE_ALLREDUCE` | `1` | **`0`** (b12x RoCE pairs HCA index h with peer HCA h and stripes all peers over all HCAs, `b12x/comm/roce/_roce_proxy.c:290-330,389-465`, `roce_oneshot.py:236-240`; a triangle has one peer per port). The wrapper degrades gracefully on init failure (`b12x_roce_all_reduce.py:100-118`), but a connect that succeeds locally and fails on first write is the DSv4 `IBV_WC_RETRY_EXC_ERR` case; do not risk it. |
| NCCL | per-node HCA pin | DSv4 recipe: `NCCL_IB_SUBNET_AWARE_ROUTING=1`, `NCCL_NET_PLUGIN=none`, both HCAs, mesh routes (`dsv4-traps-and-flags.md`) |
| `--mm-encoder-tp-mode data`, `--load-format b12x`, `--gdn-decode-kernel b12x`, `--kv-cache-dtype fp8`, MTP 4 | keep | keep |
| `VLLM_QWEN3_8_FLASH_NEXT_HC_TP` | 1 (effective) | irrelevant (auto-replicated) |

### 4.3 Code sites (read-only inventory; all are no-ops with the padded config)

| Purpose | File:line | Change |
|---|---|---|
| Full-attn divisibility (non-b12x) | `vllm/models/qwen4_exp/nvidia/qsa.py:207-221` | none |
| Full-attn divisibility (b12x path) | `.../b12x_qsa.py:903-912` | none |
| Model load entry points | `.../model.py:613-654` (`Qwen4ExpModel.load_weights`), `:876-884`, `:1086-1095` | none (no load-time transform) |
| MTP load | `.../mtp.py:833-855`, `_remap_mtp_weight_name` `:118` | none |
| GDN sizes from config | `qwen_gdn_linear_attn.py:654-660,735-743,1449-1470`; `qwen3_5.py:223-224` (`in_proj_qkv` -> shards (0,1,2)) | none |
| Vocab | `vocab_parallel_embedding.py:355` | none (native) |
| HC | `hyperconnection.py:207-216` | none (auto fallback) |
| PLE | `_contracts.py:381`, `ple_hash/geometry.py:97`, `b12x_ple.py:636-757` | config only + shard trim |
| **Optional fallbacks** (only if a GPU test fails) | (a) `qwen_gdn_linear_attn.py:1412-1428` `maybe_disable_tp` -> force True for `in_proj_ba` if N=36 misbehaves (needs unpadded `in_proj_b/a` rows + slice `split_ba:1430-1438`); (b) `qwen3_next.py:161` force `replicate_shared_expert=True` if MXFP8 pad misbehaves | write only on evidence |

### 4.4 b12x kernel interactions with padded shapes (read)

| Kernel | Constraint | Padded shape | Source |
|---|---|---|---|
| MoE NVFP4 (b12x) | per-rank gated half-size padded to multiple of 64 by vLLM before scale swizzle; kernel extent used as-is for NVFP4; `size_n % 64` in repack helpers | 256 | `b12x_moe.py:33-49,98-108`; `b12x/moe/fused_moe/_impl.py:4551,5191`; `flashinfer_b12x_moe.py:76-78` |
| MoE FC2 activation quant | all-zero tile guarded (`scale=max(amax>0 ? c/amax : 0, 1e-12)`) | pad columns produce all-zero blocks | `b12x/moe/_shared/kernels/dynamic.py:1860-1862` [V]; per-16-block helper `nvfp4_scale_from_amax` not read -> R5 |
| MXFP8 dense linear | K%32 assert; K%128 for quantized activations; N%8 for the default A16 fast path | K: 3072, 2304, 256, 2560 ok; N: 6656, 6144, 512 ok; **merged in_proj_ba N=36 not %8** | `vllm/model_executor/kernels/linear/mxfp8/b12x.py:63,78-80`; `b12x/gemm/blockscaled/_tuning.py:94,139,162`; `gemm/_shared/wo_mxfp8.py:444` |
| GDN decode/prefill | head dim 128; prefill V:K=3:1; decode Hv%Hk==0 | 6:18 per rank | `gdn_prefill/_impl.py:54`, `gdn_decode/_impl.py:86` |
| QSA (CuTe sparse GQA) | q_heads % kv_heads == 0, head_dim 256, selection width >= 2051; grid `(rows, kv_heads, splits)` | (12,1) | `attention/qsa/_contract.py:284-306`; `attention/paged/_selected_forward.py:143-170,323` |
| PLE embedding | `padded_vocab % tp == 0`; shard rows `ceil(padded/split_ngram_parts)`; NVFP4 group16 | 320,001,504 | `_contracts.py:381-386` |
| Loader | meta-tensor views only, exact shape equality, no dtype change for views | converted on disk | `_checkpoint.py:357-437` |
| Comm | b12x PCIe collectives are single-node; multi-node uses NCCL; RoCEnante disabled | NCCL ring | `cuda_communicator.py:89-92,490-509` |

---

## 5. CPU-only correctness test plan (before any GPU boot)

All runnable with `docker run --rm --memory 8g --cpus 4 --entrypoint python3 -v ~/.cache/huggingface:/hf:ro ...`
(no `--gpus`). Memory need is < 2 GB; **do not run against a node whose page cache matters while SGLang is live** without `ionice`/`--memory` limits (sparkmain showed 3 GiB free / 35 GiB available when checked).

| ID | Test | Status |
|---|---|---|
| **T-A** | Real-tensor exactness per family: dequantize MXFP8 (`w * 2^(e-127)`) and NVFP4 (fp4 x E4M3 block scale x `weight_scale_2`), run reference math with original shapes vs padded shapes split across 3 emulated ranks with vLLM's slicing (contiguous row/col slices, per-block q/k/v slicing for GDN), compare in fp32. Covers: routed expert 640->768 (both pad-scale fills), shared expert MXFP8 (0x00 and 0x7F), full-attention PAD (36/3) and COPY (24/6) incl. sigmoid gate and causal softmax, negative control (kv 2->3 without q pad must FAIL), GDN 16/48 vs 18/54 (recurrent delta rule, conv, L2 norm with eps, A_log/dt gating, gated RMSNorm, out_proj) and COPY 48/48. | **DONE, ALL PASS** (Appendix A). Errors 8e-7 .. 2.6e-6 relative. |
| T-B | Converter header audit (`--check`): for every tensor pattern in the 394-pattern table, assert new shape/dtype equals the formula; real regions byte-identical to source (`view(uint8)` equality); pad regions equal the intended fill; every original tensor name appears exactly once in the new index; hardlinked shards byte-identical (`sha256`); tensor count preserved. | to write |
| T-C | **NaN-poison coverage** (the DSv4 lesson: zeros can hide a bug that NaNs expose): allocate the per-rank destination buffers implied by the padded config (shapes from formulas) filled with NaN, replay vLLM's shard narrowing for tp_rank 0/1/2 over the padded tensors (`w13` row narrow `256*rank`, `w2` col narrow, `qkv` head narrow, GDN per-block narrow, PLE row range `[rank*106667168, (rank+1)*106667168)`), assert no NaN remains and that pad lanes are exactly zero. | to write |
| T-D | PLE geometry, pure Python (no CUDA): call `b12x.sequence.ple_hash.reference.ple_table_geometry` + `align_up` with `table_alignment` in {128, 96}; assert 128 -> padded%3=2 (reproduces the failure), 96 -> padded=320,001,504, %3=0, `ceil(padded/128)=2,500,012`, last-shard expected rows = 2,499,980, every index `< total`; compare against `ngram_heads_vocab_sizes`/offsets read from the checkpoint; assert the 3 rank slices tile `[0,padded)` exactly once. Already verified numerically in this session (numbers in 1.1). | numbers verified; script to write |
| T-E | Config lint: instantiate `Qwen4ExpTextConfig` from the padded `config.json` on CPU, then evaluate every predicate in 1.4 with tp=3 in pure Python (heads, kv, GDN divides, `value_heads==3*key_heads`, MoE divides, HC fallback, vocab lcm, PLE `%tp`). | to write |
| T-F | Repeat T-A over more layers (0, 3, 10, 23, 47, MTP) and 5 random experts each; add multi-token prefill for GDN (chunked path equals recurrent). | to write |
| T-G (GPU, later, single node) | One-layer kernel checks against the CPU reference: b12x MoE at I=256 with zero-tail vs I=640 unpadded (NaN-poisoned pad); MXFP8 GEMM N=36 K=2560; QSA (12,1) with a zero kv head; b12x GDN 6:18 with zero heads. | not now |
| T-H (GPU, after boot) | Teacher-forced logprob gate: native TP=3 vs native TP=2 on the same engine (clean reference, unlike GLM where TP=2 vs TP=3 differed 19/23 top-1 unexplained), >=500 tokens across code/prose/long-context; require |dlogprob| within 2x of TP=2-vs-TP=2 rerun drift; then 6/6 battery + tool-call validate. Beware QSA `persistent_topk` non-determinism on GB10. | not now |

---

## 6. Ranked risks and ordered go/no-go

### Risks (highest first)

| # | Risk | Likelihood / impact | Evidence | Mitigation |
|---|---|---|---|---|
| R1 | **No speed win, possible regression vs 86.2 tok/s**: HC weights double (`hyperconnection.py:207-216`), attention does not shrink, comm moves from RoCE one-shot to NCCL ring | High / decides the project | Sections 3.1-3.2 | G1 probe before building |
| R2 | b12x RoCE one-shot cannot serve a triangle; if left enabled it may pass init and fail on first write (`IBV_WC_RETRY_EXC_ERR`, as DSv4 saw) | Certain if left on / medium | `_roce_proxy.c:290-330`, `roce_oneshot.py:236-240` | `VLLM_ENABLE_ROCE_ALLREDUCE=0`; optional later: per-peer HCA selection patch in b12x C proxy |
| R3 | merged `in_proj_ba` N=36 misses b12x's N%8 fast path / may hit a workspace-quant raise | Medium / 36 small GEMMs slower or boot error | `_tuning.py:94,139,162` [I on behavior] | T-G microbench; fallback 4.3(a) |
| R4 | PLE geometry edit wrong -> loud (loader compares geometry buffers and shard shapes) | Low silent / medium loud | `b12x_ple.py:634-650,743-757` | T-D/T-B |
| R5 | zero-block handling in b12x MoE per-16-block FC2 quant (tile-level guard verified, per-block helper not read) -> NaN in padded columns | Low-medium / severe (poisons all tokens, loud NaNs) | `dynamic.py:1860-1862` | T-G NaN-poison; if it fails, use I=672 (real 224/rank, vLLM auto-pads to 256 with the same zero tails) or nonzero-eps pad |
| R6 | Silent wrongness: fluent but slightly wrong output from a padded shard (GLM sharded TP3 showed 19/23 top-1 vs TP=2, cause unexplained) | Medium / severe | `SHARDED-TP3-CANDIDATE.md` | T-A done; T-C; T-H gate; never quote tok/s before T-H |
| R7 | Conversion I/O on unified-memory nodes: ~112 GiB per node, page cache starves the GPU allocator (observed on this cluster), SGLang currently live | Medium / node wedge risk | handoff 2026-09-04 notes; `dsv4-traps-and-flags.md` signature 5 | run only with the cluster quiet, stream tensor by tensor, `drop_caches`, one node converts then `rsync` over the mesh |
| R8 | Hybrid KV-cache page sizing shifts (GDN state shrinks to ~41 MiB) -> block size / kernel block-size multiples change | Low / boot error | `qsa.py:76` `MultipleOf(16)`; `_contract.py:120-134` needs page % 8 | recipe pins `--block-size 16` already |
| R9 | Upstream instability independent of TP: MTP=4 async illegal-memory-access with QSA capture metadata (PR #865 not in image), MTP acceptance decay (sglang #37326 analog) | Medium / existing | `TRIAL-LOG-2026-09-29.md` | keep the TP=2 mitigations (`--max-cudagraph-capture-size 80`) |
| R10 | Rank 2 attention lanes are pure waste; PAD gives no attention speedup | Certain / small | Section 2 | accept; A/B PLAN-H later |
| R11 | Each node needs the 112 GiB padded copy (3 x 112 GiB) | Low | | disk check first (`df`) |

### Ordered go/no-go

| Step | Action | Cost | Go criterion |
|---|---|---|---|
| G0 | CPU exactness (T-A) - **done, all pass** | 0 | met |
| **G1** | **No-build probe on the production stack, TP=2, same harness:** run with `VLLM_QWEN3_8_FLASH_NEXT_HC_TP=0` (HC replicated, as TP=3 will force) and with `VLLM_ENABLE_ROCE_ALLREDUCE=0` (NCCL). Median of 5 at c=1 plus c=16 aggregate. | 1-2 cluster hours, 2 nodes, no code | If both together stay >= ~80 tok/s (i.e. within ~7% of 86.2), TP=3 has headroom to at least tie. If < 70, stop: **NO-GO**. Per-variable results also tell which regression to attack (HC MXFP8 mixers, per-peer RoCE). |
| G2 | Write converter + T-B/T-C/T-D/T-E (CPU) | ~1 day | all pass on real headers |
| G3 | Convert on one node with the cluster quiet, hash-audit, `rsync` to the other two | ~1 h I/O | hashes match |
| G4 | T-G one-layer GPU checks (I=256 MoE with NaN-poisoned tails, N=36 MXFP8 GEMM, QSA (12,1) zero kv head) on one GPU | ~2 h | no NaN, rel err <= TP=2 kernel noise |
| G5 | Boot TP=3 (SGLang stopped by the operator), `RoCE=0`, MTP 4; read log for: HC fallback, RoCEnante disabled, PLE geometry, MoE backend `b12x`, GDN `b12x` | 1 boot | all as predicted |
| G6 | T-H logprob gate vs native TP=2 + 6/6 battery + tool-call validate | ~2 h | pass |
| G7 | Benchmark policy sweep (256-token window, n>=5, exclusivity) vs 86.2 (c=1) and 408 (c=16) | ~2 h | Publish only if T-H passed |

**Recommendation:** GO for G0-G1 now (G0 already done). Everything after G1 is conditional. Even on
success expect +0..10% single-stream and a possible +10-25% aggregate; the third Spark already has a
useful job as the independent solo instance. Non-TP decompositions for the third node are a separate
design question (`docs/NON-TP-DECOMPOSITIONS.md` if present).

---

## 7. Verified vs inferred ledger

**Verified by reading code or running a command** (file:line in the image unless noted):

- Checkpoint: 394 tensor patterns, no dense MLP, MTP experts NVFP4 / MTP attention+shared BF16, shared expert and all attention/GDN projections MXFP8 with U8 E8M0 `weight_scale`, PLE 128 shards x `[2500012,80]`; sizes by family (experts 63.3 GiB, PLE 26.8, embed+lm_head 2.4, GDN 2.0, HC 1.2, attn 0.57, vision 0.38, shared 0.23).
- PLE: `total=320001446`, padded 320,001,536 (%3=2), alignment 96 gives 320,001,504 (%3=0), shard rows 2,500,012, last shard 2,499,980 (executed on real checkpoint tensors `ngram_heads_vocab_sizes`).
- All file:line facts cited above, incl. `qsa.py:207-221`, `b12x_qsa.py:903-912`, `hyperconnection.py:207-216,690,703`, `vocab_parallel_embedding.py:355,646`, `qwen_gdn_linear_attn.py:307-367`, `gdn_prefill/_impl.py:54`, `b12x_moe.py:33-49,98-108`, `_checkpoint.py:348-437`, `mtp.py:226-245`, `_contracts.py:381-386`, `geometry.py:97`, `b12x_ple.py:620-757`, `cuda_communicator.py:89-92,490-509`, `_roce_proxy.c:290-465`, `config/model.py:1443`, `qwen3_next.py:96-125,183-197`, `mxfp8/b12x.py:63,78-80`, `_tuning.py:94,139,162`.
- BHCC pieces: `model.diff` applies, `mtp.diff` 3/4 hunks fail (dry runs), `tp_pad._transform` only matches `.weight`.
- Recipe: production TP=2 uses RoCE all-reduce, fp8 KV, b12x loader, `--mm-encoder-tp-mode data`, util 0.7.
- CPU exactness: Appendix A, all pass.

**Inferred (needs GPU/measurement):** step-time model and every speed number in Section 3; per-op NCCL vs RoCE latencies; N=36 behavior; per-16-block zero-quant guard; whether `ModelOpt` post-load code tolerates the padded tensors end to end (only read, not run); LogitsProcessor trimming of pad columns; `export-manifest.json` unused; KV-token estimates (overhead 15 GiB assumed); accepted tokens/step (~3.3-3.5).

---

## Appendix A. CPU exactness test (run, all pass)

`~/agent-scratch/tp3_cpu_check.py` on sparkmain. Output of the run:

```
PASS NVFP4 pad dequants to exactly 0 with scale=0x00
PASS routed expert 640->768, 3-way TP partial sums == unpadded (scale=0x00) rel err 8.38e-07
PASS NVFP4 pad dequants to exactly 0 with scale=0x38(1.0)
PASS routed expert 640->768, 3-way TP partial sums == unpadded (scale=0x38(1.0)) rel err 8.38e-07
PASS NVFP4 real scales byte-identical after pad
PASS MXFP8 pad dequants to exactly 0 with E8M0=0x00 (2^-127)
PASS shared expert 640->768 3-way == unpadded (E8M0=0x00 (2^-127)) rel err 3.63e-07
PASS MXFP8 pad dequants to exactly 0 with E8M0=0x7F (1.0)
PASS shared expert 640->768 3-way == unpadded (E8M0=0x7F (1.0)) rel err 3.63e-07
PASS attention PAD (q 36/kv 3, per rank 12q:1kv) == original 24/2, rel err 2.56e-06
PASS attention COPY (q 24/kv 6, per rank 8q:2kv) == original, rel err 2.53e-06
PASS NEGATIVE CONTROL: kv 2->3 with q left at 24 is wrong (rel err 8.26e-01); q must pad to 36
PASS GDN PAD 16/48 -> 18/54 (zero heads, per rank 6k:18v, ratio 3:1) == original, rel err 2.27e-06
PASS GDN COPY 16/48 -> 48/48 (per rank 16k:16v, ratio 1:1) == original, rel err 1.93e-06
ALL PASS
```

What it does and does not prove: real layer-0 expert 3 / shared expert / GDN and layer-3 attention tensors,
dequantized to fp32, reference math (dense causal attention with sigmoid output gate; recurrent gated
delta rule with conv1d, L2-norm eps 1e-6, softplus/A_log gating, gated RMSNorm, out_proj), 3 emulated
ranks with contiguous slicing. It proves the plan's algebra and slicing/ownership (including the
per-rank q-to-kv mapping and the GDN v-head-to-k-head pairing at rank 2). It does not exercise b12x
kernels, RoPE/QK-norm (per-head-dim ops that commute with head replication/padding), or activation
quantization. Layout assumptions taken from vLLM: q_proj rows are per-head `[q256 | gate256]`
(`qwen3_next.py:426-431`); GDN q/k/v blocks are sliced per block per rank (`mamba_v2_sharded_weight_loader`,
`qwen_gdn_linear_attn.py:718-726`).

Core of the script (unchanged from the run; helpers `mx`, `nv`, `padrows`, `padcols`, `attn_local`,
`gdn_local`, `build_gdn`, `gdn_ranks` are in the file):

```python
def mx(w, s):  # MXFP8 [N,K] e4m3 + [N,K/32] uint8 E8M0
    N, K = w.shape
    return (w.float().view(N, K//32, 32) * torch.exp2(s.float() - 127).unsqueeze(-1)).view(N, K)
FP4 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6])
def nv(w, s, s2):  # NVFP4 [N,K/2] u8 (low nibble = even elem), [N,K/16] e4m3, scalar
    dec = lambda n: torch.where(((n >> 3) & 1) == 1, -FP4[n & 7], FP4[n & 7])
    x = torch.stack([dec((w & 0xF).long()), dec((w >> 4).long())], -1).view(w.shape[0], -1)
    return x * s.float().repeat_interleave(16, dim=1) * s2.float()
# expert: pad gate/up rows 640->768 and down cols (bytes) 320->384, scales with fill; then
#   y = sum_r (silu(x @ G[256r:256r+256].T) * (x @ U[256r:256r+256].T)) @ D[:, 256r:256r+256].T
# attention PAD: q rows 12288->18432, k/v rows 512->768, o cols 6144->9216; rank r takes
#   q heads 12r..12r+11 and kv head r; local group = 12/1.
# GDN PAD: q,k blocks 2048->2304, v block 6144->6912 (pad at the end of each block); rank r takes
#   q/k heads 6r..6r+5, v heads 18r..18r+17; local v head j uses local k head j//3.
```

## Appendix B. Weight-streaming bound script (reproduces Section 3.1)

```python
H = 2560
expert_bytes = lambda I: 1.6875 * I * H      # w13 (I*H) + scales (I*H/8) + w2 (H*I/2) + scales (H*I/16)
def model(tp, B):
    pad = tp == 3; I = 256 if pad else 320
    Ed = 512 * (1 - (1 - 10/512) ** B)
    experts = 48 * Ed * expert_bytes(I)
    gdn   = 2.005 * 2**30 * (1.125 if pad else 1) / tp
    attn  = 0.574 * 2**30 * (1.5 if pad else 1) / tp + 0.019 * 2**30
    shared= 0.227 * 2**30 * (1.2 if pad else 1) / tp
    hc    = 1.193 * 2**30 / (1 if pad else tp)          # replicated at TP=3, sharded at TP=2
    gates = 0.117 * 2**30
    lm    = 248320 * 2560 * 1.0625 / tp * (248448/248320 if pad else 1)   # MXFP8 lm_head
    mtp   = (10 * expert_bytes(I) + 0.169 * 2**30) * 4
    return experts + gdn + attn + shared + hc + gates + lm * 5 + mtp
```

Per-family sizes come from the shard headers (`hdr.json` extracted with a header-only read).
Scratch inputs used in this session live under `~/agent-scratch/` on sparkmain (image file extract
`img/`, `hdr.json`, `ple_geo.py`, `tp3_cpu_check.py`); none of it is part of the repo.
