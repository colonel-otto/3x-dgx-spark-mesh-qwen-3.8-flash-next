# Patch inventory -- Qwen3.8-Flash-Next on 3x DGX Spark

**Taken:** 2026-09-23, from the live nodes (sparkmain, spark1, spark-sep).
**Purpose:** list every local modification we run, so we can update the engine
to upstream and re-decide each patch against it, instead of carrying hand
patches forward blindly. Default policy: **upstream wins**; a local patch
stays only if the updated upstream stack demonstrably still needs it.

## 1. vLLM lane (`eugr/spark-vllm-b12x:latest`, built 2026-08-23, vLLM `0.1.dev20133+gb5f995e73`, b12x 1.2.6)

Mechanism: whole-file bind mounts from `/opt/qwen-patches/` into the container
(`-v /opt/qwen-patches/<f>:/usr/local/lib/python3.12/dist-packages/vllm/...`).
Identical (md5) on all 3 nodes and identical to `patches/*.py` in this repo.
Unified diffs vs the stock files of that image:
`patches/diffs-vs-eugr-b12x-20260823/`.

| # | File | Changed lines vs stock | What it does | Why we needed it | Expected status after upstream update (TO VERIFY) |
|---|---|---|---|---|---|
| V1 | `config/virtual_tp.py` | 256 | Adds `Qwen4ExpConfig` / `Qwen4ExpTextConfig` shims + registration; DFlash virtual-TP plan | Stock image had no Qwen4Exp config; TP=3 padding plan | Config part likely obsolete (upstream eugr `e5174b8` adds native Qwen3.8-next support). **TP=3 padding part is the one piece upstream may not have** -- nobody publishes TP=3 |
| V2 | `model_executor/models/registry.py` | 1 | Maps `Qwen4ExpForConditionalGeneration` -> our `Qwen3NextForCausalLM` | No native model class | Likely obsolete (native model upstream) |
| V3 | `config/speculative.py` | 4 | Lets `qwen4_exp` / `qwen4_exp_text` use the MTP path | Model type not recognised for MTP | Likely obsolete |
| V4 | `model_executor/models/qwen3_next.py` | 285 | Port of Qwen4Exp onto Qwen3-Next: grouped RMSNorm, HyperConnections, mRoPE | No native model | Likely obsolete -- replaced by native implementation |
| V5 | `model_executor/models/qwen3_next_mtp.py` | 191 | MTP drafter port (dual `fc_embedding`/`fc_hidden`, weight remap) | No native MTP | Likely obsolete. **Only ever run with `num_speculative_tokens=1`; crashes at 4** (below) |
| V6 | `model_executor/layers/vocab_parallel_embedding.py` | 6 | Vocab padding = lcm(pad, tp_size) | TP=3 vocab split | **Probably still needed for TP=3** unless upstream handles non-power-of-2 TP |
| V7 | `model_executor/warmup/kernel_warmup.py` | 4 | Autotune process group -> `None` | Multi-node GLOO hang during autotune | Unknown; re-test on new image |

Also present in `/opt/qwen-patches/` but **not mounted** by the Flash-Next
recipe: `core.py`, `qwen3_dflash.py` (used by other model lanes).

### Evidence the V1-V5 port is the limiting factor
2026-09-23: same image + patches with `num_speculative_tokens: 4` crashed in
warmup: `CUDA error: an illegal memory access` surfacing in
`speculator._multi_step_decode -> _build_draft_attn_metadata -> flashinfer build`
(log: `sparkmain:~/vllm-mtp4-20260923.log`). The community stacks that run
MTP=4 (eugr upstream recipe `qwen3.8-flash-next-nvfp4-cluster.yaml`;
ursuciprian `local-inference-lab/vllm@8e1f1e58`) use a native
`qwen3_8_flash_next` model, not a Qwen3-Next port. Faulting kernel not
isolated (async CUDA error).

## 2. SGLang lane (`sglang-flashnext-tp3:local`, live since 2026-09-23)

Built by `docker/build-sglang-flashnext.sh` + `docker/Dockerfile.sglang-flashnext-tp3`
on top of `lmsysorg/sglang@sha256:5ae58167...` (sglang `g593134d17`, 2026-09-03).
Local files identical (md5) to `sparkmain:~/3spark-qwen-flash-build/docker/`.

| # | Patch | What it does | Still needed? |
|---|---|---|---|
| S1 | `patches/sglang_upstream/enable_unaligned_tp_on_cuda.py` | Lifts SGLang's CPU-only unaligned-TP padding onto CUDA behind `SGLANG_FORCE_UNALIGNED_TP=1`; MoE loader gate; vision `original_num_heads` | Yes for TP=3 until upstream supports non-power-of-2 TP on CUDA. Inert when env unset |

SM121 QSA fixes (#36806, #36845) are already in this base; no patch.

## 3. Launcher (`sparkmain:~/eugr-launcher`, clone of `eugr/spark-vllm-docker`)

- At `e9cf359` (2026-08-26). Upstream has moved on (Flash-Next support,
  B12X autotune re-enabled, b12x fixes).
- **17 of our recipes are untracked files in that clone and in no git repo**
  (dsv4-*, qwen3.8-27b-*, qwen3.8-next-flash-nvfp4-tp2/tp3/tp3-mtp4). A
  `git clean` or re-clone would lose them.
- Build scripts on sparkmain (`~/3spark-qwen-flash-build/`) are not a git repo;
  their content matches this repo's `docker/` and `scripts/`.

## 4. Result against the updated engine (2026-09-23)

New image: `eugr/spark-vllm-b12x:latest` pulled 2026-09-23, tagged
`vllm-node-b12x` on all 3 nodes (old one kept as
`eugr/spark-vllm-b12x:local-20260823`). Contents (from its build metadata):
vLLM `local-inference-lab/vllm@57fdda71` (`dev/karmic-kraken`), b12x 1.3.0,
flashinfer 0.7.0, torch 2.13.0+cu130.

`scripts/upstream-check.sh --image vllm-node-b12x` (control: every diff
applies cleanly to the old image):

| # | Path | vs new image |
|---|---|---|
| V7 | `model_executor/warmup/kernel_warmup.py` | applies cleanly |
| V2 | `model_executor/models/registry.py` | conflicts -- and moot: new registry maps `Qwen4ExpFor{CausalLM,ConditionalGeneration}` and `Qwen3_8FlashNext*` to native `vllm.models.qwen4_exp` |
| V3 | `config/speculative.py` | conflicts |
| V4 | `model_executor/models/qwen3_next.py` | conflicts -- moot, model no longer routed through Qwen3-Next |
| V5 | `model_executor/models/qwen3_next_mtp.py` | conflicts -- moot, same reason |
| V6 | `model_executor/layers/vocab_parallel_embedding.py` | conflicts |
| V1 | `config/virtual_tp.py` | file does not exist; no `virtual_tp` / unaligned-TP code anywhere in the new vLLM |

**TP=3 on the native model is not supported upstream.** Checkpoint dims
that do not divide by 3: `num_key_value_heads` 2, `linear_num_key_heads` 16,
`moe_intermediate_size` 640, `vocab_size` 248320 (`num_attention_heads` 24
and `linear_num_value_heads` 48 do). The native `qwen4_exp` code has explicit
divisibility checks (e.g. `amd/qsa.py`: "QSA KV heads must be divisible by TP
size"). A TP=3 route on the new engine therefore needs the V1/V6 padding
*idea* re-implemented against `vllm/models/qwen4_exp/nvidia/`, not the old
files.

## 5. Re-decide procedure after updating the engine

For each patch above, on the new upstream stack:
1. Boot **without** the patch at TP=2 (all dims divide by 2). If it serves and
   passes `scripts/sglang-flashnext-validate.sh` + `scripts/qwen-next-tool-validate.sh`,
   the patch is obsolete for TP=2.
2. Boot at TP=3 without it. Keep only the patches TP=3 still fails without,
   and re-express them as diffs against the new upstream commit.
3. Record the verdict in this table with the upstream commit it was tested on.
