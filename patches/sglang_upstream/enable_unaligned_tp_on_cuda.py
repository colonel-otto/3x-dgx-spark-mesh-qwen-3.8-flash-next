#!/usr/bin/env python3
"""Lift the CPU-only gate on SGLang's unaligned-TP padding for CUDA.

FLASH-NEXT COPY (2026-09-04). Ported verbatim from
3spark-qwen-3.8-27b-dense/patches/sglang_upstream/ and extended with two
call sites the dense model never exercised (patch_fused_moe_layer,
patch_qwen3_vl_vision_heads, at the bottom). Target image:
lmsysorg/sglang:dev-qwen38flashnext (qwen4-main branch, sglang
0.0.0.dev1+g593134d17, transformers 5.12.1 with SGLang's own Qwen4ExpConfig).
All seven original anchors were verified to apply cleanly against that image
in a throwaway container on 2026-09-04; the two additions were written against
its exact source lines. Qwen4ExpForConditionalGeneration subclasses
Qwen3VLForConditionalGeneration and reuses qwen3_5.py's GDN / attention /
ForCausalLM classes, so the qwen3_5.py packed-loader lift is live for
Flash-Next too.

Dry-run of adjust_config_with_unaligned_cpu_tp(tp=3) on the real
RadixArk/Qwen3.8-Flash-Next-NVFP4 config inside that image (no GPU):
  Q heads 24->36, KV 2->3, GDN key 16->18 / value 48->54,
  moe_intermediate 640->672, shared_expert_intermediate 640->672,
  vision heads 16->18 (original_num_heads=16 kept), vision inter 4304->4320.
These match the vLLM virtual_tp plan that serves the model today, axis for
axis, so the padding MATH is already validated on this checkpoint; only the
SGLang loaders are new ground. The QSA indexer (indexer_n_heads=4) is a
ReplicatedLinear and is NOT sharded, so 4 % 3 is not a problem.

--- original dense-repo rationale follows ---

WHY THIS EXISTS: Qwen3.8-27B-FP8 (and any hybrid GDN model loaded via
sglang/srt/models/qwen3_5.py) crashes at TP=3 with
"AssertionError: 248320 is not divisible by 3" -- vocab_size, num_attention
heads, and the GDN linear_num_key_heads/linear_num_value_heads are not
evenly divisible by 3.

Upstream SGLang already has the fix for this -- adjust_config_with_unaligned_
cpu_tp() in srt/configs/update_config.py pads all of these generically
(architecture-agnostic: works for qwen3_5.py, qwen3_next.py, and everything
else) via pad_vocab_size(). It is gated `if self.device == "cpu":` at
model_runner.py, and VocabParallelEmbedding has a second, independent
`_is_cpu` gate in vocab_parallel_embedding.py that only widens padding_size
to a TP-size multiple on the CPU backend.

Investigated 2026-09-03: no PR discussion or code comment documents a CUDA
correctness reason for either gate. The mechanism traces to PR #6771 (CPU
NUMA TP=6) and was reused as-is for XPU (#13972) -- it reads as unexercised
scope, not a deliberate exclusion. This patch lifts BOTH gates behind a new
opt-in env var, SGLANG_FORCE_UNALIGNED_TP=1, so it changes nothing for any
existing CPU/XPU/evenly-divisible-TP deployment -- only a run that explicitly
sets the flag (our TP=3 boot scripts do) takes the new path.

Correctness is NOT assumed just because it boots -- run
scripts/sglang-validate.sh (6/6 battery) before trusting any output from a
server that used this flag. If the battery fails, the CPU-only gate was
protecting something real, and patches/sglang/*.py (hand-written per-model
padding) is the fallback path.

Applied at Docker build time (see docker/Dockerfile.sglang-tp3) against a
matched SGLang checkout inside the image.
"""
import re
import sys
from pathlib import Path

MODEL_RUNNER = Path("/sgl-workspace/sglang/python/sglang/srt/model_executor/model_runner.py")
VOCAB_EMBED = Path("/sgl-workspace/sglang/python/sglang/srt/layers/vocab_parallel_embedding.py")
UPDATE_CONFIG = Path("/sgl-workspace/sglang/python/sglang/srt/configs/update_config.py")
QWEN3_5 = Path("/sgl-workspace/sglang/python/sglang/srt/models/qwen3_5.py")
WEIGHT_UTILS = Path("/sgl-workspace/sglang/python/sglang/srt/model_loader/weight_utils.py")
MAMBA = Path("/sgl-workspace/sglang/python/sglang/srt/layers/attention/mamba/mamba.py")
PARAMETER = Path("/sgl-workspace/sglang/python/sglang/srt/layers/parameter.py")
# Flash-Next additions (2026-09-04):
FUSED_MOE_LAYER = Path("/sgl-workspace/sglang/python/sglang/srt/layers/moe/fused_moe_triton/layer.py")
QWEN3_VL = Path("/sgl-workspace/sglang/python/sglang/srt/models/qwen3_vl.py")
LINEAR = Path("/sgl-workspace/sglang/python/sglang/srt/layers/linear.py")


SENTINEL = "SGLANG_FORCE_UNALIGNED_TP"


def already_patched(path, marker=SENTINEL):
    """True if this file already carries our patch.

    Every patch_*() below fails when its anchor is missing. That is correct for
    a MOVED anchor (upstream changed) but wrong for an ALREADY-PATCHED file --
    the anchor is gone because we consumed it. Without this distinction a
    re-run, or a Docker layer that re-executes this script, reports
    "FATAL: expected block not found" and looks like an upstream break.
    """
    try:
        return marker in path.read_text()
    except FileNotFoundError:
        print(f"FATAL: {path} does not exist. The SGLang layout changed, or "
              f"this is not the expected image.", file=sys.stderr)
        sys.exit(1)


def verify_all_applied():
    """Post-condition: every target file must carry the sentinel.

    A build that silently applies 6 of 7 patches produces an image that boots
    and then crashes on rank 2 mid-load -- which is how mamba.py's dead shim
    was found (the function was written, then clobbered by a later import).
    Checking the end state costs nothing and turns that class of bug into a
    build failure.
    """
    missing = [p for p in (MODEL_RUNNER, VOCAB_EMBED, UPDATE_CONFIG, QWEN3_5,
                           WEIGHT_UTILS, MAMBA, PARAMETER,
                           FUSED_MOE_LAYER, QWEN3_VL, LINEAR)
               if SENTINEL not in p.read_text()]
    if missing:
        print("FATAL: patch verification failed. These files do NOT carry "
              f"{SENTINEL} after patching:", file=sys.stderr)
        for m in missing:
            print(f"  - {m}", file=sys.stderr)
        sys.exit(1)
    print(f"verified: all 10 targets carry {SENTINEL}")


def verify_shims_live():
    """mamba.py and weight_utils.py shadow is_cpu() by defining a function AFTER
    the import that binds it. If a later import re-binds the name, the shim
    becomes dead code that still LOOKS correct in the file -- this actually
    happened in mamba.py (fixed by moving the anchor after the import block).
    Assert the shadow is the last binding of `is_cpu` in each file.
    """
    for path in (WEIGHT_UTILS, MAMBA):
        text = path.read_text()
        shim_at = text.find("def is_cpu():  # noqa: F811")
        if shim_at < 0:
            print(f"FATAL: is_cpu shim missing from {path}", file=sys.stderr)
            sys.exit(1)
        tail = text[shim_at:]
        # Any later import that re-binds the bare name kills the shadow.
        for pat in (r"^\s*from\s.*\bimport\b.*\bis_cpu\b", r"^\s*import\s.*\bis_cpu\b"):
            m = re.search(pat, tail, re.M)
            if m:
                line = text[:shim_at].count(chr(10)) + tail[:m.start()].count(chr(10)) + 1
                print(f"FATAL: {path} re-imports is_cpu at line ~{line}, AFTER the "
                      f"shim. The shadow is dead code -- move the anchor below "
                      f"that import.", file=sys.stderr)
                sys.exit(1)
    print("verified: is_cpu shims are the final binding in both files")


def patch_model_runner():
    src = MODEL_RUNNER.read_text()
    old = (
        '        if self.device == "cpu":\n'
        "            self.model_config = adjust_config_with_unaligned_cpu_tp(\n"
        "                self.model_config, self.load_config, self.ps.tp_size\n"
        "            )\n"
    )
    if old not in src:
        print(f"FATAL: expected block not found in {MODEL_RUNNER}", file=sys.stderr)
        sys.exit(1)
    new = (
        '        if self.device == "cpu" or (\n'
        '            os.environ.get("SGLANG_FORCE_UNALIGNED_TP") == "1"\n'
        "            and (\n"
        "                self.model_config.num_attention_heads % self.ps.tp_size != 0\n"
        "                or self.model_config.get_total_num_kv_heads() % self.ps.tp_size != 0\n"
        "            )\n"
        "        ):\n"
        "            self.model_config = adjust_config_with_unaligned_cpu_tp(\n"
        "                self.model_config, self.load_config, self.ps.tp_size\n"
        "            )\n"
    )
    src = src.replace(old, new, 1)
    if not re.search(r"^import os$", src, re.M):
        src = src.replace("import contextlib\n", "import contextlib\nimport os\n", 1)
    if not re.search(r"^import os$", src, re.M):
        print(f"FATAL: could not insert 'import os' into {MODEL_RUNNER}", file=sys.stderr)
        sys.exit(1)
    MODEL_RUNNER.write_text(src)
    print(f"patched {MODEL_RUNNER}")


def patch_vocab_embedding():
    src = VOCAB_EMBED.read_text()
    old = (
        "        if (\n"
        "            _is_cpu\n"
        "            and pad_vocab_size(self.org_vocab_size, padding_size) % self.tp_size != 0\n"
        "        ):\n"
        "            padding_size *= self.tp_size\n"
    )
    if old not in src:
        print(f"FATAL: expected block not found in {VOCAB_EMBED}", file=sys.stderr)
        sys.exit(1)
    new = (
        "        if (\n"
        "            (_is_cpu or os.environ.get(\"SGLANG_FORCE_UNALIGNED_TP\") == \"1\")\n"
        "            and pad_vocab_size(self.org_vocab_size, padding_size) % self.tp_size != 0\n"
        "        ):\n"
        "            padding_size *= self.tp_size\n"
    )
    src = src.replace(old, new, 1)
    if not re.search(r"^import os$", src, re.M):
        src = src.replace("import logging\n", "import logging\nimport os\n", 1)
    if not re.search(r"^import os$", src, re.M):
        print(f"FATAL: could not insert 'import os' into {VOCAB_EMBED}", file=sys.stderr)
        sys.exit(1)
    VOCAB_EMBED.write_text(src)
    print(f"patched {VOCAB_EMBED}")


def patch_update_config():
    """adjust_tp_num_heads_if_necessary(..., is_post_update=True) -- the
    branch both model_runner.py call sites use -- writes the padded GDN
    linear-attn head counts to linear_num_key_heads_cpu / _value_heads_cpu,
    NOT to the plain linear_num_key_heads / linear_num_value_heads.
    qwen3_5.py's Qwen3_5GatedDeltaNet reads the _cpu-suffixed attrs only
    `if _is_cpu`; on CUDA it reads the plain (unpadded) attrs regardless of
    what adjust_config_with_unaligned_cpu_tp computed, so the vocab crash
    goes away but conv1d's output_size (key_dim*2 + value_dim, derived from
    the unpadded head counts) still isn't divisible by tp_size. Observed
    2026-09-03: "AssertionError: 10240 is not divisible by 3" after the
    vocab-only fix. This makes is_post_update also write the plain attrs
    when SGLANG_FORCE_UNALIGNED_TP=1, so any model file that reads
    linear_num_key_heads/linear_num_value_heads directly (not through the
    _is_cpu split) sees the padded value on our CUDA opt-in path too. The
    _cpu writes are left in place -- untouched for real CPU runs.
    """
    src = UPDATE_CONFIG.read_text()
    old = (
        "            if is_post_update:\n"
        "                update_config(\n"
        '                    model_config, "linear_num_key_heads_cpu", linear_num_key_heads_cpu\n'
        "                )\n"
        "                update_config(\n"
        "                    model_config,\n"
        '                    "linear_num_value_heads_cpu",\n'
        "                    linear_num_value_heads_cpu,\n"
        "                )\n"
    )
    if old not in src:
        print(f"FATAL: expected block not found in {UPDATE_CONFIG}", file=sys.stderr)
        sys.exit(1)
    new = (
        "            if is_post_update:\n"
        "                update_config(\n"
        '                    model_config, "linear_num_key_heads_cpu", linear_num_key_heads_cpu\n'
        "                )\n"
        "                update_config(\n"
        "                    model_config,\n"
        '                    "linear_num_value_heads_cpu",\n'
        "                    linear_num_value_heads_cpu,\n"
        "                )\n"
        '                if os.environ.get("SGLANG_FORCE_UNALIGNED_TP") == "1":\n'
        "                    update_config(\n"
        '                        model_config, "linear_num_key_heads", linear_num_key_heads_cpu\n'
        "                    )\n"
        "                    update_config(\n"
        "                        model_config,\n"
        '                        "linear_num_value_heads",\n'
        "                        linear_num_value_heads_cpu,\n"
        "                    )\n"
    )
    src = src.replace(old, new, 1)
    # Second edit in this file (Flash-Next, found on the THIRD live TP=3 boot,
    # 2026-09-04 13:11, rank 2): the flashinfer_cutlass NVFP4 MoE path
    # swizzles w13_weight_scale in 128-row tiles, so the per-rank
    # 2 x intermediate must be a multiple of 128, i.e. intermediate_per_rank
    # % 64 == 0. Stock alignment is tp x NVFP4 group = 48 -> 640 pads to 672
    # -> 224/rank -> 448 rows -> swizzle pads to 512 and
    # modelopt_quant.py:2751 asserts "The intermediate size required padding,
    # but padding is also implemented for gated activations" (it refuses to
    # split-pad w1/w3). TP=2 never sees this: 320/rank -> 640 rows = 5 x 128.
    # The trtllm path rounds the partition up to 128 itself; cutlass does not.
    # Fix at the CONFIG level so every loader zero-fills consistently: under
    # the flag, raise the MoE/shared/dense intermediate alignment to
    # tp x SGLANG_UNALIGNED_TP_MOE_ALIGN (default 64): 640 -> 768 -> 256/rank
    # (rank 2 holds 128 real + 128 zero rows; zero gate/up rows give
    # silu(0)*0 = 0 and zero down_proj rows contribute nothing).
    old_align = "    intermediate_padding_size = tp_size * get_moe_padding_size(weight_block_size)\n"
    if src.count(old_align) != 1:
        print(f"FATAL: expected exactly one intermediate_padding_size line in {UPDATE_CONFIG}, "
              f"found {src.count(old_align)}", file=sys.stderr)
        sys.exit(1)
    new_align = (
        old_align
        + '    if os.environ.get("SGLANG_FORCE_UNALIGNED_TP") == "1":\n'
        + "        # enable_unaligned_tp_on_cuda.py: flashinfer_cutlass NVFP4 MoE needs\n"
        + "        # (2 * intermediate_per_rank) % 128 == 0 (blockscale swizzle tiles).\n"
        + '        _align = int(os.environ.get("SGLANG_UNALIGNED_TP_MOE_ALIGN", "64"))\n'
        + "        intermediate_padding_size = max(intermediate_padding_size, tp_size * _align)\n"
    )
    src = src.replace(old_align, new_align, 1)
    if not re.search(r"^import os$", src, re.M):
        src = src.replace("import logging\n", "import logging\nimport os\n", 1)
    if not re.search(r"^import os$", src, re.M):
        print(f"FATAL: could not insert 'import os' into {UPDATE_CONFIG}", file=sys.stderr)
        sys.exit(1)
    UPDATE_CONFIG.write_text(src)
    print(f"patched {UPDATE_CONFIG}")


def patch_qwen3_5_weight_loader():
    """qwen3_5.py's packed in_proj_qkvz/in_proj_ba weight_loader splits a
    LOADED (checkpoint) tensor using the padded split_sizes directly on
    CUDA -- but the checkpoint tensor itself is never padded, so
    sum(split_sizes) (e.g. 2304+2304+6912=11520 after GDN head padding)
    does not match the loaded tensor's actual size (10240). Crashes with
    "RuntimeError: split_with_sizes expects split_sizes to sum exactly to
    10240 ... but got split_sizes=[2304, 2304, 6912]". Observed 2026-09-03,
    the crash immediately after the earlier config-level fixes got the
    model to actually start loading weights.

    The existing `if _is_cpu:` branch already does the right thing: it
    rescales split_sizes PROPORTIONALLY to the loaded tensor's real size
    (cpu_split_sizes sums to target_size_sim by construction) rather than
    assuming the tensor was pre-padded. This is not a padding computation
    upstream forgot on CUDA -- it is a real, different code path, which is
    why the CPU-only gate is more load-bearing than the two config-level
    gates above. Reuse that exact logic under our opt-in flag instead of
    writing new splitting math.
    """
    src = QWEN3_5.read_text()
    old = (
        "                    split_dim = getattr(param, \"output_dim\", 0)\n"
        "                    if _is_cpu:\n"
    )
    if old not in src:
        print(f"FATAL: expected block not found in {QWEN3_5}", file=sys.stderr)
        sys.exit(1)
    new = (
        "                    split_dim = getattr(param, \"output_dim\", 0)\n"
        '                    if _is_cpu or os.environ.get("SGLANG_FORCE_UNALIGNED_TP") == "1":\n'
    )
    src = src.replace(old, new, 1)
    if not re.search(r"^import os$", src, re.M):
        src = src.replace("import logging\n", "import logging\nimport os\n", 1)
    if not re.search(r"^import os$", src, re.M):
        print(f"FATAL: could not insert 'import os' into {QWEN3_5}", file=sys.stderr)
        sys.exit(1)
    QWEN3_5.write_text(src)
    print(f"patched {QWEN3_5}")


def patch_weight_utils():
    """weight_loader/weight_utils.py has multiple loader factories
    (sharded_weight_loader and others) that each check `is_cpu()` to decide
    whether to zero-pad a checkpoint tensor that is smaller than the padded
    (virtual) shard -- via narrow_padded_param_and_loaded_weight, which is
    exactly the zero-tail padding our hand-written patches/sglang/linear.py
    tried to reinvent. Observed 2026-09-03: rank 2 crashed with
    "RuntimeError: start (36) + length (18) exceeds dimension size (48)"
    in sharded_weight_loader -- a DIFFERENT loader than the packed-qkvz one
    already patched in qwen3_5.py, confirming this "loaded tensor smaller
    than padded shard" problem recurs at every call site that assumes the
    checkpoint was pre-padded.

    Rather than chase each call site individually, rebind `is_cpu` inside
    this module's own namespace so every is_cpu() check in weight_utils.py
    -- present and future -- takes the already-correct CPU zero-padding
    branch when SGLANG_FORCE_UNALIGNED_TP=1, without touching the real
    is_cpu() used elsewhere in the codebase for CPU-only kernel dispatch
    (fused_sigmoid_mul_cpu etc. in qwen3_5.py and others), which would
    break if actually run on CUDA. This is intentionally scoped to
    weight-loading code only.
    """
    src = WEIGHT_UTILS.read_text()
    anchor = "from sglang.srt.utils.common import is_cuda_alike\n"
    if anchor not in src:
        print(f"FATAL: expected anchor not found in {WEIGHT_UTILS}", file=sys.stderr)
        sys.exit(1)
    shim = (
        anchor
        + "_orig_is_cpu = is_cpu\n"
        + "def is_cpu():  # noqa: F811 -- intentional shadow, see enable_unaligned_tp_on_cuda.py\n"
        + '    return _orig_is_cpu() or os.environ.get("SGLANG_FORCE_UNALIGNED_TP") == "1"\n'
    )
    src = src.replace(anchor, shim, 1)
    WEIGHT_UTILS.write_text(src)
    print(f"patched {WEIGHT_UTILS}")


def patch_mamba():
    """mamba_v2_sharded_weight_loader in layers/attention/mamba/mamba.py has
    its own is_cpu()-gated zero-padding logic (comment: "CPU logic of
    padding size for qwen3-next / TODO: make this common for all mamba") --
    a THIRD independent implementation of the same "loaded checkpoint tensor
    smaller than the padded shard" fix, this time hand-rolled rather than
    calling narrow_padded_param_and_loaded_weight. Observed 2026-09-03: rank
    2 crashed here after the qwen3_5.py and weight_utils.py loaders were
    already fixed -- "RuntimeError: The expanded size of the tensor (2304)
    must match the existing size (1024)".

    Found via scripts/audit_unaligned_tp_gates.py, which statically enumerates
    every is_cpu() weight-padding gate reachable from this model's load path
    instead of discovering them one crash/rebuild cycle at a time. Re-run
    that script after any SGLang upgrade or model change -- it exits 1 and
    lists exact file:line gaps if a new one appears.

    Same shim approach as patch_weight_utils(): is_cpu has only two uses in
    this file (both weight-padding, confirmed by the audit script), so
    shadowing it module-wide is safe here too.
    """
    src = MAMBA.read_text()
    # Anchor AFTER the `from sglang.srt.utils import (is_cpu, ...)` block --
    # get_parallel is imported BEFORE is_cpu in this file, so shimming there
    # (as first attempted) had `is_cpu` re-imported and silently clobber the
    # shadow immediately after. Verified 2026-09-03 via
    # scripts/audit_unaligned_tp_gates.py still flagging this file as a GAP
    # after a real rebuild -- the shim function existed in the file but was
    # dead code, overwritten by the later import.
    anchor = "    set_weight_attrs,\n)\n"
    if anchor not in src:
        print(f"FATAL: expected anchor not found in {MAMBA}", file=sys.stderr)
        sys.exit(1)
    if not re.search(r"^import os$", src, re.M):
        src = src.replace("import logging\n", "import logging\nimport os\n", 1)
    if not re.search(r"^import os$", src, re.M):
        print(f"FATAL: could not insert 'import os' into {MAMBA}", file=sys.stderr)
        sys.exit(1)
    shim = (
        anchor
        + "_orig_is_cpu = is_cpu\n"
        + "def is_cpu():  # noqa: F811 -- intentional shadow, see enable_unaligned_tp_on_cuda.py\n"
        + '    return _orig_is_cpu() or os.environ.get("SGLANG_FORCE_UNALIGNED_TP") == "1"\n'
    )
    src = src.replace(anchor, shim, 1)
    MAMBA.write_text(src)
    print(f"patched {MAMBA}")


def patch_parameter():
    """layers/parameter.py (BasevLLMParameter and subclasses -- the actual
    weight_loader_v2 path qwen3_5.py's QKV projections use) has FOUR
    `_is_cpu` gates, all using the same narrow_padded_param_and_loaded_weight
    helper as weight_utils.py's sharded_weight_loader, but via a cached
    module-level `_is_cpu = is_cpu()` boolean rather than a live function
    call -- so the function-shadow trick used for weight_utils.py/mamba.py
    does not apply here (rebinding is_cpu() after the fact would not change
    an already-evaluated boolean). Observed 2026-09-03: rank 2 crashed in
    load_qkv_weight (line ~267) with "RuntimeError: start (1024) + length
    (512) exceeds dimension size (1024)" -- a fourth independent call site
    for the same class of bug, found only because a real boot got far enough
    to reach it; scripts/audit_unaligned_tp_gates.py had not been told to
    scan this file (IN_SCOPE_PREFIXES was built from files already seen
    crashing, not a real import-graph trace -- fixed alongside this patch).

    Same technique as patch_vocab_embedding(): edit the `_is_cpu = is_cpu()`
    assignment itself so the cached value is True under our flag too, rather
    than trying to shadow a function that's called exactly once at import
    time.
    """
    src = PARAMETER.read_text()
    old = "_is_cpu = is_cpu()\n"
    if old not in src:
        print(f"FATAL: expected line not found in {PARAMETER}", file=sys.stderr)
        sys.exit(1)
    new = '_is_cpu = is_cpu() or os.environ.get("SGLANG_FORCE_UNALIGNED_TP") == "1"\n'
    src = src.replace(old, new, 1)
    if not re.search(r"^import os$", src, re.M):
        src = src.replace("import logging\n", "import logging\nimport os\n", 1)
    if not re.search(r"^import os$", src, re.M):
        print(f"FATAL: could not insert 'import os' into {PARAMETER}", file=sys.stderr)
        sys.exit(1)
    PARAMETER.write_text(src)
    print(f"patched {PARAMETER}")


def patch_fused_moe_layer():
    """layers/moe/fused_moe_triton/layer.py decides whether an expert weight
    loader may receive a checkpoint slice SMALLER than the padded per-rank
    shard via `use_padded_loading`, which returns
    `_is_cpu or self.use_flashinfer_trtllm_moe or aiter_padded`. On the
    Flash-Next day-0 image the NVFP4 MoE runner is flashinfer_cutlass (the
    only GB10-viable backend; trtllm-gen has no sm_121 kernels), so on CUDA
    that property is False and every _load_w13 / _load_w2 call takes the
    strict narrow() path.

    At TP=3 moe_intermediate_size is padded 640->672 (224/rank); rank 2's
    slice of the real 640-wide expert tensor is only 192 wide, so the strict
    path raises the familiar "start + length exceeds dimension size" -- for
    all 294,914 expert tensors. Same fix as patch_parameter(): `_is_cpu` here
    is a cached module-level bool, so rewrite the assignment rather than
    shadow the function. The `if _is_cpu and is_bias` branches inside the
    loaders also flip under the flag, but this checkpoint's experts carry no
    biases, so they are unreachable. Anchored 2026-09-04 against line 93 of
    the dev-qwen38flashnext image. NOT found by the static audit: its regex
    only sees `is_cpu(` call sites, and this consumer sits 500 lines below
    the cached assignment -- found by reading the loader.

    Correctness of the padded NVFP4 shard (uint8-packed weights, 16-wide
    block scales: 640 = 224 + 224 + 192, all multiples of 16, w2 packed input
    dim 224/2 = 112) is arithmetically clean but UNTESTED on a live boot.
    """
    src = FUSED_MOE_LAYER.read_text()
    old = "_is_cpu = is_cpu()\n"
    if src.count(old) != 1:
        print(f"FATAL: expected exactly one '_is_cpu = is_cpu()' in {FUSED_MOE_LAYER}, "
              f"found {src.count(old)}", file=sys.stderr)
        sys.exit(1)
    new = ('_is_cpu = is_cpu() or os.environ.get("SGLANG_FORCE_UNALIGNED_TP") == "1"'
           "  # enable_unaligned_tp_on_cuda.py: use_padded_loading on CUDA\n")
    src = src.replace(old, new, 1)
    if not re.search(r"^import os$", src, re.M):
        src = src.replace("import logging\n", "import logging\nimport os\n", 1)
    if not re.search(r"^import os$", src, re.M):
        print(f"FATAL: could not insert 'import os' into {FUSED_MOE_LAYER}", file=sys.stderr)
        sys.exit(1)
    FUSED_MOE_LAYER.write_text(src)
    print(f"patched {FUSED_MOE_LAYER}")


def patch_qwen3_vl_vision_heads():
    """The vision-tower head-count corruption from the dense repo, verbatim.
    adjust_config_with_unaligned_cpu_tp pads vision_config.num_heads 16->18
    for Flash-Next at TP=3 (dry-run verified 2026-09-04) with no awareness
    that --mm-enable-dp-encoder keeps the tower replicated at TP=1.
    Qwen3VLVisionModel.__init__ reads the padded value into self.num_heads
    and derives the rotary head_dim from it (1152//18 = 64 instead of
    1152//16 = 72), which crashes deterministically on the first forward
    pass -- including SGLang's own warmup request -- while /v1/models still
    answers 200. Upstream only reads original_num_heads under `_is_cpu`.

    The dense repo fixed this by COPYING a whole qwen3_vl.py from v0.5.18;
    that file has drifted on the qwen4-main branch, so this copy makes two
    targeted edits anchored on the exact upstream lines instead.
    Qwen4ExpForConditionalGeneration subclasses Qwen3VLForConditionalGeneration
    and builds this exact class, so the fix is live for Flash-Next. The
    inserted comment carries the SENTINEL string on purpose so
    already_patched()/verify_all_applied() see this file like the others.
    """
    src = QWEN3_VL.read_text()
    old1 = "        self.num_heads = vision_config.num_heads\n"
    if src.count(old1) != 1:
        print(f"FATAL: expected exactly one num_heads assignment in {QWEN3_VL}, "
              f"found {src.count(old1)}", file=sys.stderr)
        sys.exit(1)
    new1 = (
        "        # enable_unaligned_tp_on_cuda.py (SGLANG_FORCE_UNALIGNED_TP): read the\n"
        "        # unpadded original_num_heads preserved by adjust_config_with_unaligned_cpu_tp;\n"
        "        # the padded value is wrong for a TP=1 (dp-encoder) tower on ANY backend.\n"
        '        self.num_heads = getattr(vision_config, "original_num_heads", vision_config.num_heads)\n'
    )
    src = src.replace(old1, new1, 1)
    old2 = (
        '        if _is_cpu and hasattr(vision_config, "original_num_heads"):\n'
        "            head_dim = self.hidden_size // vision_config.original_num_heads\n"
        "        else:\n"
        "            head_dim = self.hidden_size // self.num_heads\n"
    )
    if old2 not in src:
        print(f"FATAL: expected head_dim block not found in {QWEN3_VL}", file=sys.stderr)
        sys.exit(1)
    new2 = (
        "        # enable_unaligned_tp_on_cuda.py: self.num_heads is already the unpadded\n"
        "        # original_num_heads (see above); no is_cpu split needed here.\n"
        "        head_dim = self.hidden_size // self.num_heads\n"
    )
    src = src.replace(old2, new2, 1)
    # Third edit, found on the FIRST live TP=3 boot (2026-09-04 12:54, ranks 1
    # and 2): adjust_config_with_unaligned_cpu_tp also pads
    # vision_config.intermediate_size 4304 -> 4320 (tp x NVFP4 group = 48
    # alignment) and stores the original in original_intermediate_size. The
    # vision MLP is built from the padded value, but with --mm-enable-dp-encoder
    # the tower is TP=1, so ColumnParallelLinear.weight_loader (tp_size=1,
    # shard_size = full 4320) narrows the real 4304-wide checkpoint tensor and
    # dies: "start (0) + length (4320) exceeds dimension size (4304)"
    # (linear.py:452 via qwen4_exp.py load_weights). Same disease as the
    # head count: a TP=1 tower must be built from ORIGINAL sizes.
    old3 = "                    intermediate_dim=vision_config.intermediate_size,\n"
    if src.count(old3) != 1:
        print(f"FATAL: expected exactly one vision intermediate_dim kwarg in {QWEN3_VL}, "
              f"found {src.count(old3)}", file=sys.stderr)
        sys.exit(1)
    new3 = (
        "                    # enable_unaligned_tp_on_cuda.py: TP=1 (dp-encoder) tower must use\n"
        "                    # the UNPADDED intermediate size or its weight loader narrows past\n"
        "                    # the end of the checkpoint tensor (4304 vs padded 4320).\n"
        '                    intermediate_dim=getattr(vision_config, "original_intermediate_size", vision_config.intermediate_size),\n'
    )
    src = src.replace(old3, new3, 1)
    QWEN3_VL.write_text(src)
    print(f"patched {QWEN3_VL}")


def patch_linear_layer():
    """layers/linear.py -- the v1 `weight_loader` path of ColumnParallelLinear,
    MergedColumnParallelLinear, QKVParallelLinear and RowParallelLinear.

    Found on the SECOND live TP=3 boot (2026-09-04 13:01, rank 2):
    "RuntimeError: start (512) + length (256) exceeds dimension size (512)"
    at linear.py:1394 (QKVParallelLinear.weight_loader) via qwen4_exp.py
    load_weights. 512 = 2 KV heads x head_dim 256; padded to 3 heads, rank 2
    asks for rows 512..768 of a 512-row k_proj / v_proj tensor.

    Why the dense model never hit this: its FP8 checkpoint quantizes q/k/v,
    so the QKV projection carried BasevLLMParameter weights and loaded through
    parameter.py's weight_loader_v2 (patched as patch_parameter). On the
    NVFP4 Flash-Next checkpoint qwen3_5.py explicitly builds qkv_proj with
    quant_config=None ("qkv_proj is not quantized for fp4"), so the plain
    torch Parameter takes the v1 loader in linear.py instead. The static
    audit had linear.py classified as "CUDA branch already pads via
    pad_or_narrow_weight" -- true for weight_loader_v2, FALSE for these v1
    loaders, which take a strict narrow() unless `_is_cpu`.

    All eight `_is_cpu` uses in this file (lines ~436, 651, 717, 793, 839,
    1272, 1376, 1548 on the day-0 image) are weight-loading padding branches
    (narrow_padded_param_and_loaded_weight, adjust_shard_offsets,
    pad_loaded_weight); none is kernel dispatch. Reviewed 2026-09-04. Same
    cached-bool rewrite as patch_parameter().
    """
    src = LINEAR.read_text()
    old = "_is_cpu = is_cpu()\n"
    if src.count(old) != 1:
        print(f"FATAL: expected exactly one '_is_cpu = is_cpu()' in {LINEAR}, "
              f"found {src.count(old)}", file=sys.stderr)
        sys.exit(1)
    new = ('_is_cpu = is_cpu() or os.environ.get("SGLANG_FORCE_UNALIGNED_TP") == "1"'
           "  # enable_unaligned_tp_on_cuda.py: v1 weight_loader padding on CUDA\n")
    src = src.replace(old, new, 1)
    if not re.search(r"^import os$", src, re.M):
        src = src.replace("import logging\n", "import logging\nimport os\n", 1)
    if not re.search(r"^import os$", src, re.M):
        print(f"FATAL: could not insert 'import os' into {LINEAR}", file=sys.stderr)
        sys.exit(1)
    LINEAR.write_text(src)
    print(f"patched {LINEAR}")


PATCHES = (
    ("model_runner", MODEL_RUNNER, patch_model_runner),
    ("vocab_embedding", VOCAB_EMBED, patch_vocab_embedding),
    ("update_config", UPDATE_CONFIG, patch_update_config),
    ("qwen3_5_weight_loader", QWEN3_5, patch_qwen3_5_weight_loader),
    ("weight_utils", WEIGHT_UTILS, patch_weight_utils),
    ("mamba", MAMBA, patch_mamba),
    ("parameter", PARAMETER, patch_parameter),
    # Flash-Next additions (2026-09-04):
    ("fused_moe_layer", FUSED_MOE_LAYER, patch_fused_moe_layer),
    ("qwen3_vl_vision_heads", QWEN3_VL, patch_qwen3_vl_vision_heads),
    ("linear_layer", LINEAR, patch_linear_layer),
)

if __name__ == "__main__":
    for name, path, fn in PATCHES:
        if already_patched(path):
            print(f"skip {name}: {path} already carries {SENTINEL}")
            continue
        fn()
    verify_all_applied()
    verify_shims_live()
    print("enable_unaligned_tp_on_cuda: OK")
