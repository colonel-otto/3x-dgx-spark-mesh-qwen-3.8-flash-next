#!/usr/bin/env python3
"""Static audit: find every is_cpu()-gated weight-padding branch reachable
from Qwen3.8-Flash-Next's ACTUAL import graph (models/qwen4_exp.py and
models/qwen4_exp_mtp.py on the qwen4-main day-0 image), and check whether our
unaligned-TP patch (patches/sglang_upstream/enable_unaligned_tp_on_cuda.py)
already covers it.

FLASH-NEXT COPY (2026-09-04), ported from 3spark-qwen-3.8-27b-dense. First
run against lmsysorg/sglang:dev-qwen38flashnext found 607 modules in the
real import graph and 9 weight-padding call sites: 7 already covered, 2 new
(dp_attention.py memcpy dispatch, forward_batch_info.py pin_memory) that are
runtime dispatch, not padding -- reviewed and listed below. NOTE the audit
regex only sees `is_cpu(` CALL SITES; a cached module-level `_is_cpu` bool
consumed 500 lines later (fused_moe_triton/layer.py::use_padded_loading) is
invisible to it and was found by reading the loader. Keep COVERED_FILES in
sync with the patch anyway so the OK/GAP table stays honest.

WHY THIS EXISTS: getting SGLang TP=3 to actually boot has been a sequence of
"patch one is_cpu() gate, rebuild, boot, crash on the next one" cycles --
qwen3_5.py's packed weight_loader, weight_utils.py's sharded_weight_loader,
mamba.py's mamba_v2_sharded_weight_loader, and layers/parameter.py's
BasevLLMParameter subclasses all hit the same class of bug (checkpoint
tensor smaller than the padded/virtual shard) at different call sites. Each
cycle cost a full image rebuild + 3-node boot attempt (minutes) to discover
ONE more gap. This script finds them all statically in seconds.

SCOPE, v2 (2026-09-03): the first version hand-maintained a list of
"in scope" file prefixes built from files already seen crashing -- which is
exactly backwards, and is why it missed layers/parameter.py until a live
boot crashed there. This version imports the actual model classes
(sglang.srt.models.qwen3_5, sglang.srt.models.qwen3_vl) inside the running
container and diffs sys.modules before/after to get the REAL import graph --
every sglang.srt module actually loaded for this checkpoint, not a guess.
Must run inside the built image (needs a real Python env with sglang
importable); pass --sglang-root only to override where those files live on
disk once the module names are known.

Usage (inside the built image):
    python3 scripts/audit_unaligned_tp_gates.py

Exit 0: every is_cpu() weight-padding gate in scope is covered by our patch
        (either directly shadowed/rewritten, or hand-reviewed as
        kernel-dispatch / already-CUDA-safe -- see the REVIEWED_* sets).
Exit 1: found an is_cpu() gate that looks weight-loading-related and is NOT
        yet covered -- prints the file:line so it can be added to
        enable_unaligned_tp_on_cuda.py BEFORE the next rebuild.
"""
import argparse
import importlib
import re
import sys
from pathlib import Path

# Files our patch script currently shadows/rewrites is_cpu()-gated logic in.
# Keep in sync with patches/sglang_upstream/enable_unaligned_tp_on_cuda.py's
# patch_*() targets.
COVERED_FILES = {
    "srt/model_executor/model_runner.py",       # patch_model_runner (device== gate, not is_cpu())
    "srt/layers/vocab_parallel_embedding.py",   # patch_vocab_embedding
    "srt/configs/update_config.py",             # patch_update_config
    "srt/models/qwen3_5.py",                    # patch_qwen3_5_weight_loader (ONE specific block)
    "srt/model_loader/weight_utils.py",         # patch_weight_utils (module-wide is_cpu shadow)
    "srt/layers/attention/mamba/mamba.py",      # patch_mamba (module-wide is_cpu shadow)
    "srt/layers/parameter.py",                  # patch_parameter (_is_cpu cached-bool rewrite)
    "srt/layers/moe/fused_moe_triton/layer.py", # patch_fused_moe_layer (_is_cpu cached-bool rewrite; use_padded_loading)
    "srt/models/qwen3_vl.py",                   # patch_qwen3_vl_vision_heads (original_num_heads + intermediate, 3 targeted edits)
    "srt/layers/linear.py",                     # patch_linear_layer (_is_cpu cached-bool rewrite; v1 weight_loader narrow paths)
}

# Modules whose is_cpu() usage is confirmed CPU-kernel-dispatch (module-level
# import selection, runtime shape assertions unrelated to checkpoint
# padding), not weight padding -- reviewed by hand 2026-09-03, safe to skip.
REVIEWED_KERNEL_DISPATCH_ONLY = {
    "srt/layers/attention/linear/gdn_backend.py",  # is_cpu() selects causal_conv1d_fn_cpu vs
    # _cuda at import time, and gates a conv-state-shape assert vs FLA_CHUNK_SIZE that has
    # nothing to do with checkpoint tensor size -- not a padding decision.
    "srt/layers/conv.py",  # is_cpu() + cpu_has_amx_support() selects conv3d_embed_cpu (Intel
    # AMX kernel) for vision patch embedding at import time -- kernel dispatch, no weight padding.
    "srt/layers/dp_attention.py",  # `_is_cpu` selects memcpy_cpu vs memcpy_triton for DP-attention
    # token gathers (line ~493) -- runtime kernel dispatch, no checkpoint tensor is touched.
    # Reviewed 2026-09-04 on dev-qwen38flashnext.
    "srt/model_executor/forward_batch_info.py",  # `_is_cpu` only gates use_pin_memory and a
    # CPU-side tensor placement for the forward batch -- runtime, not weight loading.
    # Reviewed 2026-09-04 on dev-qwen38flashnext.
    "srt/utils/common.py",  # device_context()'s `device.type == "cpu" and is_cpu()` only takes
    # the cpu-device branch when the ACTUAL device passed in is CPU -- irrelevant to a CUDA run
    # regardless of what is_cpu() returns, and this is the file is_cpu() itself lives in (not
    # something we'd shim anyway).
}

# Files whose CUDA (non-_is_cpu) branch already does correct zero-tail
# padding independently of is_cpu() -- via pad_or_narrow_weight in
# srt/layers/utils/common.py, which is NOT is_cpu()-gated at all. Reviewed
# 2026-09-03.
# 2026-09-04: linear.py was listed here as "CUDA branch already pads" -- that
# was TRUE only for weight_loader_v2 (BasevLLMParameter). The v1 weight_loader
# paths (plain torch Parameters, e.g. Flash-Next's unquantized qkv_proj under
# NVFP4) take a strict narrow() unless _is_cpu, and rank 2 died there on the
# second live TP=3 boot. linear.py is now a COVERED file (patch_linear_layer).
REVIEWED_ALREADY_PADS_ON_CUDA = set()

WEIGHT_PADDING_HINTS = re.compile(
    r"pad|shard|split|narrow|weight_loader|loader\b|full_dim|split_sizes|shard_size",
    re.IGNORECASE,
)

# The model classes actually loaded for this checkpoint
# (Qwen4ExpForConditionalGeneration + its MTP draft class, TP=3,
# --mm-enable-dp-encoder). qwen4_exp.py itself imports qwen3_5.py and
# qwen3_vl.py, so those land in the graph transitively. Update if the target
# model or server flags change.
ENTRYPOINT_MODULES = [
    "sglang.srt.models.qwen4_exp",
    "sglang.srt.models.qwen4_exp_mtp",
]


def discover_real_import_graph() -> list[str]:
    """Import the entrypoint modules and diff sys.modules to get every
    sglang.srt module actually loaded -- the real scope, not a guess."""
    before = set(sys.modules.keys())
    for m in ENTRYPOINT_MODULES:
        importlib.import_module(m)
    after = set(sys.modules.keys())
    return sorted(m for m in (after - before) if m.startswith("sglang.srt"))


def module_to_relpath(modname: str) -> str:
    # sglang.srt.models.qwen3_5 -> srt/models/qwen3_5.py
    parts = modname.split(".")[1:]  # drop leading "sglang"
    return "/".join(parts) + ".py"


def find_is_cpu_call_sites(root: Path, rel_path: str) -> list[tuple[int, str]]:
    fpath = root / rel_path
    if not fpath.is_file():
        return []
    text = fpath.read_text(errors="replace")
    hits = []
    for i, line in enumerate(text.splitlines(), start=1):
        code_part = line.split("#", 1)[0]  # strip trailing comments; a line
        # that's ENTIRELY a comment (or docstring prose mentioning is_cpu())
        # has an empty code_part and is correctly skipped.
        if re.search(r"\bis_cpu\s*\(", code_part) and "def is_cpu" not in code_part:
            hits.append((i, line.strip()))
    return hits


def looks_weight_related(root: Path, rel_path: str, lineno: int, window: int = 15) -> bool:
    fpath = root / rel_path
    lines = fpath.read_text(errors="replace").splitlines()
    lo = max(0, lineno - window)
    hi = min(len(lines), lineno + window)
    context = "\n".join(lines[lo:hi])
    return bool(WEIGHT_PADDING_HINTS.search(context))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--sglang-root",
        default="/sgl-workspace/sglang/python/sglang",
        help="Path to the sglang package root (contains srt/)",
    )
    args = ap.parse_args()
    root = Path(args.sglang_root)
    if not root.is_dir():
        print(f"FATAL: {root} is not a directory", file=sys.stderr)
        sys.exit(2)

    try:
        modules = discover_real_import_graph()
    except Exception as e:
        print(f"FATAL: could not import entrypoint modules to build the real "
              f"import graph: {e}", file=sys.stderr)
        print("This script must run inside the built image (needs sglang "
              "importable, torch etc. present).", file=sys.stderr)
        sys.exit(2)

    print(f"=== Real import graph: {len(modules)} sglang.srt modules loaded "
          f"by {', '.join(ENTRYPOINT_MODULES)} ===")

    findings = []  # (rel_path, lineno, line, covered, reason)
    seen_rels = set()
    for modname in modules:
        rel = module_to_relpath(modname)
        if rel in seen_rels:
            continue
        seen_rels.add(rel)
        for lineno, line in find_is_cpu_call_sites(root, rel):
            if not looks_weight_related(root, rel, lineno):
                continue
            if rel in REVIEWED_KERNEL_DISPATCH_ONLY or rel in REVIEWED_ALREADY_PADS_ON_CUDA:
                reason = (
                    "kernel-dispatch, not padding"
                    if rel in REVIEWED_KERNEL_DISPATCH_ONLY
                    else "CUDA branch already pads via pad_or_narrow_weight"
                )
                findings.append((rel, lineno, line, True, reason))
                continue
            covered = rel in COVERED_FILES
            findings.append((rel, lineno, line, covered, "shadowed/rewritten by our patch" if covered else ""))

    uncovered = [f for f in findings if not f[3]]

    print(f"=== {len(findings)} weight-padding is_cpu() call sites found in the real import graph ===")
    for rel, lineno, line, covered, reason in sorted(findings, key=lambda t: (t[0], t[1])):
        mark = "OK  " if covered else "GAP "
        suffix = f"  ({reason})" if reason else ""
        print(f"  [{mark}] {rel}:{lineno}: {line}{suffix}")

    if uncovered:
        print()
        print(f"=== {len(uncovered)} UNCOVERED gate(s) -- add to enable_unaligned_tp_on_cuda.py before next rebuild ===")
        for rel, lineno, line, _, _r in sorted(uncovered, key=lambda t: (t[0], t[1])):
            print(f"  {rel}:{lineno}")
        sys.exit(1)

    print()
    print("=== All in-scope weight-padding is_cpu() gates are covered ===")
    sys.exit(0)


if __name__ == "__main__":
    main()
