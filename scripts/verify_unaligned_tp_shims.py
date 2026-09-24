#!/usr/bin/env python3
"""Functional check (not static grep) that our is_cpu() shims actually take
effect at import time, for every module the static audit
(audit_unaligned_tp_gates.py) says should be shadowed.

WHY THIS EXISTS: the static audit only proves the shim SOURCE TEXT exists
somewhere in the file -- it can't tell a live shadow from one that gets
immediately clobbered by a later `from sglang.srt.utils import is_cpu`
re-import. That exact bug happened 2026-09-03 in mamba.py: the shim was
inserted before the is_cpu import instead of after, so is_cpu() inside that
module still returned the real (unshadowed) value despite the patch
"succeeding" and the audit script showing no static gap once COVERED_FILES
was updated by hand. This script catches that class of bug by actually
importing each module with SGLANG_FORCE_UNALIGNED_TP=1 set and calling
is_cpu() from inside it, rather than trusting that inserting text near an
import means the shadow survives to the end of the module.

Usage (inside the built image):
    SGLANG_FORCE_UNALIGNED_TP=1 python3 scripts/verify_unaligned_tp_shims.py
"""
import importlib
import os
import sys

MODULES_EXPECTING_SHADOWED_IS_CPU = [
    "sglang.srt.model_loader.weight_utils",
    "sglang.srt.layers.attention.mamba.mamba",
]


def main():
    if os.environ.get("SGLANG_FORCE_UNALIGNED_TP") != "1":
        print(
            "FATAL: run with SGLANG_FORCE_UNALIGNED_TP=1 set "
            "(this script tests that the flag actually changes is_cpu())",
            file=sys.stderr,
        )
        sys.exit(2)

    failures = []
    for modname in MODULES_EXPECTING_SHADOWED_IS_CPU:
        try:
            mod = importlib.import_module(modname)
        except Exception as e:
            failures.append(f"{modname}: import failed: {e}")
            continue
        is_cpu_fn = getattr(mod, "is_cpu", None)
        if is_cpu_fn is None:
            failures.append(f"{modname}: has no is_cpu attribute at all")
            continue
        try:
            result = is_cpu_fn()
        except Exception as e:
            failures.append(f"{modname}.is_cpu() raised: {e}")
            continue
        if result is not True:
            failures.append(
                f"{modname}.is_cpu() returned {result!r} with the flag set -- "
                "shim is not live (likely clobbered by a later re-import)"
            )
        else:
            print(f"  OK  {modname}.is_cpu() == True under the flag")

    if failures:
        print()
        print(f"=== {len(failures)} shim(s) NOT actually live ===")
        for f in failures:
            print(f"  FAIL  {f}")
        sys.exit(1)

    print()
    print("=== All expected is_cpu() shims are live ===")
    sys.exit(0)


if __name__ == "__main__":
    main()
