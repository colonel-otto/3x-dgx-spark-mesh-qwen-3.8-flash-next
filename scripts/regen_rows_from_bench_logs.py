#!/usr/bin/env python3
"""Regenerate rows.tsv from bench-c*.log (median-of-trials decode, agg, ttft,
decode min/max). Written 2026-09-04 because qwen-next-sweep.sh's '^FINAL' grep
captured both FINAL lines of the policy-compliant harness and split every row
across two lines (SGLang TP=2 bundle 20260904T1308Z). Reads the per-trial
lines, so it is independent of the harness's summary formatting.

Usage: regen_rows_from_bench_logs.py <bundle_dir> <nodes> [mnbt]
"""
import re
import statistics
import sys
from pathlib import Path

bundle = Path(sys.argv[1]); nodes = sys.argv[2]; mnbt = sys.argv[3] if len(sys.argv) > 3 else "8192"
rows = ["nodes\tmnbt\tc\tmedian_decode_tok_s\tagg_tok_s\tttft_ms\tdecode_min\tdecode_max"]
pat = re.compile(r"median_decode=([0-9.]+) tok/s agg=([0-9.]+) ttft=([0-9]+)ms")
for c in (1, 4, 8, 16):
    f = bundle / f"bench-c{c}.log"
    if not f.exists():
        continue
    trials = [tuple(map(float, m.groups())) for m in pat.finditer(f.read_text(errors="replace"))]
    if not trials:
        continue
    dec = [t[0] for t in trials]; agg = [t[1] for t in trials]; ttft = [t[2] for t in trials]
    rows.append("\t".join([nodes, mnbt, str(c), f"{statistics.median(dec):.1f}", f"{statistics.median(agg):.1f}",
                           f"{statistics.median(ttft):.0f}", f"{min(dec):.1f}", f"{max(dec):.1f}"]))
(bundle / "rows.tsv").write_text("\n".join(rows) + "\n")
print("\n".join(rows))
