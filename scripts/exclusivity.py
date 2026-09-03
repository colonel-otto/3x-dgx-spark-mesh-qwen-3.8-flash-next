#!/usr/bin/env python3
"""Cluster exclusivity and idle pre-flight gate.

Ported from 3spark-dsv4/scripts/exclusivity.py (2026-09-03). Enforces
benchmark policy requirement 5: ensuring no competing background clients
pollute cluster measurements by checking running/waiting request gauges and
verifying request_success_total deltas. Engine-agnostic vLLM /metrics scrape
-- no changes needed for this repo's eugr/spark-vllm-b12x engine.
"""
import argparse
import json
import re
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_METRICS_URL = "http://127.0.0.1:8100/metrics"

# Counters vLLM splits across label sets; their series must be summed, not
# overwritten, to get the engine-wide total. Gauges are deliberately absent.
_SUMMED_COUNTERS = frozenset({"vllm:request_success_total"})

def get_engine_metrics(metrics_url: str = DEFAULT_METRICS_URL) -> dict[str, float]:
    """Scrapes and parses Prometheus metrics from vLLM engine."""
    try:
        with urllib.request.urlopen(metrics_url, timeout=5) as resp:
            content = resp.read().decode("utf-8")
    except Exception as e:
        raise ConnectionError(f"Failed to query vLLM metrics at {metrics_url}: {e}") from e

    metrics: dict[str, float] = {}
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^([a-zA-Z0-9_:]+)(?:\{[^}]*\})?\s+([0-9.eE+-]+)", line)
        if m:
            name, val = m.group(1), float(m.group(2))
            # vLLM exposes counters split across label sets -- notably
            # request_success_total, which carries one series per finish reason
            # (stop / length / abort / error / repetition). Assigning here would
            # keep only the LAST series and silently discard the rest: a sweep
            # whose requests all finish as "length" (the ignore_eos window we
            # assert) would read a delta of 0 and the gate would pass or fail
            # for entirely the wrong reason. Counters must be summed across
            # labels; gauges (running/waiting) must not be, so they are assigned.
            if name in _SUMMED_COUNTERS:
                metrics[name] = metrics.get(name, 0.0) + val
            else:
                metrics[name] = val
    return metrics

def assert_idle(
    metrics_url: str = DEFAULT_METRICS_URL,
    timeout_s: float = 30.0,
    poll_interval_s: float = 1.0,
    allow_foreign: bool = False,
) -> float:
    """Asserts that the cluster has zero running and zero waiting requests.

    Returns the initial vllm:request_success_total value.
    """
    t_end = time.time() + timeout_s
    while True:
        metrics = get_engine_metrics(metrics_url)
        running = metrics.get("vllm:num_requests_running", 0.0)
        waiting = metrics.get("vllm:num_requests_waiting", 0.0)
        success_total = metrics.get("vllm:request_success_total", 0.0)

        if running == 0.0 and waiting == 0.0:
            return success_total

        if allow_foreign:
            print(f"[WARN] Cluster not idle: running={running}, waiting={waiting} (proceeding due to --allow-foreign)")
            return success_total

        if time.time() >= t_end:
            raise RuntimeError(
                f"Cluster not idle: running={running}, waiting={waiting} after {timeout_s}s timeout. "
                "Another client is currently active on the cluster. Refusing to run benchmark."
            )
        time.sleep(poll_interval_s)

def verify_exclusivity(
    start_success_total: float,
    expected_requests: int,
    metrics_url: str = DEFAULT_METRICS_URL,
    output_file: str | Path | None = None,
    allow_foreign: bool = False,
) -> dict[str, Any]:
    """Verifies that request_success_total delta matches expected_requests exactly."""
    metrics = get_engine_metrics(metrics_url)
    end_success_total = metrics.get("vllm:request_success_total", 0.0)
    actual_delta = int(end_success_total - start_success_total)

    foreign_requests = max(0, actual_delta - expected_requests)
    exclusive_pass = (actual_delta == expected_requests)

    record: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "PASS" if exclusive_pass else ("WARN" if allow_foreign else "FAIL"),
        "start_request_success_total": start_success_total,
        "end_request_success_total": end_success_total,
        "expected_requests": expected_requests,
        "actual_requests_delta": actual_delta,
        "foreign_requests_detected": foreign_requests,
        "is_exclusive": exclusive_pass,
    }

    if output_file is not None:
        p = Path(output_file)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    if not exclusive_pass and not allow_foreign:
        raise RuntimeError(
            f"Exclusivity check failed: expected {expected_requests} requests, but saw delta of {actual_delta} "
            f"({foreign_requests} foreign requests detected). Benchmark results are contaminated and void."
        )

    return record

def main() -> None:
    parser = argparse.ArgumentParser(description="Enforce vLLM cluster exclusivity and idle state.")
    parser.add_argument("--url", default=DEFAULT_METRICS_URL, help="Prometheus metrics URL")
    parser.add_argument("--check-idle", action="store_true", help="Assert cluster is idle and print start success total")
    parser.add_argument("--timeout", type=float, default=30.0, help="Idle check timeout in seconds")
    parser.add_argument("--verify", action="store_true", help="Verify request count delta")
    parser.add_argument("--start-total", type=float, help="Starting request_success_total")
    parser.add_argument("--expected", type=int, help="Expected request count issued by harness")
    parser.add_argument("--out", type=str, help="Path to write exclusivity.json")
    parser.add_argument("--allow-foreign", action="store_true", help="Warn instead of error on foreign traffic")

    args = parser.parse_args()

    if args.check_idle:
        try:
            start_val = assert_idle(args.url, timeout_s=args.timeout, allow_foreign=args.allow_foreign)
            print(f"IDLE_OK start_request_success_total={start_val}")
            sys.exit(0)
        except Exception as e:
            print(f"IDLE_FAIL {e}", file=sys.stderr)
            sys.exit(1)

    if args.verify:
        if args.start_total is None or args.expected is None:
            parser.error("--verify requires --start-total and --expected")
        try:
            rec = verify_exclusivity(
                args.start_total,
                args.expected,
                metrics_url=args.url,
                output_file=args.out,
                allow_foreign=args.allow_foreign,
            )
            print(f"EXCLUSIVITY_{rec['status']} delta={rec['actual_requests_delta']} expected={rec['expected_requests']}")
            sys.exit(0 if rec["status"] in ("PASS", "WARN") else 1)
        except Exception as e:
            print(f"EXCLUSIVITY_FAIL {e}", file=sys.stderr)
            sys.exit(1)

    parser.print_help()

if __name__ == "__main__":
    main()
