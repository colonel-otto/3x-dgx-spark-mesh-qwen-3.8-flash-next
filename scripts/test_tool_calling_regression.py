#!/usr/bin/env python3
"""test_tool_calling_regression.py — Diagnostic regression test for Qwen 3.8 Flash Next.

Tests the complete multi-turn agent loop:
  - Turn 1: User prompt with compaction header -> Model generates tool_calls
  - Turn 2: Assistant tool message + Role 'tool' execution result -> Model generates follow-up

Captures full telemetry:
  - finish_reason ('tool_calls' vs 'stop' vs 'length')
  - tool_calls valid JSON schema
  - message.content contamination check (search for echoed <Objective> / Work State)
  - reasoning_content separation
  - completion_tokens, prompt_tokens, and total_tokens
  - saves all raw JSON responses for audit

Install dependencies with: python -m pip install -r scripts/requirements-eval.txt
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "view_file",
            "description": "View the contents of a file from the local filesystem.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute or relative path to file to read.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Optional maximum number of lines to return.",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep_search",
            "description": "Search for a pattern or regular expression in files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query pattern."},
                    "path": {"type": "string", "description": "Path to search within."},
                },
                "required": ["query"],
            },
        },
    },
]

COMPACTION_PROMPT_BLOCK = """<Objective
- Verify whether every invariant, owner, policy value, baseline, count, and coverage claim in docs/ui-canon/mobile-invariants.md is correctly implemented and consistently reflected in code/config.
Important Details
- Conversation began in a read-only/analysis-oriented verification mode.
- User question: is everything in @docs/ui-canon/mobile-invariants.md implemented correctly?
- The doc claims: 18 distinct mobile invariants. 18 enforced, 0 unowned.
- Document identifies four enforcement homes:
  - scripts/validate/lib/mobile-visual-containment-engine.ts
  - tests/e2e/full-check.spec.ts
  - tools/eslint-plugin-local/rules/no-viewport-height-fallback.mjs
  - scripts/validate/validate-mobile-viewport-meta.ts
Work State
Completed
- Read and reviewed docs/ui-canon/mobile-invariants.md in full.
- Built verification plan.
Active
- Beginning static verification of whether the implementation matches the invariant ownership map.
- Still need to verify:
  - design-system/ui-visual-surface-contract.json existence
  - design-system/mobile-app-invariants.json baselines
Blocked
- Need to check if design-system/ui-visual-surface-contract.json exists.
Next Move
1. Read design-system/ui-visual-surface-contract.json
Relevant Files
- docs/ui-canon/mobile-invariants.md
- design-system/ui-visual-surface-contract.json
- design-system/mobile-app-invariants.json>"""

ECHO_SIGNATURES = [
    "<Objective",
    "Work State",
    "Important Details",
    "Next Move",
    "Relevant Files",
    "Completed\n- Read and reviewed",
    "Active\n- Beginning static",
]


def post_json(url: str, payload: dict[str, Any], timeout: float = 120.0) -> tuple[dict[str, Any], float]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            elapsed = time.perf_counter() - t0
            raw = resp.read().decode("utf-8")
            return json.loads(raw), elapsed
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code}: {e.reason}\nBody: {err_body}") from e


def validate_tool_calls(tool_calls: Any) -> tuple[list[dict[str, Any]], list[str]]:
    """Validate the wire envelope and the actual declared JSON parameter schemas."""
    if tool_calls is None:
        return [], []
    if not isinstance(tool_calls, list):
        return [], ["tool_calls must be an array"]
    declared = {tool["function"]["name"]: tool["function"]["parameters"] for tool in TOOLS}
    parsed, errors, seen_ids = [], [], set()

    def reject_constant(value):
        raise ValueError(f"Invalid JSON constant: {value}")

    for index, call in enumerate(tool_calls):
        try:
            if not isinstance(call, dict) or call.get("type") != "function":
                raise ValueError("expected a function tool call")
            call_id = call.get("id")
            if not isinstance(call_id, str) or not call_id.strip() or call_id in seen_ids:
                raise ValueError("tool call ID must be nonempty and unique")
            seen_ids.add(call_id)
            function = call.get("function")
            if not isinstance(function, dict):
                raise ValueError("missing function object")
            name = function.get("name")
            if not isinstance(name, str) or name not in declared:
                raise ValueError(f"unknown function: {name!r}")
            if not isinstance(function.get("arguments"), str):
                raise ValueError("arguments must be a JSON string")
            arguments = json.loads(function["arguments"], parse_constant=reject_constant)
            violations = list(Draft202012Validator(declared[name]).iter_errors(arguments))
            if violations:
                raise ValueError("; ".join(error.message for error in violations))
            parsed.append({"id": call_id, "name": name, "arguments": arguments})
        except ValueError as exc:
            errors.append(f"tool_calls[{index}]: {exc}")
    return parsed, errors


def analyze_turn(
    turn_idx: int,
    raw_response: dict[str, Any],
    expected_finish: str | None,
    must_have_tools: bool,
    must_have_clean_content: bool,
) -> dict[str, Any]:
    choice = raw_response["choices"][0]
    msg = choice.get("message", {})
    usage = raw_response.get("usage", {})

    finish_reason = choice.get("finish_reason")
    tool_calls = msg.get("tool_calls")
    parsed_calls, tool_errors = validate_tool_calls(tool_calls)
    content = msg.get("content")
    reasoning = msg.get("reasoning") or msg.get("reasoning_content")

    # Contamination check
    contaminated = False
    found_signatures = []
    if isinstance(content, str) and content:
        for sig in ECHO_SIGNATURES:
            if sig in content:
                contaminated = True
                found_signatures.append(sig)

    reasons = list(tool_errors)
    if content is not None and not isinstance(content, str):
        reasons.append("content must be text or null")
    # A usable continuation is either executable calls or a completed final
    # answer. Reasoning alone and truncated output cannot advance the loop.
    if tool_calls:
        if finish_reason != "tool_calls":
            reasons.append("tool calls require finish_reason=tool_calls")
    elif finish_reason != "stop" or not isinstance(content, str) or not content.strip():
        reasons.append("expected valid tool calls or a nonempty final answer with finish_reason=stop")
    passed = not reasons

    if expected_finish and finish_reason != expected_finish:
        passed = False
        reasons.append(f"finish_reason={finish_reason} (expected {expected_finish})")

    if must_have_tools and not tool_calls:
        passed = False
        reasons.append("expected tool_calls but none found")

    if must_have_clean_content and isinstance(content, str) and content.strip():
        passed = False
        reasons.append(f"content should be null/empty on tool turn, got {len(content)} chars")

    if contaminated:
        passed = False
        reasons.append(f"detected echoed compaction signatures: {found_signatures}")

    return {
        "turn": turn_idx,
        "finish_reason": finish_reason,
        "tool_calls_count": len(tool_calls) if isinstance(tool_calls, list) else 0,
        "tool_calls": tool_calls,
        "validated_tool_calls": parsed_calls,
        "tool_validation_errors": tool_errors,
        "content_length": len(content) if isinstance(content, str) else 0,
        "content": content,
        "reasoning_length": len(reasoning) if reasoning else 0,
        "reasoning": reasoning,
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
        "contaminated": contaminated,
        "found_signatures": found_signatures,
        "passed": passed,
        "reasons": reasons,
    }


def run_trial(
    trial_num: int,
    base_url: str,
    model_name: str,
    temperature: float,
    max_tokens: int,
    output_dir: Path,
) -> list[dict[str, Any]]:
    print(f"\n==================== TRIAL {trial_num} ====================")
    endpoint = f"{base_url.rstrip('/')}/chat/completions"
    history: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": "You are a software engineering assistant. Use available tools to read files and inspect code.",
        },
        {
            "role": "user",
            "content": f"{COMPACTION_PROMPT_BLOCK}\n\nPlease inspect design-system/ui-visual-surface-contract.json and report what you find.",
        },
    ]

    turn_results = []

    # ---------------- TURN 1: User prompt -> Expect tool call ----------------
    print(f"[Turn 1] Sending prompt with compaction block + request...")
    body_t1 = {
        "model": model_name,
        "messages": history,
        "tools": TOOLS,
        "tool_choice": "auto",
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    raw_t1, elapsed_t1 = post_json(endpoint, body_t1)
    (output_dir / f"trial_{trial_num}_turn_1_raw.json").write_text(
        json.dumps(raw_t1, indent=2), encoding="utf-8"
    )

    t1_analysis = analyze_turn(
        turn_idx=1,
        raw_response=raw_t1,
        expected_finish="tool_calls",
        must_have_tools=True,
        must_have_clean_content=True,
    )
    t1_analysis["latency_sec"] = round(elapsed_t1, 3)
    turn_results.append(t1_analysis)

    print(f"  -> Finish: {t1_analysis['finish_reason']}")
    print(f"  -> Tools:  {t1_analysis['tool_calls_count']}")
    print(f"  -> Content length: {t1_analysis['content_length']} chars")
    print(f"  -> Completion tokens: {t1_analysis['completion_tokens']}")
    print(f"  -> Pass:   {t1_analysis['passed']}")
    if t1_analysis["reasons"]:
        print(f"     Issues: {', '.join(t1_analysis['reasons'])}")

    msg_t1 = raw_t1["choices"][0]["message"]
    history.append(msg_t1)

    tool_calls = msg_t1.get("tool_calls")
    if not t1_analysis["passed"]:
        print("  [ERROR] Cannot proceed to Turn 2: Turn 1 failed validation.")
        return turn_results

    # ---------------- TURN 2: Tool execution result -> Expect follow-up ----------------
    # Append simulated tool outputs for all emitted tool calls
    for tc in t1_analysis["validated_tool_calls"]:
        tc_id = tc["id"]
        fn_name = tc["name"]
        arguments = tc["arguments"]
        if fn_name == "view_file":
            result = f"Error: File not found: {arguments['path']}"
        else:
            result = f"No matches for {arguments['query']!r} in {arguments.get('path', '.')}"
        history.append({
            "role": "tool",
            "tool_call_id": tc_id,
            "name": fn_name,
            "content": result,
        })

    print(f"\n[Turn 2] Sending tool execution result (file not found)...")
    body_t2 = {
        "model": model_name,
        "messages": history,
        "tools": TOOLS,
        "tool_choice": "auto",
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    raw_t2, elapsed_t2 = post_json(endpoint, body_t2)
    (output_dir / f"trial_{trial_num}_turn_2_raw.json").write_text(
        json.dumps(raw_t2, indent=2), encoding="utf-8"
    )

    t2_analysis = analyze_turn(
        turn_idx=2,
        raw_response=raw_t2,
        expected_finish=None,
        must_have_tools=False,
        must_have_clean_content=False,
    )
    t2_analysis["latency_sec"] = round(elapsed_t2, 3)
    turn_results.append(t2_analysis)

    print(f"  -> Finish: {t2_analysis['finish_reason']}")
    print(f"  -> Tools:  {t2_analysis['tool_calls_count']}")
    print(f"  -> Content length: {t2_analysis['content_length']} chars")
    print(f"  -> Completion tokens: {t2_analysis['completion_tokens']}")
    print(f"  -> Contaminated: {t2_analysis['contaminated']}")
    print(f"  -> Pass:   {t2_analysis['passed']}")
    if t2_analysis["reasons"]:
        print(f"     Issues: {', '.join(t2_analysis['reasons'])}")

    msg_t2 = raw_t2["choices"][0]["message"]
    history.append(msg_t2)

    return turn_results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run multi-turn tool calling and compaction regression tests."
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8100/v1"),
        help="Target OpenAI-compatible API base URL (default: http://127.0.0.1:8100/v1)",
    )
    parser.add_argument(
        "--model-name",
        default=os.environ.get("MODEL_NAME", "qwen3.8-flash-next-nvfp4"),
        help="Target model identifier (default: qwen3.8-flash-next-nvfp4)",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=3,
        help="Number of repeatable trials to execute (default: 3)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Maximum completion tokens per turn (default: 1024)",
    )
    parser.add_argument(
        "--tag",
        default="baseline",
        help="Diagnostic label for this A/B configuration (e.g. baseline, prefix_off, stop_tuned)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for raw JSON payloads (default: results/diagnostics/<timestamp>-<tag>)",
    )

    args = parser.parse_args()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.out_dir is None:
        root_dir = Path(__file__).resolve().parents[1]
        out_dir = root_dir / "results" / "diagnostics" / f"{timestamp}_{args.tag}"
    else:
        out_dir = args.out_dir

    out_dir.mkdir(parents=True, exist_ok=True)

    print("======================================================================")
    print(" QWEN 3.8 FLASH NEXT — TOOL CALLING & COMPACTION REGRESSION HARNESS")
    print("======================================================================")
    print(f"Target URL:    {args.base_url}")
    print(f"Model Name:    {args.model_name}")
    print(f"Configuration: {args.tag}")
    print(f"Trials:        {args.trials}")
    print(f"Temperature:   {args.temperature}")
    print(f"Max Tokens:    {args.max_tokens}")
    print(f"Output Directory: {out_dir}")

    all_trials = []
    overall_pass = True

    for i in range(1, args.trials + 1):
        trial_res = run_trial(
            trial_num=i,
            base_url=args.base_url,
            model_name=args.model_name,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            output_dir=out_dir,
        )
        all_trials.append(trial_res)
        for turn in trial_res:
            if not turn["passed"]:
                overall_pass = False

    # Render summary table
    summary_lines = [
        f"# Tool Calling Regression Report — {args.tag}",
        f"- Date: {datetime.datetime.now().isoformat()}",
        f"- Base URL: `{args.base_url}`",
        f"- Model: `{args.model_name}`",
        f"- Temperature: `{args.temperature}`",
        f"- Max Tokens: `{args.max_tokens}`",
        f"- Status: **{'PASS' if overall_pass else 'FAIL / CONTAMINATED'}**",
        "",
        "## Per-Trial Results",
        "",
        "| Trial | Turn | Finish Reason | Tools | Content Len | Comp Tokens | Latency (s) | Contaminated | Status |",
        "|---|---|---|---|---|---|---|---|---|",
    ]

    for t_idx, trial in enumerate(all_trials, start=1):
        for turn in trial:
            status_badge = "PASS" if turn["passed"] else "FAIL"
            contam_badge = "YES" if turn["contaminated"] else "NO"
            summary_lines.append(
                f"| {t_idx} | {turn['turn']} | `{turn['finish_reason']}` | "
                f"{turn['tool_calls_count']} | {turn['content_length']} | "
                f"{turn['completion_tokens']} | {turn['latency_sec']} | "
                f"{contam_badge} | **{status_badge}** |"
            )

    summary_file = out_dir / "SUMMARY.md"
    summary_file.write_text("\n".join(summary_lines), encoding="utf-8")

    meta = {
        "tag": args.tag,
        "base_url": args.base_url,
        "model_name": args.model_name,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "overall_pass": overall_pass,
        "trials": all_trials,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("\n========================= SUMMARY =========================")
    print("\n".join(summary_lines))
    print(f"\nSaved full results to: {out_dir}")

    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
