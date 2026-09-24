#!/usr/bin/env python3
"""run_humaneval.py — Standard HumanEval pass@1 evaluator against OpenAI-compatible endpoint.

Downloads HumanEval.jsonl.gz if not present, sends prompts concurrently to the endpoint,
extracts the completion, executes tests in a restricted gVisor container, and reports pass@1.
Requires Docker with the runsc runtime and a locally available trusted Python image.
See docs/HUMANEVAL-SANDBOX.md. There is no host-execution fallback.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import json
import re
import subprocess
import time
import urllib.request
import uuid
from pathlib import Path

DATA_URL = "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz"
SANDBOX_IMAGE = "python:3.11-slim"


class SandboxError(RuntimeError):
    """Infrastructure failure: abort evaluation instead of scoring it as a failure."""


def ensure_dataset(cache_dir: Path) -> list[dict]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    gz_path = cache_dir / "HumanEval.jsonl.gz"
    if not gz_path.exists():
        print(f"Downloading HumanEval dataset from {DATA_URL}...")
        urllib.request.urlretrieve(DATA_URL, gz_path)
    
    problems = []
    with gzip.open(gz_path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                problems.append(json.loads(line))
    return problems


def extract_code(prompt: str, response: str) -> str:
    # Check for markdown code blocks
    match = re.search(
        r"```(?:python)?[^\S\r\n]*\r?\n(.*?)^[^\S\r\n]*```",
        response, re.DOTALL | re.MULTILINE | re.IGNORECASE,
    )
    if match:
        code = match.group(1)
    else:
        code = response

    # If code already includes the prompt, don't duplicate
    if prompt.strip() in code:
        return code
    return prompt + "\n" + code


def run_test(full_code: str, test_code: str, entry_point: str, timeout: float = 5.0,
             sandbox_image: str = SANDBOX_IMAGE) -> bool:
    script = f"{full_code}\n\n{test_code}\n\ncheck({entry_point})\n"
    name = f"humaneval-{uuid.uuid4().hex}"
    # No host mounts, inherited environment, network, or GPU access. The image is
    # trusted infrastructure; candidate code is sent only over stdin after create.
    create = [
        "docker", "create", "--pull=never", "--name", name, "--runtime=runsc",
        "--network=none", "--read-only", "--user=65534:65534",
        "--cap-drop=ALL", "--security-opt=no-new-privileges=true",
        "--pids-limit=64", "--memory=256m", "--memory-swap=256m", "--cpus=1",
        "--ulimit=nofile=64:64", "--ulimit=fsize=1048576:1048576",
        "--tmpfs=/tmp:rw,noexec,nosuid,nodev,size=16m", "--workdir=/tmp",
        "--log-driver=none", "--interactive", "--entrypoint=python",
        sandbox_image, "-I", "-B", "-",
    ]
    created = False
    try:
        subprocess.run(create, capture_output=True, text=True, timeout=30, check=True)
        created = True
        mounts = subprocess.run(
            ["docker", "inspect", "--format={{json .Mounts}}", name],
            capture_output=True, text=True, timeout=30, check=True,
        )
        # Docker images can declare VOLUME entries, creating writable host
        # volumes even without a --volume argument. Reject them before start.
        if any(mount.get("Type") != "tmpfs" for mount in json.loads(mounts.stdout)):
            raise SandboxError("Sandbox image introduced host-backed mounts; use a volume-free image")
        try:
            # Discard untrusted output so a print loop cannot exhaust host memory.
            proc = subprocess.run(
                ["docker", "start", "--attach", "--interactive", name],
                input=script, text=True, encoding="utf-8", stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return False
        state = subprocess.run(
            ["docker", "inspect", "--format={{json .State}}", name],
            capture_output=True, text=True, timeout=30, check=True,
        )
        status = json.loads(state.stdout)
        if status.get("Error") or status.get("Status") != "exited":
            raise SandboxError(f"Sandbox did not execute successfully: {status}")
        return proc.returncode == 0 and status["ExitCode"] == 0
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError) as exc:
        raise SandboxError(f"HumanEval sandbox unavailable: {exc}") from exc
    finally:
        # Kill the whole container, including descendants, even after a timeout.
        # Cleanup failure is fatal, not a scored candidate failure.
        if created:
            try:
                subprocess.run(["docker", "rm", "--force", "--volumes", name],
                               capture_output=True, text=True, timeout=30, check=True)
            except (OSError, subprocess.SubprocessError) as exc:
                raise SandboxError(f"Cannot remove sandbox {name}: {exc}") from exc


def call_model(base_url: str, model: str, prompt: str, max_tokens: int = 512) -> str:
    url = f"{base_url.rstrip('/')}/chat/completions"
    user_content = (
        f"Complete the following Python code. Return only the executable code block inside ```python and ```.\n\n"
        f"```python\n{prompt}```"
    )
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": user_content}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "chat_template_kwargs": {"thinking": False, "enable_thinking": False},
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.load(resp)
        msg = data["choices"][0]["message"]
        return msg.get("content") or msg.get("reasoning_content") or ""


def eval_one(prob: dict, base_url: str, model: str, sandbox_image: str = SANDBOX_IMAGE) -> dict:
    task_id = prob["task_id"]
    prompt = prob["prompt"]
    test_code = prob["test"]
    entry_point = prob["entry_point"]
    
    t0 = time.perf_counter()
    try:
        response = call_model(base_url, model, prompt)
        full_code = extract_code(prompt, response)
        passed = run_test(full_code, test_code, entry_point, sandbox_image=sandbox_image)
        dt = time.perf_counter() - t0
        return {
            "task_id": task_id,
            "passed": passed,
            "latency": dt,
            "error": None,
        }
    except SandboxError:
        raise
    except Exception as e:
        dt = time.perf_counter() - t0
        return {
            "task_id": task_id,
            "passed": False,
            "latency": dt,
            "error": str(e),
        }


def main():
    parser = argparse.ArgumentParser(description="HumanEval pass@1 evaluator")
    parser.add_argument("--base-url", default="http://127.0.0.1:8100/v1")
    parser.add_argument("--model", default="qwen3.8-flash-next")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default="results/humaneval.json")
    parser.add_argument("--sandbox-image", default=SANDBOX_IMAGE,
                        help="Trusted local Python image (prefer an immutable digest); requires runsc")
    args = parser.parse_args()

    if not run_test("def candidate(): return 42", "def check(fn): assert fn() == 42",
                    "candidate", timeout=30, sandbox_image=args.sandbox_image):
        raise SandboxError("Sandbox preflight failed; refusing to evaluate completions")

    cache_dir = Path(__file__).resolve().parent.parent / "scratch" / "benchmarks"
    problems = ensure_dataset(cache_dir)
    if args.limit:
        problems = problems[:args.limit]

    print(f"=== HumanEval Evaluation: {args.model} on {args.base_url} ===")
    print(f"Total problems: {len(problems)}, Concurrency: {args.workers}")

    results = []
    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(eval_one, p, args.base_url, args.model, args.sandbox_image): p for p in problems}
        for i, fut in enumerate(concurrent.futures.as_completed(futures), 1):
            res = fut.result()
            results.append(res)
            status_char = "PASS" if res["passed"] else "FAIL"
            print(f"[{i}/{len(problems)}] {res['task_id']}: {status_char} ({res['latency']:.2f}s)")

    total_time = time.perf_counter() - t0
    n_passed = sum(1 for r in results if r["passed"])
    pass_rate = (n_passed / len(problems)) * 100

    print("\n" + "=" * 50)
    print(f"HumanEval pass@1: {pass_rate:.1f}% ({n_passed}/{len(problems)}) in {total_time:.1f}s")
    print("=" * 50)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": args.model,
                "base_url": args.base_url,
                "sandbox_runtime": "runsc",
                "sandbox_image": args.sandbox_image,
                "total": len(problems),
                "passed": n_passed,
                "pass_at_1": pass_rate,
                "total_time_s": total_time,
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"Saved detailed results to {out_path}")


if __name__ == "__main__":
    main()
