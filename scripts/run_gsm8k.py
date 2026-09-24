#!/usr/bin/env python3
"""run_gsm8k.py — GSM8K evaluation against OpenAI-compatible endpoint.

Downloads GSM8K test.jsonl if not present, queries the endpoint concurrently,
extracts numerical answers, and compares with ground truth.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

DATA_URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"


def ensure_dataset(cache_dir: Path) -> list[dict]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = cache_dir / "gsm8k_test.jsonl"
    if not jsonl_path.exists():
        print(f"Downloading GSM8K test dataset from {DATA_URL}...")
        urllib.request.urlretrieve(DATA_URL, jsonl_path)
    
    problems = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                problems.append(json.loads(line))
    return problems


def extract_prediction(text: str) -> str:
    m = re.findall(r"(?:the\s+)?answer\s+is:?\s*[$]?\s*([-\d.,]+)", text, re.IGNORECASE)
    if m:
        cand = m[-1].replace(",", "").rstrip(".")
        try:
            val = float(cand)
            return str(int(val)) if val.is_integer() else str(val)
        except ValueError:
            pass
    # fallback: look for final #### or final number
    m_box = re.findall(r"####\s*([-\d.,]+)", text)
    if m_box:
        cand = m_box[-1].replace(",", "").rstrip(".")
        try:
            val = float(cand)
            return str(int(val)) if val.is_integer() else str(val)
        except ValueError:
            pass
    nums = re.findall(r"[-+]?(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+)", text)
    if nums:
        candidate = nums[-1].replace(",", "")
        try:
            val = float(candidate)
            return str(int(val)) if val.is_integer() else str(val)
        except ValueError:
            return candidate
    return ""


def clean_target(answer_str: str) -> str:
    parts = answer_str.split("####")
    if len(parts) > 1:
        target = parts[1].strip().replace(",", "")
        try:
            val = float(target)
            return str(int(val)) if val.is_integer() else str(val)
        except ValueError:
            return target
    return answer_str.strip()


def call_model(base_url: str, model: str, question: str, max_tokens: int = 512) -> str:
    url = f"{base_url.rstrip('/')}/chat/completions"
    user_content = (
        f"Solve this math problem. Show your reasoning step by step, and end your response with exactly "
        f"'The answer is: <number>'.\n\nProblem: {question}"
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
        return (msg.get("content") or msg.get("reasoning_content") or "").strip()


def eval_one(prob: dict, base_url: str, model: str, idx: int) -> dict:
    question = prob["question"]
    gold = clean_target(prob["answer"])
    
    t0 = time.perf_counter()
    try:
        response = call_model(base_url, model, question)
        pred = extract_prediction(response)
        correct = (pred == gold)
        dt = time.perf_counter() - t0
        return {
            "index": idx,
            "question": question,
            "gold": gold,
            "prediction": pred,
            "correct": correct,
            "response": response,
            "latency": dt,
            "error": None,
        }
    except Exception as e:
        dt = time.perf_counter() - t0
        return {
            "index": idx,
            "question": question,
            "gold": gold,
            "prediction": "",
            "correct": False,
            "response": "",
            "latency": dt,
            "error": str(e),
        }


def main():
    parser = argparse.ArgumentParser(description="GSM8K evaluator")
    parser.add_argument("--base-url", default="http://127.0.0.1:8100/v1")
    parser.add_argument("--model", default="qwen3.8-flash-next")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--output", default="results/gsm8k.json")
    args = parser.parse_args()

    cache_dir = Path(__file__).resolve().parent.parent / "scratch" / "benchmarks"
    problems = ensure_dataset(cache_dir)
    if args.limit:
        problems = problems[:args.limit]

    print(f"=== GSM8K Evaluation: {args.model} on {args.base_url} ===")
    print(f"Total problems: {len(problems)}, Concurrency: {args.workers}")

    results = []
    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(eval_one, p, args.base_url, args.model, i): i for i, p in enumerate(problems, 1)}
        for count, fut in enumerate(concurrent.futures.as_completed(futures), 1):
            res = fut.result()
            results.append(res)
            mark = "PASS" if res["correct"] else f"FAIL (pred={res['prediction']}, gold={res['gold']})"
            print(f"[{count}/{len(problems)}] Problem #{res['index']}: {mark} ({res['latency']:.2f}s)")

    total_time = time.perf_counter() - t0
    n_correct = sum(1 for r in results if r["correct"])
    acc = (n_correct / len(problems)) * 100

    print("\n" + "=" * 50)
    print(f"GSM8K Accuracy: {acc:.1f}% ({n_correct}/{len(problems)}) in {total_time:.1f}s")
    print("=" * 50)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": args.model,
                "base_url": args.base_url,
                "total": len(problems),
                "correct": n_correct,
                "accuracy": acc,
                "total_time_s": total_time,
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"Saved detailed results to {out_path}")


if __name__ == "__main__":
    main()
