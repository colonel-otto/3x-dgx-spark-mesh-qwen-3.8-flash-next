#!/usr/bin/env python3
"""Matched-workload decode harness (originally a replica of MiaAI's 2026-08-14
matrix methodology): unique cold prefix per request, thinking=false,
min_tokens==max_tokens, ignore_eos, numbered-word instruction.

Reports median per-stream decode tok/s after first token, plus the observed
spread across trials.

BENCHMARK-POLICY.md compliance (see 3spark-dsv4/docs/BENCHMARK-POLICY.md):
  Req 2 -- the decode window is forced (min_tokens == max_tokens + ignore_eos),
           defaults to 256, and every reply is ASSERTED to return exactly
           max_tokens. A short completion raises WindowCollapse and aborts the
           run; it must never be silently recorded. A 128-token window measured
           against a 256-token arm is NOT a matched comparison -- the SGLang
           bundles of 2026-09-03 were measured at 128 while every vLLM bundle
           was measured at 256, which invalidated that cross-stack table.
  Req 3 -- per-trial values are printed and the full sorted spread is reported,
           never a bare median.
  Determinism -- temperature defaults to 0 with a fixed seed so speculative
           acceptance variance does not enter a small trial sample. top_p is
           only sent when sampling is actually enabled.

Usage: bench-miaai.py --base-url ... --prompt 256 --output-tokens 256 --concurrency 1
"""
import argparse, asyncio, json, statistics, time, urllib.error, urllib.request

def request_json(url, body):
    for attempt in range(4):
        req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=3600) as resp:
                return json.load(resp)
        except urllib.error.URLError:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)

def tokenize_url(base_url):
    return base_url.removesuffix("/v1") + "/tokenize"

class WindowCollapse(RuntimeError):
    """Raised when the engine returned fewer tokens than the forced window."""


def count_tokens(base_url, model, text):
    return request_json(tokenize_url(base_url), {"model": model, "prompt": text})["count"]


def build_prompt(base_url, model, target, nonce):
    unit = "benchmark context datum "
    text = f"unique request {nonce} " + unit * max(1, target // 3)
    while True:
        count = request_json(tokenize_url(base_url), {"model": model, "prompt": text})["count"]
        if count >= target:
            return text
        text += unit * max(1, (target - count) // 3)

def stream_one(base_url, model, prompt, output_tokens, temperature, top_p, seed):
    # ignore_eos + min_tokens == max_tokens forces the window; the assert further
    # down is what makes it a guarantee rather than a polite request.
    instruction = (chr(10) + f"Return exactly {output_tokens} numbered "
                   "lowercase English words, then stop.")
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt + instruction}],
        "stream": True, "stream_options": {"include_usage": True},
        "temperature": temperature,
        "max_tokens": output_tokens, "min_tokens": output_tokens, "ignore_eos": True,
        "chat_template_kwargs": {"thinking": False},
    }
    # top_p is only meaningful when sampling; sending it at temperature 0 invites
    # an engine-specific reading of a parameter that should be inert.
    if temperature > 0:
        body["top_p"] = top_p
    if seed is not None:
        body["seed"] = seed
    req = urllib.request.Request(f"{base_url}/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    first = None
    usage = None
    output = []
    with urllib.request.urlopen(req, timeout=3600) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line[6:])
            choices = event.get("choices") or []
            delta = choices[0].get("delta", {}) if choices else {}
            if first is None and (delta.get("content") or delta.get("reasoning") or delta.get("reasoning_content")):
                first = time.perf_counter()
            output.append(delta.get("content") or "")
            if event.get("usage"):
                usage = event["usage"]
    finished = time.perf_counter()
    if usage is None:
        raise WindowCollapse(
            "engine returned no usage block; cannot verify the decode window. "
            "stream_options.include_usage must be honoured by the engine.")
    completed = usage.get("completion_tokens", 0)
    # BENCHMARK-POLICY.md Req 2: a collapsed window measures draft-acceptance
    # variance, not throughput. Abort loudly instead of recording the number.
    if completed != output_tokens:
        raise WindowCollapse(
            f"decode window collapsed: asked for exactly {output_tokens} tokens, "
            f"engine returned {completed}. This run is not publishable; the "
            f"engine ignored min_tokens/ignore_eos or hit a cap.")
    ttft = (first or finished) - started
    return {"ttft_s": ttft, "elapsed_s": finished - started,
            "output_tokens": completed,
            "output_tok_s": completed / max(0.001, finished - (first or finished)),
            "prompt_tokens": (usage or {}).get("prompt_tokens", 0)}

async def run_case(base_url, model, target_prompt_tokens, concurrency, nonce_base,
                   output_tokens, temperature, top_p, seed):
    prompts = await asyncio.gather(*[
        asyncio.to_thread(build_prompt, base_url, model, target_prompt_tokens,
                          f"p{target_prompt_tokens}-c{concurrency}-{nonce_base}-r{index}")
        for index in range(concurrency)
    ])
    started = time.perf_counter()
    results = await asyncio.gather(*[
        asyncio.to_thread(stream_one, base_url, model, p,
                          output_tokens, temperature, top_p, seed)
        for p in prompts])
    elapsed = time.perf_counter() - started
    total = sum(r["output_tokens"] for r in results)
    prompt_lens = [r["prompt_tokens"] for r in results if r["prompt_tokens"]]
    return {"concurrency": concurrency, "elapsed_s": elapsed,
            "prompt_tokens_min": min(prompt_lens) if prompt_lens else 0,
            "prompt_tokens_max": max(prompt_lens) if prompt_lens else 0,
            "output_tokens_each": output_tokens,
            "aggregate_tok_s": total / max(0.001, elapsed),
            "median_ttft_s": statistics.median(r["ttft_s"] for r in results),
            "median_output_tok_s": statistics.median(r["output_tok_s"] for r in results),
            "requests": results}

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8888/v1")
    ap.add_argument("--model", default="deepseek-v4-flash-0731")
    ap.add_argument("--prompt", type=int, default=256)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--repeat", type=int, default=5, help="how many sequential trials (unique nonces)")
    ap.add_argument("--output-tokens", type=int, default=256,
                    help="forced decode window. BENCHMARK-POLICY.md Req 2 requires >=256; "
                         "every arm in a comparison MUST use the same value.")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="default 0 (deterministic). Non-zero adds sampling variance.")
    ap.add_argument("--top-p", type=float, default=0.95, help="only sent when --temperature > 0")
    ap.add_argument("--seed", type=int, default=1234, help="fixed by default; --seed -1 to omit")
    ap.add_argument("--warmup", type=int, default=0,
                    help="discarded warm-up trials run before the measured ones")
    args = ap.parse_args()
    seed = None if args.seed < 0 else args.seed

    if args.output_tokens < 256:
        print(f"WARNING: --output-tokens {args.output_tokens} is below the 256 floor in "
              f"BENCHMARK-POLICY.md Req 2. A short window measures draft-acceptance "
              f"variance, not throughput. This run is not publishable.", flush=True)

    print(f"config: prompt>={args.prompt} output_tokens={args.output_tokens} "
          f"temperature={args.temperature} "
          f"top_p={args.top_p if args.temperature > 0 else 'unset'} seed={seed} "
          f"concurrency={args.concurrency} warmup={args.warmup} repeat={args.repeat}",
          flush=True)

    for rep in range(args.warmup):
        await run_case(args.base_url, args.model, args.prompt, args.concurrency,
                       f"warmup{rep}", args.output_tokens, args.temperature,
                       args.top_p, seed)
        print(f"warmup {rep}: discarded", flush=True)

    rows = []
    for rep in range(args.repeat):
        case = await run_case(args.base_url, args.model, args.prompt, args.concurrency, rep,
                              args.output_tokens, args.temperature, args.top_p, seed)
        rows.append(case)
        med = case["median_output_tok_s"]
        print(f"trial {rep}: c={case['concurrency']} p={args.prompt} "
              f"median_decode={med:.1f} tok/s agg={case['aggregate_tok_s']:.1f} "
              f"ttft={case['median_ttft_s']*1000:.0f}ms "
              f"prompt_tok=[{case['prompt_tokens_min']}-{case['prompt_tokens_max']}] "
              f"n={[r['output_tokens'] for r in case['requests']]}", flush=True)

    # BENCHMARK-POLICY.md Req 3: publish the sorted spread, never a bare median.
    if args.repeat > 1:
        meds = sorted(r["median_output_tok_s"] for r in rows)
        aggs = sorted(r["aggregate_tok_s"] for r in rows)
        spread = (meds[-1] - meds[0]) / meds[0] * 100 if meds[0] else 0.0
        print()
        print(f"per-trial decode (sorted): {[f'{m:.1f}' for m in meds]}")
        print(f"per-trial agg    (sorted): {[f'{a:.1f}' for a in aggs]}")
        print(f"FINAL: median-of-trials decode = {statistics.median(meds):.1f} tok/s "
              f"[min {meds[0]:.1f} max {meds[-1]:.1f}, spread {spread:.1f}%]")
        print(f"FINAL: median-of-trials agg    = {statistics.median(aggs):.1f} tok/s "
              f"[min {aggs[0]:.1f} max {aggs[-1]:.1f}]")
        if spread > 15:
            print(f"NOTE: {spread:.1f}% spread across trials. Per "
                  f"feedback_noise_floor_before_reporting_delta, do not report a "
                  f"delta against another arm smaller than this spread.")

asyncio.run(main())
