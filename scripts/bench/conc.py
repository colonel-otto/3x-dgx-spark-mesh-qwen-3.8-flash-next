"""Concurrency benchmark: N simultaneous streaming chat requests, unique prompts.

usage: python conc.py <base_url> <model> <c> [<c> ...]
Reports per-request decode tok/s (median, min) and aggregate tok/s per level.
Same prompt family and settings for every engine: thinking off, 512 max tokens,
model-default sampling.
"""
import json
import statistics
import sys
import threading
import time
import urllib.request

BASE, MODEL = sys.argv[1], sys.argv[2]
LEVELS = [int(x) for x in sys.argv[3:]] or [1, 4, 16]
TOPICS = ["tides", "volcanoes", "glaciers", "coral reefs", "deserts", "rivers",
          "monsoons", "auroras", "earthquakes", "rainforests", "tundra", "caves",
          "hurricanes", "wetlands", "savannas", "fjords"]


def one(i, out, nonce):
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content":
                      f"[{nonce}-{i}] Write a 400-word essay on {TOPICS[i % len(TOPICS)]}."}],
        "max_tokens": 512, "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    t0 = time.time()
    first = None
    n = 0
    req = urllib.request.Request(f"{BASE}/chat/completions", body,
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            if not line.startswith(b"data: {"):
                continue
            d = json.loads(line[6:])
            if d.get("usage"):
                n = d["usage"]["completion_tokens"]
            if first is None and d.get("choices") and d["choices"][0]["delta"].get("content"):
                first = time.time()
    t1 = time.time()
    out[i] = (n, first - t0 if first else None, (n - 1) / (t1 - first) if first and n > 1 else 0.0, t0, t1)


for c in LEVELS:
    out = {}
    nonce = int(time.time())
    threads = [threading.Thread(target=one, args=(i, out, nonce)) for i in range(c)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    toks = sum(v[0] for v in out.values())
    span = max(v[4] for v in out.values()) - min(v[3] for v in out.values())
    rates = [v[2] for v in out.values()]
    ttfts = [v[1] for v in out.values() if v[1] is not None]
    print(f"c={c:<3} per-req decode median {statistics.median(rates):6.1f} min {min(rates):6.1f} tok/s | "
          f"aggregate {toks / span:7.1f} tok/s | ttft median {statistics.median(ttfts):.2f}s | tokens {toks}")
