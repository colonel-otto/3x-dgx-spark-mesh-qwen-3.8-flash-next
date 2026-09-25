import json,time,urllib.request
import os
MODEL=os.environ.get("MODEL","qwen3.8-flash-next")
BASE=os.environ.get("BASE","http://192.168.10.10:8100/v1")  # set to your head node
for i in range(3):
    body=json.dumps({"model":MODEL,"messages":[{"role":"user","content":"Write a Python module implementing a thread-safe LRU cache with TTL expiry, type hints, docstrings and pytest tests."}],"max_tokens":512,"stream":True,"stream_options":{"include_usage":True},"chat_template_kwargs":{"enable_thinking":False}}).encode()
    t0=time.time();first=None;n=0
    r=urllib.request.urlopen(urllib.request.Request(BASE+"/chat/completions",body,{"Content-Type":"application/json"}))
    for line in r:
        if not line.startswith(b"data: {"):continue
        d=json.loads(line[6:])
        if d.get("usage"):n=d["usage"]["completion_tokens"]
        if first is None and d.get("choices") and d["choices"][0]["delta"].get("content"):first=time.time()
    t=time.time()
    print(f"ttft={first-t0:.2f}s tokens={n} decode={(n-1)/(t-first):.1f} tok/s")
