"""Multi-turn tool-call reliability probe (reproducer for issue #3).

Runs N agent-style sessions against the local OpenAI endpoint. Each session gives the model a
small toolset and a task that needs several sequential tool calls, feeds back canned tool
results, and checks every tool call for: (a) name in the allowed set, (b) arguments parse as JSON
with the required keys. Reports per-session call counts and every malformed call verbatim.
"""
import json, sys, time, urllib.request

URL = "http://127.0.0.1:8100/v1/chat/completions"
MODEL = sys.argv[1] if len(sys.argv) > 1 else "qwen3.8-flash-next"
SESSIONS = int(sys.argv[2]) if len(sys.argv) > 2 else 6
MAX_TURNS = int(sys.argv[3]) if len(sys.argv) > 3 else 14

def tool(name, desc, props, req):
    return {"type": "function", "function": {"name": name, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": req}}}

TOOLS = [
    tool("bash", "Run a shell command", {"command": {"type": "string"}, "description": {"type": "string"}}, ["command"]),
    tool("read", "Read a file", {"filePath": {"type": "string"}}, ["filePath"]),
    tool("glob", "Find files by glob pattern", {"pattern": {"type": "string"}}, ["pattern"]),
    tool("grep", "Search file contents", {"pattern": {"type": "string"}, "path": {"type": "string"}}, ["pattern"]),
    tool("edit", "Edit a file", {"filePath": {"type": "string"}, "oldString": {"type": "string"}, "newString": {"type": "string"}}, ["filePath", "oldString", "newString"]),
    tool("todowrite", "Update the todo list", {"todos": {"type": "array", "items": {"type": "object"}}}, ["todos"]),
]
NAMES = {t["function"]["name"]: t["function"]["parameters"]["required"] for t in TOOLS}

TASKS = [
    "Find every Python file under /srv/app, grep them for 'TODO', read the file with the most hits, then summarise it. Use tools step by step, one at a time.",
    "List files in /etc/nginx with bash, read nginx.conf, grep for 'proxy_pass' in /etc/nginx, then write a todo list of follow-ups. Use tools step by step.",
    "Inspect /opt/service: glob for *.yaml, read the first result, grep for 'replicas' across the directory, then edit the file to set replicas to 3. Use tools step by step.",
]

def canned(name, args):
    if name == "glob":  return "\n".join(f"/srv/app/mod{i}.py" for i in range(1, 6))
    if name == "grep":  return "/srv/app/mod3.py:14:# TODO fix\n/srv/app/mod3.py:52:# TODO cache\n/srv/app/mod1.py:7:# TODO tidy"
    if name == "read":  return "def main():\n    # TODO fix\n    return 0\n" * 3
    if name == "bash":  return "total 12\n-rw-r--r-- 1 root root 1024 nginx.conf\n-rw-r--r-- 1 root root 88 mime.types"
    return "ok"

import os
STREAM = os.environ.get("STREAM", "0") == "1"

def call(messages):
    body = {"model": MODEL, "messages": messages, "tools": TOOLS, "tool_choice": "auto",
            "max_tokens": 512, "temperature": 0.2}
    if STREAM:
        body["stream"] = True
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        if not STREAM:
            return json.load(r)["choices"][0]
        content, calls = "", {}
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:") or line.endswith("[DONE]"):
                continue
            d = json.loads(line[5:])["choices"][0].get("delta", {})
            content += d.get("content") or ""
            for tc in d.get("tool_calls") or []:
                c = calls.setdefault(tc["index"], {"id": tc.get("id") or f"call_{tc['index']}",
                                                   "type": "function", "function": {"name": "", "arguments": ""}})
                if tc.get("id"): c["id"] = tc["id"]
                f = tc.get("function") or {}
                c["function"]["name"] += f.get("name") or ""
                c["function"]["arguments"] += f.get("arguments") or ""
        return {"message": {"content": content or None, "tool_calls": [calls[i] for i in sorted(calls)]}}

total_calls = bad = 0
first_bad_at = []
for s in range(SESSIONS):
    msgs = [{"role": "user", "content": TASKS[s % len(TASKS)]}]
    calls_here = 0
    for turn in range(MAX_TURNS):
        ch = call(msgs)
        m = ch["message"]
        tcs = m.get("tool_calls") or []
        if not tcs:
            break
        msgs.append({"role": "assistant", "content": m.get("content"), "tool_calls": tcs})
        for tc in tcs:
            total_calls += 1; calls_here += 1
            name = tc["function"]["name"]; raw = tc["function"]["arguments"]
            problem = None
            if name not in NAMES:
                problem = f"unknown tool name {name!r}"
            else:
                try:
                    args = json.loads(raw)
                    missing = [k for k in NAMES[name] if k not in args]
                    if missing: problem = f"missing args {missing}"
                except Exception as e:
                    problem = f"arguments not JSON ({e})"
            if problem:
                bad += 1; first_bad_at.append((s, calls_here))
                print(f"  BAD session={s} call#{calls_here}: {problem}\n    name={name!r}\n    args={raw!r}")
            msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": canned(name, {})})
    print(f"session {s}: {calls_here} tool calls")

print(f"RESULT total_calls={total_calls} malformed={bad} first_bad_at(session,call#)={first_bad_at[:5]}")
