# Tool Calling Regression Report — vllm-colonel-tp2-rung2-20260924
- Date: 2026-09-23T21:13:31.433203
- Base URL: `http://192.168.10.10:8100/v1`
- Model: `qwen3.8-flash-next-nvfp4`
- Temperature: `0.0`
- Max Tokens: `1024`
- Status: **FAIL / CONTAMINATED**

## Per-Trial Results

| Trial | Turn | Finish Reason | Tools | Content Len | Comp Tokens | Latency (s) | Contaminated | Status |
|---|---|---|---|---|---|---|---|---|
| 1 | 1 | `tool_calls` | 2 | 112 | 144 | 2.702 | NO | **FAIL** |
| 2 | 1 | `tool_calls` | 3 | 90 | 198 | 3.064 | NO | **FAIL** |
| 3 | 1 | `tool_calls` | 2 | 82 | 140 | 2.037 | NO | **FAIL** |