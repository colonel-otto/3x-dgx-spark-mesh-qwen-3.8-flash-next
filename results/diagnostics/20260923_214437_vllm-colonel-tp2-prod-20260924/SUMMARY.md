# Tool Calling Regression Report — vllm-colonel-tp2-prod-20260924
- Date: 2026-09-23T21:44:46.362823
- Base URL: `http://192.168.10.10:8100/v1`
- Model: `qwen3.8-flash-next-nvfp4`
- Temperature: `0.0`
- Max Tokens: `1024`
- Status: **FAIL / CONTAMINATED**

## Per-Trial Results

| Trial | Turn | Finish Reason | Tools | Content Len | Comp Tokens | Latency (s) | Contaminated | Status |
|---|---|---|---|---|---|---|---|---|
| 1 | 1 | `tool_calls` | 2 | 0 | 130 | 2.464 | NO | **PASS** |
| 1 | 2 | `tool_calls` | 2 | 0 | 121 | 1.865 | NO | **PASS** |
| 2 | 1 | `tool_calls` | 2 | 69 | 145 | 2.058 | NO | **FAIL** |
| 3 | 1 | `tool_calls` | 2 | 69 | 144 | 2.047 | NO | **FAIL** |