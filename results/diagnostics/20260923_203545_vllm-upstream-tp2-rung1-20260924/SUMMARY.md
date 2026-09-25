# Tool Calling Regression Report — vllm-upstream-tp2-rung1-20260924
- Date: 2026-09-23T20:35:56.126378
- Base URL: `http://192.168.10.10:8100/v1`
- Model: `local-inference-lab/Qwen3.8-Flash-Next-NVFP4`
- Temperature: `0.0`
- Max Tokens: `1024`
- Status: **FAIL / CONTAMINATED**

## Per-Trial Results

| Trial | Turn | Finish Reason | Tools | Content Len | Comp Tokens | Latency (s) | Contaminated | Status |
|---|---|---|---|---|---|---|---|---|
| 1 | 1 | `tool_calls` | 3 | 80 | 174 | 2.915 | NO | **FAIL** |
| 2 | 1 | `tool_calls` | 3 | 0 | 179 | 2.455 | NO | **PASS** |
| 2 | 2 | `tool_calls` | 3 | 114 | 160 | 2.279 | NO | **PASS** |
| 3 | 1 | `tool_calls` | 3 | 94 | 184 | 2.583 | NO | **FAIL** |