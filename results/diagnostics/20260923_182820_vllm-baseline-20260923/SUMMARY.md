# Tool Calling Regression Report — vllm-baseline-20260923
- Date: 2026-09-23T18:28:41.077848
- Base URL: `http://192.168.10.10:8100/v1`
- Model: `qwen3.8-flash-next-nvfp4`
- Temperature: `0.0`
- Max Tokens: `1024`
- Status: **PASS**

## Per-Trial Results

| Trial | Turn | Finish Reason | Tools | Content Len | Comp Tokens | Latency (s) | Contaminated | Status |
|---|---|---|---|---|---|---|---|---|
| 1 | 1 | `tool_calls` | 1 | 0 | 108 | 3.026 | NO | **PASS** |
| 1 | 2 | `tool_calls` | 1 | 0 | 113 | 3.607 | NO | **PASS** |
| 2 | 1 | `tool_calls` | 1 | 0 | 133 | 3.784 | NO | **PASS** |
| 2 | 2 | `tool_calls` | 2 | 0 | 112 | 3.742 | NO | **PASS** |
| 3 | 1 | `tool_calls` | 1 | 0 | 82 | 2.847 | NO | **PASS** |
| 3 | 2 | `tool_calls` | 1 | 0 | 116 | 3.065 | NO | **PASS** |