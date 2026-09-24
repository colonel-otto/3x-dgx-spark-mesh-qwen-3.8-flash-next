# Tool Calling Regression Report — baseline
- Date: 2026-09-02T21:20:44.241318
- Base URL: `http://192.168.10.10:8100/v1`
- Model: `qwen3.8-flash-next-nvfp4`
- Temperature: `0.0`
- Max Tokens: `1024`
- Status: **PASS**

## Per-Trial Results

| Trial | Turn | Finish Reason | Tools | Content Len | Comp Tokens | Latency (s) | Contaminated | Status |
|---|---|---|---|---|---|---|---|---|
| 1 | 1 | `tool_calls` | 1 | 0 | 108 | 4.838 | NO | **PASS** |
| 1 | 2 | `tool_calls` | 1 | 0 | 97 | 2.738 | NO | **PASS** |
| 2 | 1 | `tool_calls` | 1 | 0 | 105 | 2.853 | NO | **PASS** |
| 2 | 2 | `tool_calls` | 1 | 0 | 105 | 2.944 | NO | **PASS** |
| 3 | 1 | `tool_calls` | 1 | 0 | 107 | 3.416 | NO | **PASS** |
| 3 | 2 | `tool_calls` | 1 | 0 | 198 | 6.059 | NO | **PASS** |