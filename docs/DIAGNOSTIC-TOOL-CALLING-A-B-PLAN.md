# Qwen 3.8 Flash Next Tool Calling & Context Compaction Diagnostic Plan

## 1. Context & Observed Behavior

During multi-turn agent runs with **OpenCode CLI** on `ssh bigbox` targeting **Qwen 3.8 Flash Next** (`RadixArk/Qwen3.8-Flash-Next-NVFP4`), the model entered an accelerated context compaction loop. The assistant message appeared to regenerate the OpenCode compaction status block (`<Objective ... Work State ... Next Move ...>`) rather than cleanly proceeding with tool execution.

This document defines the strict, repeatable A/B diagnostic methodology to isolate the root cause before modifying the live deployment.

---

## 2. Hypotheses Under Test

| ID | Hypothesis | Testable Prediction |
|---|---|---|
| **H1 (Tool Call Response Hygiene)** | vLLM with `--tool-call-parser qwen3_xml` is bleeding generated text into `message.content` alongside `message.tool_calls`. | On Turn 1 of a tool request, `choices[0].message.content` will be non-null and contain echoed text. `finish_reason` may or may not be `tool_calls`. |
| **H2 (Multi-Turn Turn-2 Echo)** | The first turn generates a clean tool call, but after receiving the `role: "tool"` execution output, the model regurgitates the prompt's `<Objective...>` block in `message.content` instead of synthesizing the tool result. | Turn 1 is clean (`content=null`), but Turn 2 has a large `completion_tokens` count containing `<Objective>` / `Work State` headers. |
| **H3 (OpenCode / Gateway Formatting)** | The raw vLLM endpoint outputs clean tool calls and clean follow-up text across turns, meaning the duplication occurs in OpenCode's message assembly, compaction prompt construction, or LiteLLM proxy parameter stripping. | All turns in `test_tool_calling_regression.py` pass cleanly (`content=null` on tool turns, no echoed signatures in follow-up turns). |
| **H4 (Chat Template / Tokenizer Stop Token Mismatch)** | The model's Jinja chat template or special token IDs do not emit/stop on `<|im_end|>` after assistant tool call blocks. | Inspecting raw SSE tokens reveals generation continuing past the closing tool XML tags. |

---

## 3. Test Harness Implementation

The diagnostic suite consists of:
1. [`scripts/test_tool_calling_regression.py`](../scripts/test_tool_calling_regression.py):
   - Multi-turn full agent loop execution (Turn 1: Tool Call Generation -> Turn 2: Tool Output Handling).
   - Fixed parameters: `temperature=0.0`, explicit `max_tokens=1024`.
   - Complete telemetry: `finish_reason`, `tool_calls` schema validation, `content` length and string signature scanning, `reasoning_content` length, token usage counters (`prompt_tokens`, `completion_tokens`), and turn latency.
   - Saves all raw JSON responses to `results/diagnostics/<timestamp>_<tag>/trial_<N>_turn_<M>_raw.json`.
2. [`scripts/qwen-next-tool-validate.sh`](../scripts/qwen-next-tool-validate.sh):
   - Executable wrapper for local and cluster execution.

---

## 4. Controlled A/B Testing Procedure

Run 3 identical trials per configuration with server restarts between configuration changes:

```
[Configuration A: Baseline]
       │
       ▼
[Run 3 Trials: test_tool_calling_regression.py --tag baseline]
       │
       ▼
[Record raw JSONs, check Turn 1 & Turn 2 content contamination]
       │
   ┌───┴───────────────────────────────────────┐
   ▼                                           ▼
[Contamination Confirmed on Raw Endpoint]   [Raw Endpoint is Clean]
   │                                           │
   ├─▶ Check Chat Template & Special Tokens    └─▶ Inspect OpenCode Session Assembly
   ├─▶ Test Specific Stop Token in Recipe          & LiteLLM Gateway Routing
   └─▶ Test Prefix-Caching Toggle (Diagnostics)
```

### Execution Steps:

#### Configuration A: Baseline (Current Checked-In Recipe)
- Ensure vLLM TP=3 is running with default recipe (`--reasoning-parser qwen3 --tool-call-parser qwen3_xml --enable-auto-tool-choice --enable-prefix-caching`).
- Run the harness:
  ```bash
  ./scripts/qwen-next-tool-validate.sh http://127.0.0.1:8100/v1 baseline 3
  ```

#### Configuration B: Controlled Stop Token A/B Probe (If H1/H4 trigger)
- Without removing tool parsers, test adding targeted stop token overrides to prevent text overrun while preserving parallel tool calls.
- Run the harness:
  ```bash
  ./scripts/qwen-next-tool-validate.sh http://127.0.0.1:8100/v1 stop_probe 3
  ```

#### Configuration C: Prefix Caching Diagnostic Toggle
- Test with `--enable-prefix-caching` omitted to isolate whether recurrent linear attention state affects multi-turn prefill consistency.
- Run the harness:
  ```bash
  ./scripts/qwen-next-tool-validate.sh http://127.0.0.1:8100/v1 prefix_off 3
  ```

---

## 5. Pass / Fail Criteria for Clean Operation

A clean agent response must satisfy:

```json
{
  "choices": [
    {
      "finish_reason": "tool_calls",
      "message": {
        "role": "assistant",
        "content": null,
        "tool_calls": [
          {
            "id": "call_xxx",
            "type": "function",
            "function": {
              "name": "view_file",
              "arguments": "{\"path\":\"design-system/ui-visual-surface-contract.json\"}"
            }
          }
        ]
      }
    }
  ]
}
```

- **Turn 1**: `finish_reason == "tool_calls"`, `message.tool_calls` length $\ge 1$, `message.content` is `null` or empty string.
- **Turn 2**: Model receives tool output and either emits next tool call or clear synthesized answer; `message.content` must **not** contain `<Objective>`, `Work State`, `Next Move`, or repeated status blocks.
