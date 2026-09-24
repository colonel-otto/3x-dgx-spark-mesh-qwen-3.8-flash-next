# Context Garble Sweep — 2026-09-04 13:03

model: qwen3.8-flash-next | endpoint: http://192.168.10.10:8100/v1 | runs/length: 1 | cold prefill: forced (unique nonce)

| ctx_len | run | verdict | finish | secs | reasoning_ch | tool_calls | flags | content |
|---|---|---|---|---|---|---|---|---|
| 2048 | 0 | CLEAN | stop | 2.0 | 0 | - | - | 'The capital of Idaho is **Boise**. Its population is roughly **235,000' |
| 8192 | 0 | CLEAN | stop | 5.7 | 0 | - | - | 'The capital of Idaho is Boise, with a population of roughly 235,000 in' |
| 32768 | 0 | CLEAN | stop | 11.5 | 0 | - | - | 'The capital of Idaho is **Boise**. Its population is roughly **235,000' |
| 131072 | 0 | CLEAN | stop | 47.6 | 0 | - | - | 'The capital of Idaho is Boise, with a city population of roughly 235,0' |
