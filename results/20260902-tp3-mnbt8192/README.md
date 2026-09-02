# Qwen 3.8 Flash Next (~180B MoE) TP=3 Benchmark Bundle

- **Date**: 2026-09-02
- **Model**: RadixArk/Qwen3.8-Flash-Next-NVFP4 (~180B total parameters, MoE + GDN hybrid)
- **Cluster**: 3x DGX Spark (GB10, 128 GB Unified Memory per node)
- **Interconnect**: 4x ConnectX-7 400 Gbps RoCE direct-attach mesh
- **Topology / TP**: TP=3 across all 3 nodes (sparkmain, spark1, spark2)
- **Engine**: vLLM (ugr/spark-vllm-b12x:latest)
- **MTP**: Native Multi-Token Prediction draft model (SpeculativeConfig(method='mtp', num_speculative_tokens=1))
- **Attention**: FlashInfer (FP8 KV cache) + B12X MoE
- **HyperConnections**: Multi-stream 4-branch residual stream with group-level RMSNorm and gated mixing
- **Max Model Length**: 32,768
- **Max Num Batched Tokens**: 8,192

## Results Summary (ows.tsv)

| Nodes | MNBT | Concurrency ($) | Median Decode (tok/s/user) | Aggregate Throughput (tok/s) | TTFT (ms) |
|---|---|---|---|---|---|
| 3 | 8192 | 1 | 40.3 | 37.3 | 227 |
| 3 | 8192 | 4 | 32.3 | 111.4 | 605 |
| 3 | 8192 | 8 | 23.9 | 149.4 | 968 |
| 3 | 8192 | 16 | 18.0 | 192.5 | 1644 |

## Acceptance Quality Verification

All acceptance battery tests passed with 100% precision:
- **Models Endpoint**: qwen3.8-flash-next-nvfp4 / qwen3.8-flash-next
- **Core Reasoning & Correctness**:
  - Capital lookup: PASS ( Paris)
  - Arithmetic: PASS (17 x 23 = 391)
  - Logical deduction: PASS (No)
- **Needle In Haystack Retrieval**: PASS (~1.5k tokens, exact key OPAL-4482)
- **Text Quality & Degeneration Check**: PASS (unique-word ratio 0.774, zero repetition)
