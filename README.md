# 3x DGX Spark Mesh: Qwen 3.8 Flash Next (~180B MoE)

High-performance deployment, virtual tensor parallel ($TP=3$) sharding, and native Multi-Token Prediction (MTP) speculative decoding recipes for **Qwen 3.8 Flash Next** (`RadixArk/Qwen3.8-Flash-Next-NVFP4`) on a 3x NVIDIA DGX Spark cluster connected via a switchless 400 Gbps RoCE mesh.

---

## Architecture Highlights

- **Base Architecture**: `Qwen4ExpForConditionalGeneration` (~180B parameter high-sparsity MoE).
- **MoE Configuration**: 512 routed experts with top-10 activation per token + shared experts.
- **Attention Design**: Hybrid Gated DeltaNet linear attention (3 layers) + Full Attention (1 layer) repeating across 48 layers.
- **Speculative Acceleration**: Native built-in 4B Multi-Token Prediction (MTP) head (`method="mtp"`).
- **Quantization**: NVIDIA NVFP4 via ModelOpt with 16-element alignment.
- **Cluster Mesh**: 3x DGX Spark (GB10 unified memory, 4x ConnectX-7 400 Gbps RoCE mesh ring).

---

## Repository Structure

```text
3x-dgx-spark-mesh-qwen-3.8-flash-next/
├── configs/
│   ├── qwen3.8-next-flash-nvfp4-tp2.yaml # 2-Node TP=2 MoE + MTP recipe
│   └── qwen3.8-next-flash-nvfp4-tp3.yaml # 3-Node TP=3 MoE + MTP recipe
├── scripts/
│   ├── qwen-next-boot-tp2.sh             # Launch 2-node cluster
│   ├── qwen-next-boot-tp3.sh             # Launch 3-node cluster
│   ├── qwen-next-quick-validate.sh      # 6-part acceptance test battery
│   ├── qwen-next-sweep.sh               # Benchmark sweep across c in {1, 4, 8, 16}
│   ├── qwen-next-stop.sh                # Cluster shutdown script
│   ├── check_no_sensitive.py            # Pre-commit sensitive data scanner
│   └── generate_results_index.py        # Provenance index generator
├── patches/
│   ├── virtual_tp.py                    # MoE & Gated DeltaNet TP=3 sharding plan
│   └── vocab_parallel_embedding.py      # Padded vocabulary parallel embedding layer
├── results/
│   ├── index.yaml                       # Machine-readable provenance index
│   └── INDEX.md                         # Rendered provenance summary table
└── docs/
    └── RESULT-QWEN3.8-FLASH-NEXT-PARITY.md
```

---

## Quick Start

### 1. Launch 3-Node Cluster with Native MTP ($TP=3$)
```bash
./scripts/qwen-next-boot-tp3.sh 8192
```

### 2. Run Acceptance Battery
```bash
./scripts/qwen-next-quick-validate.sh
```

### 3. Run Benchmark Sweep
```bash
./scripts/qwen-next-sweep.sh 3 8192 ./results/my-sweep
```
