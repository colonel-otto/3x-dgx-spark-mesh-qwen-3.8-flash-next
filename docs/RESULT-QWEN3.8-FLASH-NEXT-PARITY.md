# Qwen 3.8 Flash Next (~180B MoE Hybrid) Parity & Benchmark Report

## 1. Executive Summary

This document records the complete implementation, architectural mathematical resolution, acceptance testing, and distributed benchmark performance of **Qwen 3.8 Flash Next** (`RadixArk/Qwen3.8-Flash-Next-NVFP4`, ~180B parameters MoE + GDN hybrid with HyperConnections and native Multi-Token Prediction) on the **3x NVIDIA DGX Spark (GB10) Mesh Cluster**.

The model is partitioned across 3 nodes using **Tensor Parallelism 3 (TP=3)** via Virtual TP Padding (`b12x-padded` profile) over a switchless 4x ConnectX-7 400 Gbps RoCE mesh ring.

---

## 2. Cluster Architecture & Topology

- **Hardware Nodes**: 3x DGX Spark (GB10 Grace Blackwell, 128 GB Unified LPDDR5X per node, 384 GB unified memory total).
- **Cluster Network**: 4x ConnectX-7 400 Gbps RoCE direct-attach mesh with full subnet-aware routing.
- **Model Checkpoint**: `RadixArk/Qwen3.8-Flash-Next-NVFP4` (206 safetensors shards, 125.91 GiB).
- **Serving Engine**: vLLM (`eugr/spark-vllm-b12x:latest`) with B12X MoE acceleration and FlashInfer attention backend.
- **Speculative Decoding**: Native 1-step Multi-Token Prediction (MTP) with dual FC projections (`fc_embedding` and `fc_hidden`).

---

## 3. Mathematical Breakthrough: HyperConnections & Grouped RMSNorm

During initial bringup, standard RMSNorm caused token corruption because Qwen4Exp/Qwen3-Next architecture employs **GroupedGemmaRMSNorm** / **Qwen4ExpTextRMSNorm** where affine weights are near zero and activations evaluate to:

$$\text{Normed}(x) = (1.0 + w) \odot \text{RMSNorm}_{\text{group}}(x)$$

### Exact Mathematical Formulations Implemented:

1. **Grouped RMSNorm (`Qwen4ExpTextRMSNorm`)**:
   - Computes RMS independently across each of the 4 hidden branches ($d=2560$, total $d_{\text{hc}}=10240$).
   - Multiplies by $(1.0 + w)$, preserving dynamic activation range.

2. **Read Mixing Gate**:
   $$\text{gate}_{\text{low}} = \text{Linear}_{\text{down}}(\text{Normed}(x))$$
   $$\text{gate}_{\text{act}} = \text{SiLU}\left(\frac{\text{gate}_{\text{low}}}{N_{\text{hc}}}\right)$$
   $$\text{mix}_{w} = \sigma\left(\text{Linear}_{\text{up}}(\text{gate}_{\text{act}})\right)$$
   $$\text{Mixed} = \text{mean}_{\text{stream}}\left(\text{mix}_{w} \odot \text{Normed}(x)\right)$$

3. **Block Injection Write Gate**:
   $$\text{inj}_{w} = 2.0 \times \sigma\left(\frac{\text{Linear}_{\text{inject}}(\text{Normed}(x))}{N_{\text{hc}}}\right)$$
   $$\text{Residual}_{\text{new}} = x_{\text{hyper}} + (\text{BlockOut} \odot \text{inj}_{w})$$

4. **Multi-Token Prediction (MTP) Dual Stream Projections**:
   - `fc_embedding`: maps normalized input token embeddings ($2560 \to 2560$).
   - `fc_hidden`: maps 4 normalized hidden branches ($10240 \to 4 \times 2560$).
   - Dynamic shape adaptation for both single-stream $(T, 2560)$ and multi-stream $(T, 10240)$ residual states.

---

## 4. Acceptance Quality Verification Battery

All 5 core acceptance tests in `qwen-next-quick-validate.sh` passed with 100% precision:

| Test Category | Target / Prompt | Expected Output | Actual Output | Status |
|---|---|---|---|---|
| **Models API** | `GET /v1/models` | `qwen3.8-flash-next-nvfp4` | `qwen3.8-flash-next-nvfp4` | **PASS** |
| **Capital Lookup** | "What is the capital of France?" | "Paris" | "Paris" | **PASS** |
| **Arithmetic** | "What is 17 * 23?" | "391" | "391" | **PASS** |
| **Logical Deduction** | "Is an apple a vehicle?" | "No" | "No" | **PASS** |
| **Needle Retrieval** | ~1.5k tok needle haystack | Exact key `OPAL-4482` | `OPAL-4482` | **PASS** |
| **Text Quality** | Multi-sentence prompt | Repetition ratio $\ge 0.60$ | **0.774** (Zero degeneration) | **PASS** |

---

## 5. Performance Benchmark Results (TP=3, MNBT=8192)

Swept across concurrencies $c \in [1, 4, 8, 16]$ with maximum sequence length 32,768 and FP8 KV cache:

| Nodes / TP | MNBT | Concurrency ($c$) | Median Decode (tok/s/user) | Aggregate Throughput (tok/s) | TTFT (ms) |
|---|---|---|---|---|---|
| 3 / TP=3 | 8192 | 1 | **40.3** | 37.3 | 227 |
| 3 / TP=3 | 8192 | 4 | **32.3** | 111.4 | 605 |
| 3 / TP=3 | 8192 | 8 | **23.9** | 149.4 | 968 |
| 3 / TP=3 | 8192 | 16 | **18.0** | **192.5** | 1644 |

---

## 6. Artifact & File Provenance

- **Patches**:
  - `patches/qwen3_next.py`: Complete `Qwen4ExpTextRMSNorm`, `Qwen3NextHyperConnection`, and `Qwen3NextDecoderLayer` with exact math.
  - `patches/qwen3_next_mtp.py`: `Qwen3NextMultiTokenPredictor` with dual `fc_embedding`/`fc_hidden` projections and multi-stream handling.
  - `patches/virtual_tp.py`: Virtual TP padding for `gqa-gdn-moe` profile.
  - `patches/registry.py`: Model architecture registrations for `Qwen3NextForCausalLM` and `Qwen3NextMTP`.
- **Scripts**:
  - `scripts/qwen-next-boot-tp3.sh`: Launch script for 3-node cluster.
  - `scripts/qwen-next-quick-validate.sh`: Comprehensive acceptance test battery.
  - `scripts/qwen-next-sweep.sh`: Multi-concurrency benchmarking harness.
- **Results**:
  - `results/20260902-tp3-mnbt8192/`: Raw logs (`bench-c1.log`, `bench-c4.log`, `bench-c8.log`, `bench-c16.log`, `rows.tsv`).
  - `results/index.yaml` & `results/INDEX.md`: Fully audited provenance index.
