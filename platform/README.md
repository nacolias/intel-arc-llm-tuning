# Platform

Everything below the inference engine: kernel driver, firmware, GPU user-space runtime, Level Zero, collectives, and the PyTorch XPU build.

## Stack layers

| Layer | Component | Where it comes from |
|---|---|---|
| Kernel driver | Linux `xe` | host kernel |
| Firmware | GuC, HuC | `linux-firmware` on the host |
| GPU user-space runtime | intel/compute-runtime (Level Zero and OpenCL driver) | usually inside the container |
| Shader compiler | IGC | paired with compute-runtime |
| Level Zero loader | `level-zero` | container |
| SYCL runtime / Unified Runtime | `intel-sycl-rt`, `intel-cmplr-lib-ur` | PyTorch XPU wheels |
| Collectives | oneCCL | PyTorch XPU wheels, or rebuilt |
| Framework | PyTorch XPU | container |
| Kernels | vllm-xpu-kernels, oneDNN | engine image |

## Version matrix

Record every combination that was actually run, including failures. Link the benchmark or experiment that used it.

| Date | Host | Kernel | GuC | compute-runtime | IGC | L0 loader | oneCCL | PyTorch | Engine image | Result |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-09 | dual-b70-5800x | TODO | TODO | 26.27.39122.11 | 2.38.2 | 1.32.0 | 2022.0.0 | 2.13.0+xpu | `vllm/vllm-openai-xpu:v0.28.0` | production; see [Qwen3.8-27B](../models/qwen3.8-27b/) |

## Upgrade candidates

| Component | Candidate | Why | Status |
|---|---|---|---|
| compute-runtime | 26.31.39395.13 (IGC 2.40.13) | latest tag | planned |
| compute-runtime | master | BMG-only commits: midthread preemption timer, out-of-order list sync skip, IPC offset fix, FD leak fix | research |
| oneCCL | public rebuild | the pinned `libccl` failed a graph-replay exactness test in a community lab; rebuilt one passed 100 of 100 | research |

## Environment variable reference

Keep one row per variable we have an opinion on. Link the finding or experiment that justifies it.

| Variable | Value | Layer | Why | Evidence |
|---|---|---|---|---|
| `CCL_TOPO_P2P_ACCESS` | `1` | oneCCL | P2P works when both cards share a root complex | [finding](../findings/bmg-pcie-p2p-works.md) |
| `CCL_ENABLE_SYCL_KERNELS` | `1` | oneCCL | graph-capturable collectives | [finding](../findings/oneccl-topo-not-graph-capturable.md) |
| `CCL_SYCL_ALLREDUCE_SIMPLE_THRESHOLD` | `4294967296` | oneCCL | keeps all message sizes on the capturable simple path | [finding](../findings/oneccl-topo-not-graph-capturable.md) |
| `CCL_SYCL_ALLREDUCE_LL` | `twoshots` (to test) | oneCCL | 32% lower per-collective latency measured on 4 ranks | UNVERIFIED on 2 ranks |
| `ZE_AFFINITY_MASK` | per rank, set before L0 init | Level Zero | required for TP graph capture | [finding](../findings/tp-xpu-graphs-need-per-worker-affinity.md) |
| `UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS` | `1` | Unified Runtime | allows single allocations over 4 GB | upstream-documented |
| `SYCL_CACHE_PERSISTENT` | do not set | SYCL | cross-restart kernel cache poisoning on Battlemage | [finding](../findings/no-persistent-sycl-cache-on-bmg.md) |

## Known issues

Link to findings and upstream issues. Remove entries once fixed in a version we run, and note which version fixed them.
