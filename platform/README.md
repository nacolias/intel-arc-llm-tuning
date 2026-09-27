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
| 2026-09-23 | [quad-b70-5800x-pex88096](../hardware/hosts/quad-b70-5800x-pex88096.md) | 7.0.0-34-generic | 70.58.0 | 26.27.39122.11 (image) | 2.38.2 | 1.32.0 | TODO (not read from the image) | 2.13.0+xpu | `vllm/vllm-openai-xpu:v0.28.0` | TP=4 129.1 tok/s single-stream with `--enable-sleep-mode`; runs without it exhausted host RAM ([benchmark](../models/qwen3.8-27b/benchmarks/2026-09-23-tp4-quad-b70.md), [finding](../findings/xe-vram-overcommit-spills-into-host-ram.md)) |
| 2026-09-25 | [quad-b70-5800x-pex88096](../hardware/hosts/quad-b70-5800x-pex88096.md) | 7.0.0-34-generic | 70.58.0 | 26.35.39758.11 (host) | 2.41.9 | 1.32.0 | n/a | n/a | none: host build of llama.cpp SYCL, PR #28243 at `6fcaa16` plus local patches, oneAPI 2026.1, build `20260925-f47a6a5f3` | Qwen3.8-Flash-Next on four cards, 262K context, MTP (multi-token prediction) draft head; see [Qwen3.8-Flash-Next](../models/qwen3.8-flash-next/) |

Host runtime versions for the 2026-09-25 row were read from the installed packages on 2026-09-25.

## Card firmware

Firmware on the card's own flash, updated through the card's GSC (graphics security controller). Read-only check with `igsc` 1.3.1 on 2026-09-25, host [quad-b70-5800x-pex88096](../hardware/hosts/quad-b70-5800x-pex88096.md):

| Part | Running | Newest on LVFS, the Linux Vendor Firmware Service (checked 2026-09-25) | Status |
|---|---|---|---|
| GSC firmware code | 31.1058 on all four B70s | 31.1062 (2026-04-14) for every B70; release note mentions better low-power entry when no display is connected | not applied; effect UNVERIFIED |
| OPROM (option ROM) code | 23.1065 | 23.1066 for every B70 | not applied |
| Firmware data (board config) | 203.2 (ASRock, subsystem 1849:6025); 203.44 (Intel, 8086:1701) | 203.46 for the Intel board only; the ASRock B70 entry targets subsystem 1849:6023, not 6025 | not applied; do not force another board's data |
| OPROM data | 23.1065 | 23.1066 for the Intel board only | not applied |

- **Tooling.** Every B70 package on LVFS requires fwupd 2.1.4 or newer, which added B70 support ([fwupd #10389](https://github.com/fwupd/fwupd/pull/10389)). Ubuntu 26.04 ships fwupd 2.1.1; the fwupd snap had 2.1.7 on 2026-09-25. [`igsc`](https://github.com/intel/igsc) can flash the same Intel-signed images.
- **Treat the update as one-way.** The card refuses images with a lower security version than the one it runs. Never run `igsc arbsvn commit`.
- **Power-cycle fully after flashing,** including any separate PSU that feeds the cards.
- **Host-loaded firmware** (from `linux-firmware`, not flashed): GuC 70.58.0, HuC 8.2.10, DMC 2.6.

## Upgrade candidates

| Component | Candidate | Why | Status |
|---|---|---|---|
| compute-runtime | 26.31.39395.13 (IGC 2.40.13) | latest tag | planned |
| compute-runtime | master | BMG-only commits: midthread preemption timer, out-of-order list sync skip, IPC offset fix, FD leak fix | research |
| oneCCL | public rebuild | the pinned `libccl` failed a graph-replay exactness test in a community lab; rebuilt one passed 100 of 100 | research |
| B70 card firmware | GSC 31.1062, OPROM 23.1066 | LVFS note mentions better low-power entry without a display; needs fwupd 2.1.4 or newer | planned; one card first, measure idle before and after |
| compute-runtime in the vLLM image | 26.35.39758.x | the image's 26.27.39122.11 is in the range of the host-RAM mirror issues [#953](https://github.com/intel/compute-runtime/issues/953) and [#980](https://github.com/intel/compute-runtime/issues/980) | research |

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
| `ZES_ENABLE_SYSMAN` | `1` | Level Zero | lets llama.cpp SYCL query each GPU's free memory, which layer split uses | upstream-documented ([SYCL backend docs](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/SYCL.md)); used in every Flash-Next run |
| `GGML_SYCL_SPARSE_FA` | `1` for Qwen3.8-Flash-Next | llama.cpp SYCL | sparse flash attention for the model's QSA sparse-attention layers; upstream #28796 covers single-token batches, local patch 0006 extends it to batches of up to 32 query rows. Default 0. | [experiment](../models/qwen3.8-flash-next/experiments/2026-09-25-sparse-fa-multi-token.md) |
| `LLAMA_QSA_POOLED` | leave unset (on) | llama.cpp (local patch 0007) | caches pooled QSA indexer keys instead of recomputing them every ubatch; `0` disables it | [experiment](../models/qwen3.8-flash-next/experiments/2026-09-25-pooled-qsa-key-cache.md) |
| `GGML_SYCL_OP_PROFILE` | `<seconds>`, diagnostics only | llama.cpp SYCL (local patch 0004) | per-op timing shares, printed every `<seconds>` while `/tmp/ggml-sycl-op-profile` exists; it serializes ops, so use the shares, not the times | [patches](../models/qwen3.8-flash-next/configs/patches/README.md) |
| `GGML_SYCL_MMID_MULTITOKEN_MAX` | leave at the default (8) | llama.cpp SYCL (local patch 0004) | largest batch that takes the fused per-token MoE path; 512 made ~400-token prompt turns slower | [experiment](../models/qwen3.8-flash-next/experiments/2026-09-25-mmid-multitoken-prompt-turns.md) |
| `pcie_aspm` policy (PCIe Active State Power Management) | `powersave`, set at boot by [`tools/bmg-aspm-l1.sh`](../tools/bmg-aspm-l1.sh) after limiting L1 to the B70 links; not the `pcie_aspm.policy=` kernel parameter | kernel | idle board power 44–49 W to 4–6 W per card, speed unchanged | [finding](../findings/bmg-aspm-l1-idle-power.md) |

## Known issues

Link to findings and upstream issues. Remove entries once fixed in a version we run, and note which version fixed them.

- **`xpu-smi` run as root wakes idle cards** and reports 88–94 W per B70 instead of 45–48 W. Use the `xe` hwmon energy counters. Seen with xpu-smi 2.1.0 and compute-runtime 26.35.39758.11. [finding](../findings/xpu-smi-as-root-wakes-gpus.md)
- **`xe` spills VRAM overcommit into unswappable host RAM**, and the dma-buf export path (kernel 6.18 and later) mirrors exported VRAM buffers into host RAM. vLLM TP=4 without the sleep-mode allocator reached 92–104 GiB of driver-held RAM on 7.0.0-34-generic and crashed the host. Upstream: [compute-runtime #953](https://github.com/intel/compute-runtime/issues/953), [#980](https://github.com/intel/compute-runtime/issues/980). [finding](../findings/xe-vram-overcommit-spills-into-host-ram.md)
- **A PCIe switch uplink trained at Gen1** on one boot, cutting host bandwidth to 3.6 GB/s for four cards. Hardware, not a driver version. [finding](../findings/pcie-switch-uplink-can-train-at-gen1.md)
- **`xpu-smi discovery -d N` segfaults** with xpu-smi 2.1.0 and compute-runtime 26.35 on quad-b70-5800x-pex88096; `xpu-smi config -d N` works. Read clocks and power from sysfs.
