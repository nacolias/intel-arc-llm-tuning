# Findings

Lessons that hold beyond a single model. One file per finding, started from [`templates/finding.md`](../templates/finding.md).

| Finding | Area | Confidence | Date |
|---|---|---|---|
| [PCIe peer-to-peer works on Battlemage](bmg-pcie-p2p-works.md) | oneCCL / host | verified-here | 2026-09 |
| [Tensor-parallel XPU graphs need per-worker Level Zero affinity](tp-xpu-graphs-need-per-worker-affinity.md) | engine / Level Zero | verified-here | 2026-09 |
| [oneCCL topo algorithms cannot be captured in SYCL graphs](oneccl-topo-not-graph-capturable.md) | oneCCL | verified-here | 2026-09 |
| [Do not use a persistent SYCL kernel cache on Battlemage](no-persistent-sycl-cache-on-bmg.md) | runtime | reproduced-community | 2026-09 |
| [Prefix caching plus MTP can silently corrupt hybrid models](prefix-cache-mtp-corruption-hybrid.md) | engine | upstream-documented | 2026-09 |
| [A PCIe switch uplink can train at Gen1 and cut host bandwidth eightfold](pcie-switch-uplink-can-train-at-gen1.md) | host | verified-here | 2026-09-23 |
| [xe spills VRAM overcommit into host RAM that cannot be swapped](xe-vram-overcommit-spills-into-host-ram.md) | driver / host | verified-here | 2026-09-23 |
| [Pinning the GPU clock floor speeds up llama.cpp layer-split decode by 8-23%](gpu-clock-floor-speeds-layer-split.md) | driver / power | verified-here | 2026-09-24 |
| [llama.cpp SYCL graphs never engage with more than one GPU or with MoE](llama-cpp-sycl-graphs-off-with-multiple-gpus.md) | engine | upstream-documented, verified-here | 2026-09-24 |
| [llama.cpp SYCL MoE matmuls sync the host and run one GEMM per expert for batches over 8 tokens](llama-cpp-sycl-moe-prefill-host-sync.md) | engine | verified-here, upstream-documented | 2026-09-25 |
| [ASPM L1 on the B70 links cuts idle power from about 46 W to 5 W per card](bmg-aspm-l1-idle-power.md) | power / host | verified-here | 2026-09-25 |
| [xpu-smi run as root wakes idle B70s and reports twice their idle power](xpu-smi-as-root-wakes-gpus.md) | power / runtime | verified-here | 2026-09-25 |
