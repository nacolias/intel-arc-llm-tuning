# Findings

Lessons that hold beyond a single model. One file per finding, started from [`templates/finding.md`](../templates/finding.md).

| Finding | Area | Confidence | Date |
|---|---|---|---|
| [PCIe peer-to-peer works on Battlemage](bmg-pcie-p2p-works.md) | oneCCL / host | verified-here | 2026-09 |
| [Tensor-parallel XPU graphs need per-worker Level Zero affinity](tp-xpu-graphs-need-per-worker-affinity.md) | engine / Level Zero | verified-here | 2026-09 |
| [oneCCL topo algorithms cannot be captured in SYCL graphs](oneccl-topo-not-graph-capturable.md) | oneCCL | verified-here | 2026-09 |
| [Do not use a persistent SYCL kernel cache on Battlemage](no-persistent-sycl-cache-on-bmg.md) | runtime | reproduced-community | 2026-09 |
| [Prefix caching plus MTP can silently corrupt hybrid models](prefix-cache-mtp-corruption-hybrid.md) | engine | upstream-documented | 2026-09 |
