# PCIe peer-to-peer works on Battlemage

| | |
|---|---|
| **Date** | 2026-09 |
| **Confidence** | verified-here |
| **Applies to** | Arc Pro B70 pairs under one PCIe root complex; Linux `xe` driver with `allow_peer2peer` |
| **Area** | oneCCL / host |

## Finding

Community scripts and Intel defaults set `CCL_TOPO_P2P_ACCESS=0` on the assumption that Battlemage has no peer-to-peer. On a host where both B70s share a root complex, Level Zero reports peer access in both directions, and enabling it in oneCCL is faster.

## Evidence

Level Zero probe (`zeDeviceCanAccessPeer`) on host [dual-b70-5800x](../hardware/hosts/dual-b70-5800x.md):

```
dev0 -> dev1: getP2P r=0 flags=0x1 canAccess r=0 val=1
dev1 -> dev0: getP2P r=0 flags=0x1 canAccess r=0 val=1
```

The 6.18 `xe` driver sets `allow_peer2peer = true` and checks `pci_p2pdma_distance()` in `xe_dma_buf.c`, which is why placement under one root complex matters.

## Impact

| Measurement | P2P off | P2P on |
|---|---|---|
| 40 MB allreduce | 26.8 ms (1.57 GB/s) | 16.4 ms (2.56 GB/s) |
| 150K-token cold prefill (Qwen3.8-27B) | 213 s | about 183 s |
| Aggregate decode at c16 (Qwen3.8-27B) | about 356 tok/s | 421 to 442 tok/s |

## What to do

Probe peer access on every new host. If it returns 1 both ways, set `CCL_TOPO_P2P_ACCESS=1`.

## Still open

- 2.56 GB/s is far below PCIe 4.0 x8 practical bandwidth (about 13 GB/s). The gap is oneCCL algorithm and staging, not the link.
- compute-runtime master has IPC fixes for peer buffers that may improve stability.

## Update 2026-09-23: four cards behind a PCIe switch

P2P also works behind a Broadcom PEX88096 switch, at close to the Gen4 x16 line rate. On host [quad-b70-5800x-pex88096](../hardware/hosts/quad-b70-5800x-pex88096.md), device-to-device copies between any pair of the four B70s ran at 28.5 GB/s. The traffic is routed inside the switch.

| Measurement (1 GiB copies, `torch` 2.13.0+xpu) | Result |
|---|---|
| Any single pair, all 12 directions | 28.51–28.53 GB/s |
| Two disjoint pairs at once | 56.92 GB/s total (28.46 per pair) |
| Four-card ring, all flows at once | 47.29 GB/s total (11.82 per flow) |
| GPU2 → GPU3 after the host uplink was fixed (2026-09-24, shorter rerun) | 28.51 GB/s |

- The first three rows were measured while the switch's uplink to the CPU was stuck at Gen1 (3.6 GB/s host bandwidth; [finding](pcie-switch-uplink-can-train-at-gen1.md)). P2P at 28.5 GB/s therefore cannot have crossed the root complex.
- Level Zero reported P2P access for every pair. ACS (Access Control Services) is off on the switch downstream ports, and the IOMMU is off.
- Kernel `p2pdma` routes devices under a common switch by bus address when ACS redirect is off ([p2pdma.c](https://github.com/torvalds/linux/blob/master/drivers/pci/p2pdma.c)). If `pci_p2pdma_distance()` fails, `xe` quietly migrates the buffer to system memory instead ([xe_dma_buf.c](https://github.com/torvalds/linux/blob/master/drivers/gpu/drm/xe/xe_dma_buf.c)). A drop to about 3.6 GB/s together with rising host RAM would mean that happened. Re-check P2P bandwidth after kernel or driver updates.
- The bandwidth table is recorded on the [host page](../hardware/hosts/quad-b70-5800x-pex88096.md#measured-characteristics). Whether oneCCL collectives in vLLM TP=4 actually take this path was not measured separately.
