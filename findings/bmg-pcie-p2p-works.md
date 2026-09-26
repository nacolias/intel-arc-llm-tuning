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
