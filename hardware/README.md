# Hardware

GPU specifications and the host machines we run them in. One file per host in [`hosts/`](hosts/), started from [`templates/host.md`](../templates/host.md).

## GPUs

| GPU | Architecture | VRAM | Memory bandwidth | Power (default / max) | AOT target |
|---|---|---|---|---|---|
| Intel Arc Pro B70 | Battlemage (Xe2), BMG-G31 | 32 GB (32,656 MiB physical, about 30.3 GiB usable) | 608 GB/s | 150 W / 230 W | `bmg-g31-a0` |

Add other Arc Pro and Arc cards here as they are tested.

**Why bandwidth is the number to watch.** Single-stream decode reads every active weight once per step, so the ceiling is roughly bandwidth divided by the bytes of weights read per token. A 27B dense model at INT4 is about 15 GB, which caps one B70 near 40 tok/s without speculative decoding.

## Hosts

| Host | CPU | GPUs | PCIe | Notes |
|---|---|---|---|---|
| [dual-b70-5800x](hosts/dual-b70-5800x.md) | Ryzen 7 5800X | 2x Arc Pro B70 | 4.0 | primary box; P2P verified |

## Things to record for every host

- PCIe generation and link width per GPU slot, and whether both GPUs share a root complex. This decides whether peer-to-peer works.
- Resizable BAR and Above 4G decoding state.
- IOMMU and ASPM settings.
- Host submission latency class. Small allreduces cost about 13 us on a Zen 5 EPYC and 48 to 51 us on a Zen 3 Threadripper PRO, which is several milliseconds per token at 130 collectives per step.
- Sustained power and temperature per card under a long decode run.
