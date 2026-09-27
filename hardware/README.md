# Hardware

GPU specifications and the host machines we run them in. One file per host in [`hosts/`](hosts/), started from [`templates/host.md`](../templates/host.md).

## GPUs

| GPU | Architecture | VRAM | Memory bandwidth | Power (default / max) | Idle board power (ASPM off / L1) | AOT target |
|---|---|---|---|---|---|---|
| Intel Arc Pro B70 | Battlemage (Xe2), BMG-G31 | 32 GB (32,656 MiB physical, about 30.3 GiB usable) | 608 GB/s (vendor figure, not measured here) | 150 W / 230 W | 44–49 W / 4–6 W, model loaded ([finding](../findings/bmg-aspm-l1-idle-power.md)) | `bmg-g31-a0` |

Add other Arc Pro and Arc cards here as they are tested.

**Why bandwidth is the number to watch.** Single-stream decode reads every active weight once per step, so the ceiling is roughly bandwidth divided by the bytes of weights read per token. A 27B dense model at INT4 is about 15 GB, which caps one B70 near 40 tok/s without speculative decoding.

## Hosts

| Host | CPU | GPUs | PCIe | Notes |
|---|---|---|---|---|
| [dual-b70-5800x](hosts/dual-b70-5800x.md) | Ryzen 7 5800X | 2x Arc Pro B70 | 4.0 | primary box until 2026-09-23; P2P verified; upgraded to quad-b70-5800x-pex88096 |
| [quad-b70-5800x-pex88096](hosts/quad-b70-5800x-pex88096.md) | Ryzen 7 5800X | 4x Arc Pro B70 | 4.0; all cards behind one Broadcom PEX88096 switch on a shared x16 uplink | primary box since 2026-09-23; P2P 28.5 GB/s any pair; ASPM L1 idle 4–6 W per card |

## Things to record for every host

- PCIe generation and link width per GPU slot, and whether both GPUs share a root complex. This decides whether peer-to-peer works.
- Behind a PCIe switch, the uplink's speed and width on every boot. A switch uplink can train at Gen1 without any error ([finding](../findings/pcie-switch-uplink-can-train-at-gen1.md)).
- Resizable BAR and Above 4G decoding state.
- IOMMU and ASPM settings: the policy, and which links have L1 on.
- Host submission latency class. Small allreduces cost about 13 us on a Zen 5 EPYC and 48 to 51 us on a Zen 3 Threadripper PRO, which is several milliseconds per token at 130 collectives per step.
- Sustained power and temperature per card under a long decode run.
- Idle power per card with a model loaded, read from the `xe` hwmon energy counters, not from `xpu-smi` run as root ([finding](../findings/xpu-smi-as-root-wakes-gpus.md)).
- Host RAM held by the GPU driver during a multi-GPU load: `MemTotal − MemAvailable − AnonPages` ([finding](../findings/xe-vram-overcommit-spills-into-host-ram.md)).

## Idle power on the B70

An idle B70 with a model loaded draws about 44–49 W of board power when PCIe Active State Power Management (ASPM) is off, and 4–6 W with plain ASPM L1 on its links ([finding](../findings/bmg-aspm-l1-idle-power.md)).

- **ASPM is the switch that matters.** With ASPM off, the cards sit in package state G2 with the link in L0. With L1 on, they reach G8 and link L1.
- **Check who controls ASPM.** On the ASRock X570 board tested, the firmware leaves ASPM to the OS (the FADT, Fixed ACPI Description Table, has the "ASPM unsupported" bit clear, and `_OSC` grants control), but enables it on no link. The kernel's `default` policy keeps that, so ASPM stays off until you change the policy.
- **Enable it on the card links only.** [`tools/bmg-aspm-l1.sh`](../tools/bmg-aspm-l1.sh) turns L1 on for the B70 links (device IDs 8086:e2ff, e223, e2f7), keeps it off elsewhere, and sets the policy to `powersave`. Keep ASPM off on switch uplinks and avoid `powersupersave`.
- **Decode and prefill speed were unchanged** with L1 on (llama.cpp layer split, four cards). Tensor-parallel vLLM was not re-measured.
- **A lit console costs power.** The card driving a console idled at 11 W with L1 on, and 5 W with the console blanked.
- **Root `xpu-smi` misreports idle power.** It wakes the cards and reads 88–94 W per card ([finding](../findings/xpu-smi-as-root-wakes-gpus.md)).
