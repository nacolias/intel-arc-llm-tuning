# <hostname>

Name the host by its hardware, for example `dual-b70-5800x`, never by its real hostname. Do not record IP addresses, domains, usernames, MAC addresses or serial numbers. See [`.agents/privacy.md`](../.agents/privacy.md).

## Hardware

| Component | Detail |
|---|---|
| CPU | |
| Memory | |
| Motherboard / chipset | |
| GPUs | |
| GPU slots and PCIe link (gen, width) | |
| Root complex / switch topology | |
| Storage for models | |
| Network | |
| PSU | |

## Firmware and OS

| Item | Value |
|---|---|
| BIOS version | |
| Resizable BAR / Above 4G decoding | |
| IOMMU | |
| ASPM | |
| OS and kernel | |
| `xe` module parameters | |
| GuC firmware | |

## Measured characteristics

| Probe | Result | Date |
|---|---|---|
| `zeDeviceCanAccessPeer` both directions | | |
| Host submission latency (async launch, launch+sync) | | |
| Two-card allreduce latency at 2 rows | | |
| Large allreduce bandwidth | | |
| Sustained power per card under decode | | |

## Notes

Cooling, known quirks, recovery procedures.
