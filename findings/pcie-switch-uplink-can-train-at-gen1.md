# A PCIe switch uplink can train at Gen1 and cut host bandwidth eightfold

| | |
|---|---|
| **Date** | 2026-09-23 (found), 2026-09-24 (fixed) |
| **Confidence** | verified-here (single incident) |
| **Applies to** | Arc Pro B70 cards behind a Broadcom PEX88096 switch board reached over a SlimSAS host adapter from an AMD X570 CPU root port; Linux 7.0.0-34-generic; host [quad-b70-5800x-pex88096](../hardware/hosts/quad-b70-5800x-pex88096.md) |
| **Area** | host |

## Finding

On one boot, the link from the CPU root port to the PEX88096 switch trained at 2.5 GT/s x16 (PCIe Gen1), although both ends support and target 16 GT/s. All four cards then shared about 3.6 GB/s of host bandwidth. Peer-to-peer (P2P) traffic inside the switch was unaffected at 28.5 GB/s per pair. A runtime retrain did not help. A cold start with the SlimSAS cables reseated brought the link back to Gen4 x16. Check the uplink speed after every boot.

## Evidence

### Link state on the bad boot

- Root port and switch upstream port: `LnkSta` 2.5 GT/s x16 "(downgraded)". `LnkCap` and the `LnkCtl2` target were 16 GT/s on both, so this was not a speed cap in the BIOS or the switch EEPROM.
- Equalization failed at both higher speeds. `LnkSta2` showed `EqualizationComplete+` with `Phase1- Phase2- Phase3-`, `Phy16Sta` showed `EquComplete-`, and the switch upstream port showed `LinkEqualizationRequest+`. Every healthy Gen4 link on the host shows `Phase1+ Phase2+ Phase3+`.
- The kernel log said the switch was "limited by 2.5 GT/s PCIe x16 link at" the root port.
- The links from the switch to the four cards stayed at 16 GT/s x16 on every boot.
- The CPU root port has no AER (Advanced Error Reporting) capability, so link errors on this hop are not logged. The switch upstream port's AER counters were 0.

### Boot history with the switch board

| Boots | Uplink |
|---|---|
| First 8 boots, before a hardware rework | Gen4 x16 every time, warm and cold |
| First boot after the rework | Gen4 x2 (14 lanes lost) |
| Next boot | Gen4 x16 |
| Boot after a kernel panic and an 8-minute hang (2026-09-23) | Gen1 x16 |
| 2026-09-24: cold start after reseating the SlimSAS cables | Gen4 x16, equalization complete on both ends |

- Kernel, BIOS (P5.80) and kernel command line were identical on every boot.
- Before the switch board, a B70 plugged directly into the same CPU slot trained Gen4 x16 on all 19 recorded boots.
- Both bad boots came right after the rework, and both were cold starts. Both warm reboots with the switch trained Gen4 x16. So "a cold boot fixes it" is not supported on its own.
- A 10 GbE NIC on the chipset was removed in the same session as the reseat. Which change fixed the link cannot be separated.

### Runtime retrain failed

With the GPU services stopped, the kernel's PCIe bandwidth controller (the `cooling_device` for the root port) was asked for Gen4, then Gen3, then Gen4. Each write returned `EAGAIN`. The link retrained but came back at 2.5 GT/s x16 each time. All four GPUs stayed healthy. Gen3 failing as well points to the physical path, not a setting.

### Bandwidth, Gen1 versus Gen4

1 GiB copies, 5 repetitions, `torch` 2.13.0+xpu in the `vllm/vllm-openai-xpu:v0.28.0` image:

| Path | Uplink at Gen1 x16 (2026-09-23) | Uplink at Gen4 x16 (2026-09-24) |
|---|---|---|
| Host → GPU, pinned, one card | 3.58 GB/s | 28.36 GB/s |
| GPU → host, pinned, one card | 3.63 GB/s | 28.51 GB/s |
| Host → GPU, pageable, one card | 3.06–3.23 GB/s | not rerun |
| Host → GPU, several cards at once | 3.59 GB/s total (4 cards) | 28.36 GB/s total (2 cards) |
| GPU → GPU, any pair | 28.51–28.53 GB/s | 28.51 GB/s (GPU2 → GPU3) |

Gen1 x16 has about 4 GB/s of theoretical bandwidth, so 3.6 GB/s is the link ceiling. P2P did not change because it never crosses the uplink ([finding](bmg-pcie-p2p-works.md)).

## Impact

- **Everything that crosses the host link is about 8× slower:** model loads, vLLM sleep and wake, CPU offload, and host-staged collectives.
- **Steady-state tensor-parallel decode barely changes.** The Qwen3.8-27B TP=4 benchmark (129.1 tok/s single-stream) ran on the Gen1 boot ([benchmark](../models/qwen3.8-27b/benchmarks/2026-09-23-tp4-quad-b70.md)), because TP traffic stays inside the switch.
- **After the fix, cold loads become disk-bound.** Models live on a Gen3 x4 NVMe behind the chipset at about 3.4 GB/s.

## What to do

Check the uplink after every boot. Replace `<root-port>` with the PCI address of the CPU port that feeds the switch:

```bash
p=/sys/bus/pci/devices/<root-port>   # e.g. 0000:00:03.1
s=$(cat $p/current_link_speed); w=$(cat $p/current_link_width)
[ "$s" = "16.0 GT/s PCIe" ] && [ "$w" = "16" ] || logger -p user.warning "PCIe switch uplink degraded: $s x$w"
```

If it comes up degraded:

1. Stop GPU services and power off fully, including the switch board's own PSU.
2. Reseat and latch the SlimSAS cables at both ends, support the board against cable pull, and check its auxiliary power.
3. Cold-start and check again. Log speed and width on the next several warm and cold boots.
4. If it keeps failing: set the slot to Gen4 explicitly in the BIOS (the menu name on this board is unverified), enable the AER option if the board has one, and try a redriver or retimer host adapter.

Do not treat a Gen3 cap as a safe fallback. The 8 GT/s equalization also failed on the bad boot, and no boot has run this uplink at Gen3.

Before benchmarking anything that moves data between host and GPU, record the uplink speed with the result.

## Still open

- **Root cause.** The physical path (seating, cable, signal integrity) is the most likely cause, but that is an inference. Power sequencing between the host PSU and the switch board's own PSU is also possible.
- **Which fix worked:** the reseat or the NIC removal.
- **Single incident.** One Gen1 boot and one x2 boot in 12 boots with the switch. Re-check after any hardware change.
- **Signal margins** were not measured. `pcilmr --margin` on the root port needs 16 GT/s and a quiet link.
