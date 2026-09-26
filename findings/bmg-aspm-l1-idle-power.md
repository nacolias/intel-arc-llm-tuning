# ASPM L1 on the B70 links cuts idle power from about 46 W to 5 W per card

| | |
|---|---|
| **Date** | 2026-09-25 |
| **Confidence** | verified-here |
| **Applies to** | Arc Pro B70 (BMG-G31) behind a Broadcom PEX88096 switch on an AMD X570 root port; Linux 7.0.0-34-generic `xe`; card firmware GSC 31.1058 / OPROM 23.1065 (graphics security controller firmware / option ROM); host [quad-b70-5800x-pex88096](../hardware/hosts/quad-b70-5800x-pex88096.md) |
| **Area** | power / host |

## Finding

With PCIe Active State Power Management (ASPM) off, an idle B70 with a model loaded draws 44–49 W of board power. Turning on plain ASPM L1 on the card links only cuts that to 4–6 W per card, or 11 W for the card that drives a lit console. Four cards went from 185 W to 25–27 W. Decode and prefill speed did not change, and no new PCIe errors appeared.

On this board the operating system controls ASPM, so no BIOS change is needed. Only plain L1 is available through the PEX88096; there are no L1 substates. The SlimSAS uplink to the switch stays off.

## Evidence

### Starting state (read-only, 2026-09-25)

- `LnkCtl: ASPM Disabled` on every link from the CPU root port down to each GPU. The kernel policy was `[default]`, which keeps the firmware setting, and the firmware enables ASPM nowhere.
- The firmware leaves ASPM to the OS. The FADT (Fixed ACPI Description Table) "ASPM unsupported" bit is clear, `_OSC` hands PCIe control to Linux, and the kernel logged no "FADT indicates ASPM is unsupported" line. `pcie_aspm=force` is not needed.
- The ASRock X570 Steel Legend manual documents no ASPM option ([manual](https://download.asrock.com/Manual/X570%20Steel%20Legend.pdf)). A hidden AMD CBS option was not checked.
- All GTs were in `gt-c6` with `act_freq 0`. The xe debugfs counters (`dgfx_pkg_residencies`, `dgfx_pcie_link_residencies`) showed the cards in package state G2 with the link in L0. G8 and link L1 appeared only while a card was runtime-suspended.

### What the links support

| Link | ASPM capability | L1 exit latency | L1 substates |
|---|---|---|---|
| CPU root port → PEX88096 upstream (SlimSAS uplink) | L1 | under 64 µs | root port L1.1 only; PEX side none |
| PEX88096 downstream → card upstream port (8086:e2ff) | L0s, L1 | under 32 µs | PEX side none, so none on this link |
| Inside the card: e2f0/e2f1 → GPU (8086:e223) and audio (8086:e2f7) | L1 | under 1 µs | n/a |

All links run with `CommClk-` (separate reference clocks). The card upstream port has L1.1+ and L1.2+, but its partner is a PEX port without them. The card link stays at 16 GT/s x16 when idle; there is no dynamic link downshift.

### Recipe that was run

1. L0s and clock PM off on every link.
2. L1 off on every link except the four PEX → card links (downstream device 8086:e2ff) and the links inside each card (GPU 8086:e223 and audio 8086:e2f7 under bridges 8086:e2f0/e2f1).
3. Policy `powersave`.

Order matters. In `drivers/pci/pcie/aspm.c` (v7.0), the per-link sysfs files only change a disable mask and then re-apply the policy's target. The `default` policy's target is the firmware setting, so `echo 1 > link/l1_aspm` alone changes nothing here. Writing the policy re-applies every link. See [aspm.c v7.0](https://github.com/torvalds/linux/blob/v7.0/drivers/pci/pcie/aspm.c).

After the write, `LnkCtl` read "ASPM L1 Enabled" on a switch downstream port and its card upstream port, and "ASPM Disabled" on the SlimSAS uplink. L1 was on for 12 links: 4 × e2ff, 4 × e223, 4 × e2f7.

### Idle power

Board ("card") and package power per card from the xe hwmon energy counters, big-model mode (Qwen3.8-Flash-Next loaded on all four cards, no requests), 60 s windows, runtime trial on 2026-09-25:

| Card | Before: card / package | After: card / package |
|---|---|---|
| GPU0 | 47 W / 26 W | 5 W / 2 W |
| GPU1 | 46 W / 26 W | 6 W / 2 W |
| GPU2 (drives the console) | 47 W / 27 W | 11 W / 5 W |
| GPU3 | 45 W / 25 W | 5 W / 2 W |
| **Total card power** | **185 W** | **27 W** |

Boot-service test, same day: with the service reverted (policy `default`) the cards drew 45 / 44 / 46 / 44 W, 179 W in total. With the service on they drew 4 / 6 / 11 / 4 W, 25 W in total, over both 60 s and 300 s windows. A later read-only `status` run of [`tools/bmg-aspm-l1.sh`](../tools/bmg-aspm-l1.sh) showed 4 / 6 / 11 / 5 W.

GPU2 drives a 2560×1440 at 60 Hz framebuffer console. Blanking it dropped that card from 11 W to 5 W:

```bash
sudo sh -c 'TERM=linux setterm --blank force </dev/tty1 >/dev/tty1'
```

Redirecting only stdout fails with "Inappropriate ioctl for device".

The 2800 MHz GPU clock floor used for llama.cpp ([finding](gpu-clock-floor-speeds-layer-split.md)) costs about 0 W at true idle. The GTs still enter C6 with `min_freq` at 2800 MHz.

### Speed is unchanged

llama.cpp SYCL, Qwen3.8-Flash-Next, 4 cards, layer split, MTP draft length 3, same server session, [`tools/fnbench.py`](../tools/fnbench.py) before and after the change:

| Test | Before | After |
|---|---|---|
| Decode, short context (median of 5) | 49.2 tok/s | 50.5 tok/s |
| Decode after a 9,749-token prompt | 67.2 tok/s | 66.4 tok/s |
| Decode after a 39,119-token prompt | 55.2 tok/s | 53.7 tok/s |
| Prefill, 9,749 tokens | 644.7 tok/s | 641.2 tok/s |
| Prefill, 39,119 tokens | 573.7 tok/s | 567.6 tok/s |
| MTP drafts accepted at 9,749 / 39,119 | 186/206, 180/225 | 186/206, 178/230 |

Both runs gave the correct answer to the sanity question. The short-context runs overlap (min–max 46.9–52.0 tok/s before, 47.6–51.6 after). The long-prompt rows are single samples, and their differences are within the spread seen between runs on this model. See [Qwen3.8-Flash-Next](../models/qwen3.8-flash-next/) for the build and launch flags.

### No new errors

- No new `CESta`/`UESta` bits in `lspci -vvv` after the change. `AdvNonFatalErr+` was already set on two devices before it.
- No AER, PCIe or xe errors in the kernel log after the change.
- Caveat: the CPU root port has no AER (Advanced Error Reporting) capability, so link errors under it are not reported by the kernel's AER service. Watching the raw error-status fields is the only check.

### Residency moved to G8 and L1

Deltas of the xe debugfs counters between a snapshot before the change and one after the 60 s idle window (about 5 minutes apart, including the before-benchmark):

| Card | Package G8 share | Link L1 share |
|---|---|---|
| GPU0 | 37.8% | 31.7% |
| GPU1 | 38.0% | 32.5% |
| GPU2 | 35.9% | 32.6% |
| GPU3 | 39.9% | 31.3% |

GPU2 had never entered G8 or link L1 before the change. The counter units are not documented, so only the shares are shown.

### External comparison (not measured here)

- PCGH measured an Intel reference B70 on Windows at 49–50 W idle without ASPM, and 8 / 11 / 15 W at FHD / WQHD / UHD 60 Hz with ASPM ([PCGH, 2026-05-12](https://www.pcgameshardware.de/Arc-Pro-B70-Grafikkarte-284242/Tests/Gaming-benchmarks-Arc-Pro-B70-vs-B580-1533956/5/)). That matches our 44–49 W before and the 11 W of the console card after.
- Intel's support article says Arc low-power states below G2 require ASPM L1 ([Intel 000092564](https://www.intel.com/content/www/us/en/support/articles/000092564/graphics.html)). It is written for Windows and predates the B70.

## Impact

| | ASPM off | L1 on card links |
|---|---|---|
| Four cards, model loaded, idle | 179–185 W | 25–27 W |
| One headless card | 44–47 W | 4–6 W |
| Card driving a lit console | 46–47 W | 11 W (5 W blanked) |
| Decode and prefill | baseline | unchanged |

That is about 155–160 W less DC power at idle for four cards. Wall power and the switch board's own power supply were not measured.

## What to do

### Install the tool

[`tools/bmg-aspm-l1.sh`](../tools/bmg-aspm-l1.sh) applies the recipe by device ID, checks the result and reverts if anything does not match. [`tools/bmg-aspm-l1.service`](../tools/bmg-aspm-l1.service) runs it at boot. Try it live first:

```bash
sudo install -m 0755 tools/bmg-aspm-l1.sh /usr/local/sbin/bmg-aspm-l1.sh
sudo /usr/local/sbin/bmg-aspm-l1.sh apply    # L1 on the card links, policy powersave, self-check
sudo /usr/local/sbin/bmg-aspm-l1.sh status   # links with L1 on, board watts per GPU over 10 s
# run your benchmark; to undo: sudo /usr/local/sbin/bmg-aspm-l1.sh revert
```

Then make it permanent:

```bash
sudo install -m 0644 tools/bmg-aspm-l1.service /etc/systemd/system/bmg-aspm-l1.service
sudo systemctl daemon-reload
sudo systemctl enable --now bmg-aspm-l1.service
# undo: sudo systemctl disable --now bmg-aspm-l1.service   (ExecStop runs revert)
```

Check the result after the first boot:

```bash
grep . /sys/bus/pci/devices/*/link/l1_aspm | grep ':1$'   # only the card links (e2ff, e223, e2f7)
cat /sys/module/pcie_aspm/parameters/policy                 # [powersave]
sudo lspci -vvv -s <switch-uplink-bdf> | grep LnkCtl:       # ASPM Disabled
```

### Rules

- **Keep ASPM off on the SlimSAS uplink.** It has trained at Gen1 before ([finding](pcie-switch-uplink-can-train-at-gen1.md)); do not add risk to it.
- **Do not use the kernel parameter `pcie_aspm.policy=powersave`.** It enables ASPM on every capable link, including the uplink.
- **Do not use `powersupersave`.** L1 substates are not available through the PEX88096 anyway. A kernel quirk patch reports a B70 on an ASRock X570 board failing to come back with L1.1 enabled, while `powersave` works ([mailing-list thread](https://ratatoskr.run/intel-xe/2026/05/8035933/t), unmerged, UNVERIFIED here).
- **Prefer the boot service to toggling the policy on a busy system.** One community lab reports a kernel panic when changing the ASPM policy at runtime on Ubuntu kernel 7.0.0-28 with ASRock B70s ([b70-optimization-lab](https://github.com/steveseguin/b70-optimization-lab), UNVERIFIED here). Our runtime change on 7.0.0-34-generic, with a model loaded, was clean.
- **Do not use `pcie_aspm=force`.** It is not needed when the FADT bit is clear, and the kernel documentation warns it may lock the system up.
- **Check after each kernel upgrade.** An Intel RFC from 2026-05 would default systems with a 2025-or-later BIOS to `powersupersave` ([RFC](https://ratatoskr.run/linux-pci/2026/05/8997004/t), unmerged as of 2026-09-25). This board's BIOS is from 2026. The service runs after PCI setup, so such a kernel would enable ASPM on the uplink until the service runs. Adding `pcie_aspm.policy=default` to the kernel command line as well is a possible guard (UNVERIFIED, not applied here).
- **Blank consoles you do not look at.** A lit console keeps its card at 11 W instead of 5 W.

### Measure idle power with the hwmon energy counters

Read the xe hwmon energy counters, not `xpu-smi` as root, which wakes the cards and reads about 90 W ([finding](xpu-smi-as-root-wakes-gpus.md)). `energy1_input` is board ("card") energy and `energy2_input` is package energy, both in microjoules. The files are world-readable, so no `sudo` is needed.

```bash
# board power of one card over 60 s; <bdf> is the GPU's PCI address, e.g. 0000:14:00.0
e() { cat /sys/bus/pci/devices/<bdf>/hwmon/hwmon*/energy1_input; }
a=$(e); sleep 60; b=$(e); echo "$(( (b - a) / 60000000 )) W"
```

Stop every GPU client that polls the cards before measuring. Record the package G-state and link L-state residencies from xe debugfs alongside the watts.

## Still open

- **Card firmware 31.1062 is untested.** Its release note on LVFS (the Linux Vendor Firmware Service) mentions better low-power entry when no display is connected. It needs fwupd 2.1.4 or newer. See [platform](../platform/README.md#card-firmware).
- **D3cold is impossible on this host.** The root port has no ACPI `_PR3` power resource, and nothing on the switch path supports hotplug. Cutting power to idle cards is not an option.
- **D3hot power is unmeasured.** Community B50/B60 reports put D3hot above what these cards now draw in D0 with L1 (UNVERIFIED), so letting idle cards runtime-suspend may not help further.
- **Long-term stability.** The trial and the boot service ran on 2026-09-25. Re-check the error-status fields after a week of use.
- **vLLM tensor parallelism was not re-measured.** Only llama.cpp layer split was benchmarked with L1 on. Up to 32 µs of L1 exit latency per link could matter for small collectives. Measure a TP=4 decode before relying on it there.
- **Wall power** at each PSU, including the switch board's own supply, is unmeasured.
