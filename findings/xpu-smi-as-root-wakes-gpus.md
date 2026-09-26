# xpu-smi run as root wakes idle B70s and reports twice their idle power

| | |
|---|---|
| **Date** | 2026-09-25 |
| **Confidence** | verified-here (the root `xpu-smi` effect); upstream-documented (any `xe` ioctl resumes a suspended card) |
| **Applies to** | Arc Pro B70; xpu-smi 2.1.0; compute-runtime 26.35.39758.11; Linux 7.0.0-34-generic `xe`; host [quad-b70-5800x-pex88096](../hardware/hosts/quad-b70-5800x-pex88096.md) |
| **Area** | power / runtime |

## Finding

Run as root, `xpu-smi summary` and `xpu-smi stats` report 88–94 W per idle B70. The cards' own energy counters show 45–48 W when nothing polls them, and 70–89 W while root `xpu-smi` polls. So `xpu-smi` measures the load it creates. Run without root, it reads 46 W and does not wake the cards.

More generally, any process that opens the GPU through Level Zero, sysman or an `xe` ioctl resumes a runtime-suspended card. Use the `xe` hwmon energy counters for idle power, and file or sysfs checks for "is the GPU in use".

## Evidence

Big-model mode (Qwen3.8-Flash-Next loaded on four cards, no requests), GPU clock floor at 2800 MHz, 2026-09-25. Board ("card") power from the `xe` hwmon `energy1_input` counter over 20 s:

| Card | Quiet 20 s just before | While root `xpu-smi` polled in a loop, 20 s |
|---|---|---|
| GPU0 | 47.9 W | 88.8 W |
| GPU1 | 46.7 W | 81.3 W |
| GPU2 | 48.4 W | 80.4 W |
| GPU3 | 45.4 W | 70.4 W |

- `sudo xpu-smi` (summary and stats) printed 88–94 W per card during the same period.
- `xpu-smi stats` without root read 46 W for GPU0.
- Before and after, every GT sat in `gt-c6` with `act_freq 0`. Root `xpu-smi` woke them, and with the 2800 MHz clock floor ([finding](gpu-clock-floor-speeds-layer-split.md)) they ran at full clock. All cards were back to about 46 W within 6 s after polling stopped.
- Each root `xpu-smi` run logs the same 8-line IGSC "Cannot get version for the partition" block in the system journal. That helps tell afterwards whether someone ran it as root.
- Why root and not non-root behave differently was not traced.

Kernel side (upstream-documented, not measured here): `xe_drm_ioctl` takes a runtime-PM reference that resumes the device ([xe_device.c, v7.0](https://github.com/torvalds/linux/blob/v7.0/drivers/gpu/drm/xe/xe_device.c)). So any `xe` ioctl, including a non-root `xpu-smi`, a Level Zero init or a sysman query, resumes a card that is in D3hot. A monitoring loop that calls `xpu-smi` every 30 s probably keeps idle cards from staying suspended (UNVERIFIED).

## Impact

| | Reading per idle card |
|---|---|
| Root `xpu-smi` | 88–94 W |
| hwmon while root `xpu-smi` polls | 70–89 W |
| hwmon, quiet, ASPM off | 45–48 W |
| hwmon, quiet, ASPM L1 on card links ([finding](bmg-aspm-l1-idle-power.md)) | 4–6 W (11 W with a lit console) |

- A root `xpu-smi` reading roughly doubles the apparent idle power, and hides what ASPM or runtime suspend would save.
- A monitor that runs as root costs about 40 W per card for as long as it polls.
- `intel_gpu_top` and `gputop` under `sudo` may do the same (UNVERIFIED).

## What to do

- Measure idle power from the hwmon energy counters. `energy1_input` is board energy and `energy2_input` is package energy, in microjoules. The files are world-readable, so no `sudo` is needed:

  ```bash
  # board power of one card over 60 s; <bdf> is the GPU's PCI address, e.g. 0000:14:00.0
  e() { cat /sys/bus/pci/devices/<bdf>/hwmon/hwmon*/energy1_input; }
  a=$(e); sleep 60; b=$(e); echo "$(( (b - a) / 60000000 )) W"
  ```

  [`tools/bmg-aspm-l1.sh status`](../tools/bmg-aspm-l1.sh) prints the same for every B70 over 10 s.
- Do not run `xpu-smi` as root in monitoring loops or while measuring idle power.
- For "is anyone using the GPU", check `fuser /dev/dri/renderD*` or read `/sys/bus/pci/devices/<bdf>/power/runtime_status`. Neither goes through the driver's ioctl path, so neither should wake the card (reasoned, not measured).
- Stop all polling before a D3hot or idle measurement: `xpu-smi`, health checks, hwmon reads by other tools, and `lspci -vvv`. Each of these can wake a card or the bridges above it (UNVERIFIED for the hwmon read).

## Still open

- **Counter behaviour across suspend.** `xe_hwmon` reads a 32-bit hardware counter on each file read and accumulates the difference ([xe_hwmon.c, v7.0](https://github.com/torvalds/linux/blob/v7.0/drivers/gpu/drm/xe/xe_hwmon.c)). The raw counter wraps after about 262 kJ, which is about 90 minutes at 47 W, so read it more often than that. Whether it pauses or resets while a card is in D3hot is unknown. Cross-check D3hot numbers with a wall meter. Reading the counter may itself wake a suspended card (UNVERIFIED).
- **Why root differs.** A root-only telemetry path in xpu-smi or sysman is the likely cause; the code was not read.
- **Re-check after an xpu-smi or compute-runtime upgrade.**
