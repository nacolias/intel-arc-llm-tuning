# Pinning the GPU clock floor speeds up llama.cpp layer-split decode by 8-23%

| | |
|---|---|
| **Date** | 2026-09-24 |
| **Confidence** | verified-here |
| **Applies to** | Arc Pro B70 on the Linux `xe` driver (kernel 7.0.0-34, GuC 70.58.0); llama.cpp SYCL with `-sm layer` across 4 cards; host [quad-b70-5800x-pex88096](../hardware/hosts/quad-b70-5800x-pex88096.md) |
| **Area** | driver / power |

## Finding

With `-sm layer`, each card computes in short bursts and then waits while the other cards run. The likely mechanism is that the `xe` frequency governor clocks each card down between bursts; this was not observed directly. Pinning each card's minimum compute clock (`tile0/gt0/freq0/min_freq`) to its hardware maximum `rp0` (2800 MHz on all four B70s) while the server runs gave +16% decode at short context, +23% at ~10K and +8% at ~39K. At true idle the GTs (the GPU's graphics/compute tiles) still enter the C6 sleep state with `act_freq` 0, so the floor costs about 0 W. The catch: anything that wakes the GPUs while the floor is set runs them at full clock.

## Evidence

Qwen3.8-Flash-Next, llama.cpp PR #28243 build with the fused MUL_MAT_ID patches, MTP on, single stream, 2026-09-24 ([experiment](../models/qwen3.8-flash-next/experiments/2026-09-24-gpu-clock-floor.md)):

| Context | Driver default (`min_freq` = 400 MHz) | Floor at 2800 MHz | Change (range midpoints) |
|---|---|---|---|
| Short | 42-44 tok/s | 50 tok/s | +16% |
| ~10K | 52-54 tok/s | 64-66 tok/s | +23% |
| ~39K | 44-45 tok/s | 47-49 tok/s | +8% |

Idle cost, measured 2026-09-25 from each card's own `xe` hwmon energy counters (model loaded, no requests, floor set):

| Condition | Board power per card |
|---|---|
| Quiet, 20 s | 47.9 / 46.7 / 48.4 / 45.4 W; every GT in `gt-c6`, `act_freq` 0 |
| `sudo xpu-smi` polling in a loop, 20 s | 88.8 / 81.3 / 80.4 / 70.4 W |

The second row is the interaction: root `xpu-smi` wakes the GTs, and with the floor set they run at 2800 MHz. See [xpu-smi as root wakes the GPUs](xpu-smi-as-root-wakes-gpus.md). The PCIe ASPM (Active State Power Management) results in [bmg-aspm-l1-idle-power](bmg-aspm-l1-idle-power.md) were measured with the floor set.

## Impact

About +8-23% single-stream decode for layer-split llama.cpp, for no idle-power cost as long as nothing polls the GPUs as root.

## What to do

Set the floor only while the layer-split server runs, and restore the driver default (`rpn`, 400 MHz) when it stops. [`tools/gpu-clock-floor.sh`](../tools/gpu-clock-floor.sh) does both; call it from the service unit:

```ini
[Service]
ExecStartPre=+/usr/local/sbin/gpu-clock-floor.sh pin
ExecStopPost=+/usr/local/sbin/gpu-clock-floor.sh restore
```

The sysfs files, per card:

```bash
cat /sys/bus/pci/devices/<bdf>/tile0/gt0/freq0/rp0_freq    # 2800 on B70
echo 2800 | sudo tee /sys/bus/pci/devices/<bdf>/tile0/gt0/freq0/min_freq
```

Do not monitor the cards with root `xpu-smi` while the floor is set; read hwmon or run `xpu-smi` without sudo.

## Still open

- Not measured with vLLM tensor parallel, where all cards work at once and should downclock less.
- Measured in one session with about ±5% run-to-run noise. A 250 ms `act_freq` sampler on the unpinned build read 2550-2800 MHz during decode (see the [experiment](../models/qwen3.8-flash-next/experiments/2026-09-24-gpu-clock-floor.md)); it cannot see sub-millisecond gaps, so the downclocking is inferred, not shown.
- Re-check after a GuC or kernel update, which can change the governor.
