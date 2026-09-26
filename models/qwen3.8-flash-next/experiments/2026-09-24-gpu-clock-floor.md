# GPU clock floor at 2800 MHz while the model is served

| | |
|---|---|
| **Date** | 2026-09-24 |
| **Model / checkpoint** | huihui-ai abliterated `UD-Q4_K_XL` + unsloth MTP draft shared-Q8_0 |
| **Host** | [quad-b70-5800x-pex88096](../../../hardware/hosts/quad-b70-5800x-pex88096.md) |
| **Status** | kept |
| **Baseline** | build `20260924-7e5cb8f13` with default clocks ([fused MUL_MAT_ID](2026-09-24-fused-mul-mat-id-mtp-verify.md)) |
| **Result** | [raw/2026-09-24-speed-work.csv](../benchmarks/raw/2026-09-24-speed-work.csv) |
| **Related** | [finding: GPU clock floor speeds up layer split](../../../findings/gpu-clock-floor-speeds-layer-split.md), [finding: root xpu-smi wakes the GPUs](../../../findings/xpu-smi-as-root-wakes-gpus.md), [`tools/gpu-clock-floor.sh`](../../../tools/gpu-clock-floor.sh) |

## Hypothesis

Under `-sm layer` the four cards run one after another, so each card's compute engine is busy only 16-18% of the time during decode ([raw/2026-09-24-decode-profile.csv](../benchmarks/raw/2026-09-24-decode-profile.csv)). If the driver lowers the clock in the gaps, every kernel of the next burst starts slow. Pinning the minimum frequency to the hardware maximum should speed up each burst. Expected: a few percent to about 20%.

The downclocking itself was not observed directly. A 250 ms `act_freq` sampler on the unpinned, unpatched build read 2,550-2,800 MHz (average 2,765-2,800 MHz) during decode, and it cannot see sub-millisecond gaps.

## Change

For each card, before the server starts (as root), and back to the default when it stops:

```diff
- /sys/bus/pci/devices/<bdf>/tile0/gt0/freq0/min_freq = rpn_freq   (400 MHz, driver default)
+ /sys/bus/pci/devices/<bdf>/tile0/gt0/freq0/min_freq = rp0_freq   (2800 MHz on the B70)
```

[`tools/gpu-clock-floor.sh`](../../../tools/gpu-clock-floor.sh) `pin` / `restore` does this, for example from `ExecStartPre=+` and `ExecStopPost=+` in the service unit. No llama.cpp flag changed.

## Environment

| Component | Version / value |
|---|---|
| Engine and image (tag and digest) | llama.cpp, no container. Build `20260924-7e5cb8f13` (PR #28243 at `6fcaa16` + #28931 + 0002 + 0003) |
| Build flags | `GGML_SYCL=ON GGML_SYCL_TARGET=INTEL GGML_SYCL_F16=ON GGML_SYCL_DNN=ON GGML_SYCL_GRAPH=ON GGML_SYCL_SUPPORT_LEVEL_ZERO_API=ON GGML_NATIVE=ON`, Release, icx/icpx from oneAPI 2026.1 |
| compute-runtime / IGC | 26.35.39758.11 / IGC 2.41.9; Level Zero loader 1.32.0 |
| Kernel / `xe` driver / GuC firmware | 7.0.0-34-generic / in-tree `xe` / GuC 70.58.0 |
| oneCCL | not used |
| PyTorch XPU | not used |
| Power cap per card | 230 W |
| Tensor / pipeline parallel | layer split across 4 cards, `-ub 1024 -ts 13,13,13,10 -fit off`, MTP n-max 3 on `SYCL3` |

## Procedure

Pin the floor, restart the service, run [`tools/fnbench.py`](../../../tools/fnbench.py) (short, ~10k and ~39k prompts, greedy), and compare with the same build's runs at default clocks.

## Results

| Metric | Default clocks | Floor at 2800 MHz | Delta |
|---|---|---|---|
| Single-stream tok/s, short | 42-44 | 50 | +14% to +19% |
| Single-stream tok/s, ~10k | 52-54 | 64-66 | +19% to +27% |
| Single-stream tok/s, ~39k | 44-45 | 47-49 | +4% to +11% |
| Aggregate at c8 / c16 | not applicable (one slot) | | |

Ranges are over the session's runs, not repeated A/B pairs.

Idle cost, measured on 2026-09-25 from the cards' own hwmon energy counters with the model loaded and no requests: the GTs still enter C6 with `min_freq` at 2800 MHz (`act_freq` 0), so the floor costs about 0 W at true idle. Two caveats, both in the findings linked above:

- Anything that wakes the GPUs makes them run at full clock. `xpu-smi` run as root is one such thing: cards read 88.8 / 81.3 / 80.4 / 70.4 W on hwmon while it polled, against 45-48 W just before.
- The idle figures above are without PCIe ASPM. With ASPM L1 the four cards idle at about 25 W in total; see [findings/bmg-aspm-l1-idle-power.md](../../../findings/bmg-aspm-l1-idle-power.md).

## Correctness

Not applicable: clocks do not change results.

## Decision

Kept. The service pins the floor on start and restores the 400 MHz default on stop. The 2-lane vLLM layout keeps default clocks.

## Follow-ups

- [ ] Measure per-kernel duration with and without the floor to confirm the mechanism.
- [ ] Try an intermediate floor (for example 2000 MHz) to see how much of the gain it keeps.
