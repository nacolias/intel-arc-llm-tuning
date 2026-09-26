# <Model name>

| | |
|---|---|
| **Status** | candidate / baseline / active tuning / production / parked / abandoned |
| **Upstream** | link to the base model |
| **License** | |
| **Last updated** | YYYY-MM-DD |

## Architecture

| Property | Value |
|---|---|
| Parameters (total / active) | |
| Layers and layer types | |
| Hidden size / heads / KV heads | |
| Vocabulary | |
| Native context length | |
| Native speculative head (MTP) | yes / no |
| Notable ops (MoE, linear attention, sliding window, ...) | |

Why the architecture matters on Arc: which kernels it needs on XPU, and which are known to be slow or missing.

## Checkpoints

| Checkpoint | Quant method | Bits / group | Size on disk | XPU kernel path | Status |
|---|---|---|---|---|---|
| | | | | | tried / best / rejected |

## Current best config

- Config: [configs/](configs/)
- Benchmark: [benchmarks/](benchmarks/)

Summary of the settings that matter (parallelism, quant, KV dtype, speculative decoding, graph mode, power cap).

## Headline numbers

| Metric | Value | Benchmark |
|---|---|---|
| Single-stream decode | | |
| Aggregate at c8 | | |
| Aggregate at c16 | | |
| TTFT at long context | | |
| Max context served | | |

## What worked and what did not

- **Worked:**
- **Did not work:**

## Open questions

- [ ]

## Index

- [Research notes](research/)
- [Experiments log](experiments/README.md)
- [Benchmarks](benchmarks/README.md)
- [Configs](configs/README.md)
