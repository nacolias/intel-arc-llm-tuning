# <Configuration name>: <workload> benchmark

| | |
|---|---|
| **Date** | YYYY-MM-DD |
| **Model / checkpoint** | |
| **Host** | [host](../../../hardware/hosts/<host>.md) |
| **Config** | [config](../configs/<file>) |
| **Experiment** | [experiment](../experiments/YYYY-MM-DD-<slug>.md) |
| **Raw data** | [raw/](raw/) |

## Environment

| Component | Version / value |
|---|---|
| Engine and image (tag and digest) | |
| compute-runtime / IGC | |
| Kernel / `xe` driver / GuC firmware | |
| oneCCL | |
| PyTorch XPU | |
| Power cap per card | |
| KV cache dtype / block size | |
| Speculative decoding | |
| Graph / compile mode | |

## Workload

| | |
|---|---|
| Workload set | [benchmarks/workloads/](../../../benchmarks/workloads/) |
| Input length | |
| Output length | |
| Sampling | greedy / temperature, top_p |
| Warm-up | requests discarded before measuring |
| Repetitions | |

Command:

```bash
```

## Results

| Concurrency | Aggregate tok/s | Per-stream tok/s | TTFT p50 | TPOT p50 | Acceptance |
|---|---|---|---|---|---|
| 1 | | | | | |
| 2 | | | | | |
| 4 | | | | | |
| 8 | | | | | |
| 16 | | | | | |
| 32 | | | | | |

| Context length | TTFT (cold) | Single-stream decode tok/s |
|---|---|---|
| 2K | | |
| 32K | | |
| 128K | | |

## Observations

Power draw, temperatures, clocks, errors in logs, anything unusual.
