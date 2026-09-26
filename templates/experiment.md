# <Short title of the change>

| | |
|---|---|
| **Date** | YYYY-MM-DD |
| **Model / checkpoint** | |
| **Host** | [host](../../../hardware/hosts/<host>.md) |
| **Status** | planned / running / kept / reverted / inconclusive / blocked |
| **Baseline** | [benchmark](../benchmarks/YYYY-MM-DD-<slug>.md) |
| **Result** | [benchmark](../benchmarks/YYYY-MM-DD-<slug>.md) |
| **Related** | research notes, findings, upstream issues |

## Hypothesis

What you expect to change and why, in one or two sentences. Include the expected size of the effect.

## Change

The exact diff against the baseline: flags, env vars, image, patch, driver, quant. Paste it.

```diff
- SPEC={"method":"mtp","num_speculative_tokens":2}
+ SPEC={"method":"mtp","num_speculative_tokens":4}
```

## Environment

| Component | Version / value |
|---|---|
| Engine and image (tag and digest) | |
| compute-runtime / IGC | |
| Kernel / `xe` driver / GuC firmware | |
| oneCCL | |
| PyTorch XPU | |
| Power cap per card | |
| Tensor / pipeline parallel | |

## Procedure

Steps to reproduce, including warm-up and the benchmark command.

## Results

| Metric | Baseline | This change | Delta |
|---|---|---|---|
| Single-stream tok/s | | | |
| Aggregate tok/s at c8 | | | |
| Aggregate tok/s at c16 | | | |
| TTFT at <N>K context | | | |
| Spec-decode acceptance | | | |

## Correctness

Greedy output compared against the reference? Token-identical? Any NaN, repetition, or garbled output under concurrency?

## Decision

Kept, reverted or inconclusive, and why.

## Follow-ups

- [ ]
