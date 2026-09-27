# SYCL graphs (`GGML_SYCL_ENABLE_GRAPH=1`) with a 4-card layer split

| | |
|---|---|
| **Date** | 2026-09-24 |
| **Model / checkpoint** | Qwen3.8-Flash-Next `UD-Q4_K_XL` + unsloth MTP draft shared-Q8_0 |
| **Host** | [quad-b70-5800x-pex88096](../../../hardware/hosts/quad-b70-5800x-pex88096.md) |
| **Status** | reverted (no effect) |
| **Baseline** | same service, same build, variable unset |
| **Result** | [raw/2026-09-24-speed-work.csv](../benchmarks/raw/2026-09-24-speed-work.csv) |
| **Related** | [finding: llama.cpp SYCL graphs are off with more than one GPU](../../../findings/llama-cpp-sycl-graphs-off-with-multiple-gpus.md), [research: decode bottleneck](../research/2026-09-24-decode-bottleneck-and-speed-plan.md) |

## Hypothesis

Decode is host-bound: each token launches about 4,000 kernels. Recording the graph once and replaying it should cut launch overhead. Expected effect: unknown, possibly large.

## Change

```diff
+ Environment=GGML_SYCL_ENABLE_GRAPH=1
```

A runtime systemd drop-in on the service; the build has `GGML_SYCL_GRAPH=ON`.

## Environment

| Component | Version / value |
|---|---|
| Engine and image (tag and digest) | llama.cpp, no container. PR #28243 at `6fcaa16`, unpatched |
| Build flags | `GGML_SYCL=ON GGML_SYCL_TARGET=INTEL GGML_SYCL_F16=ON GGML_SYCL_DNN=ON GGML_SYCL_GRAPH=ON GGML_SYCL_SUPPORT_LEVEL_ZERO_API=ON GGML_NATIVE=ON`, Release, icx/icpx from oneAPI 2026.1 |
| compute-runtime / IGC | 26.35.39758.11 / IGC 2.41.9; Level Zero loader 1.32.0 |
| Kernel / `xe` driver / GuC firmware | 7.0.0-34-generic / in-tree `xe` / GuC 70.58.0 |
| oneCCL | not used |
| PyTorch XPU | not used |
| Power cap per card | 230 W |
| Tensor / pipeline parallel | layer split across 4 cards, `-ub 1024 -ts 13,13,13,10`, MTP n-max 3 |

## Procedure

Restart the service with the drop-in, send a short prompt with 512 tokens decoded, and read the eval timing from the server log. Compare with the same day's runs without the variable.

## Results

| Metric | Baseline | This change | Delta |
|---|---|---|---|
| Single-stream tok/s, short, 512 tokens | same level (exact value not kept) | 36.47 | none measurable |
| Aggregate at c8 / c16 | not applicable (one slot) | | |

One 512-token run with the variable, read from the service journal (`eval time = 14011.80 ms / 512 tokens`). Same-day runs without it were at the same level; their exact values were not kept.

## Correctness

Not applicable: graphs never engaged, so the same kernels ran.

## Decision

Reverted. In `ggml-sycl.cpp` at `6fcaa16`, `check_graph_compatibility` rejects graphs when more than one device is in use and when the graph holds any `MUL_MAT_ID` (around lines 6099-6121). A layer split over four cards hits the first check on every run, and this MoE model hits the second. The variable is a no-op for this model and host.

In the same session, the v1 Level Zero adapter (`SYCL_UR_USE_LEVEL_ZERO_V2=0`) also showed no effect. Its numbers were not kept.

## Follow-ups

- [ ] Making graphs usable here needs a per-split device check, a fused multi-token `MUL_MAT_ID`, and no oneMKL calls inside recorded graphs. The thread of upstream [#28725](https://github.com/ggml-org/llama.cpp/pull/28725) (SYCL graph record and replay) reports B70 gains from "negligible" to +1.7%, the latter on a patched Level Zero stack (UNVERIFIED here). See also the [research note](../research/2026-09-24-decode-bottleneck-and-speed-plan.md).
