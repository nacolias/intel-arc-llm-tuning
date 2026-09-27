# Flash-Next prefill on llama.cpp SYCL: where the time goes and what could fix it

| | |
|---|---|
| **Date read** | 2026-09-25 |
| **Source** | our own measurements on [quad-b70-5800x-pex88096](../../../hardware/hosts/quad-b70-5800x-pex88096.md), `perf` of the server main thread, source reading of llama.cpp PR [#28243](https://github.com/ggml-org/llama.cpp/pull/28243) at `6fcaa16` plus our patches, and upstream PR [#29245](https://github.com/ggml-org/llama.cpp/pull/29245) |
| **Author / org** | this repository (AI-assisted analysis) |
| **Type** | profiling analysis and source-code review |
| **Applies to** | Qwen3.8-Flash-Next (llama.cpp arch `qwen4exp`, 48 layers, 512 routed experts) on llama.cpp SYCL, `-sm layer` across 4x Arc Pro B70 |

## Summary

Prefill is bound by the host, not by GPU compute. For batches of more than 8 tokens, the SYCL MUL_MAT_ID (mixture-of-experts matmul) path copies the expert ids to the host, waits for the queue, and then runs one oneDNN GEMM per expert. On this model that is 144 host syncs and about 73,700 per-expert GEMMs per micro-batch. During a 30K-token prefill the main thread sat at 100% while each card was busy only 17-28% of the time. Pipeline parallelism gained nothing, because the SYCL backend has no asynchronous cross-device copy. A larger micro-batch (`-ub 1536`) gained 4-6%.

## Measurements

**Prefill speed with the production build** ([production benchmark](../benchmarks/2026-09-25-production-build.md)):

| Prompt | Prompt tok/s |
|---|---|
| ~10K tokens | 589.8 |
| ~39K tokens | 577.2 |
| 219,217 tokens | 268.3 (816.9 s) |

**Main-thread profile** (`perf`, 10 s during a 30K prefill):

| Bucket | Share |
|---|---|
| kernel, mostly `sched_yield` spinning | 28.5% |
| `libze_intel_gpu` (Level Zero driver; its poll loop 6.5%) | 23.7% |
| `libdnnl` (oneDNN; `lru_cache` lookups 4.1%, GEMM execute) | 16% |
| libc (`memcpy`, `malloc`, `free`) | 14% |
| `libsycl` | 4.3% |
| `libggml-sycl` | 3.5% |

The main thread ran at 100% of one core. Per-card compute busy was 17-28% (sum 95%). GPU2's clock dipped to 2300-2450 MHz during the run.

**Prompt-only agent turns at 135K context** (about 60 new tokens, `n_predict` 1, median 1,204 ms), op shares from `GGML_SYCL_OP_PROFILE` (patch 0004):

| Op | Share |
|---|---|
| MUL_MAT_ID down / gate / up | 16.9% / 14.8% / 14.4% (46% together) |
| CONT, permuted | 4.2% |
| Dense flash attention, 12 sparse-attention layers | about 1.6-1.9% each (about 4 ms per call) |
| CONT, indexer pooling | 3.3% |

Dense flash attention runs here because 60 query rows exceed the sparse path's 32-row limit (`GGML_SYCL_SPARSE_FA_MAX_Q`, patch 0006). For scale: a 19-token prompt at short context takes 0.3 s, and real agent turns at 126K-140K context added 23-555 tokens in 1-3.5 s (two readings of the 2026-09-25 journal; [agent-session benchmark](../benchmarks/2026-09-25-agent-session-135k.md)).

## Mechanism (source)

Line numbers are for the stock PR #28243 tree at `6fcaa16`; our patches shift them.

1. **MUL_MAT_ID above 8 tokens** (above 1 token without our patch 0003). `ggml_sycl_mul_mat_id` copies `ids` device-to-host and calls `stream->wait()` (`ggml-sycl.cpp:5159-5166`), then loops over experts and calls `ggml_sycl_mul_mat` once per expert that has tokens (`5256-5282`). With 48 layers x 3 matmuls, that is 144 host syncs per micro-batch and about 73,700 GEMMs (48 x 3 x about 512 experts). Each GEMM dequantizes the expert to f16 and builds a new oneDNN `primitive_desc`; the quantized MMQ path is disabled on SYCL.
2. **No async cross-device copies.** The SYCL backend sets `cpy_tensor_async = NULL` (`ggml-sycl.cpp:6230`; an older implementation at 5878 is left unused). So the scheduler synchronises the source device at every split. Buffer `cpy_tensor` waits on every queue of both devices, `set_tensor` waits on every queue, and `event_wait` blocks the host.
3. **`-ot` disables pipeline parallelism.** Any tensor override turns it off (`src/llama-context.cpp:428-433`, `!model.has_tensor_overrides()`). The `-ot per_layer_token_embd=CPU` we passed was redundant: that tensor is created with `TENSOR_READ_LAZY` and placed in the CPU buffer type before overrides are checked (`llama-model-loader.cpp` about 1211).

## Experiments run

| Experiment | Result | Status |
|---|---|---|
| Pipeline parallelism: drop `-ot`, `-b 4096` ("pipeline parallelism enabled", 5 graph splits) | Prefill 552.3 tok/s at 9.7K and 556.2 at 39K, against a baseline of 552-630 and 553-560: no gain. Cost: 2.6-2.9 GiB more VRAM per card, 2.5-5.7 GiB more GTT (graphics translation table, host-mapped) memory per card, driver-held host RAM 27 GiB vs 16 | reverted ([experiment](../experiments/2026-09-25-pipeline-parallel-prefill.md)) |
| `-ub 1536 -ts 12,13,13,11` vs `-ub 1024 -ts 13,13,13,10` | Prefill +4.5% at 39K (582.8 vs 553-560), +6% at 219K (270.0 vs 254.5; comparison against the unpatched build without the clock floor); decode unchanged within noise; fullest card 28.2 GiB | kept ([experiment](../experiments/2026-09-25-ubatch-1536.md)) |
| Fused per-token MUL_MAT_ID up to 512 tokens (`GGML_SYCL_MMID_MULTITOKEN_MAX=512`) | ~60-token turns 1.08-1.31 s (unchanged vs 1.11-1.35); ~400-token turns 3.40-3.64 s (worse vs 2.73-2.94) | reverted ([experiment](../experiments/2026-09-25-mmid-multitoken-prompt-turns.md)) |

The per-token fused path reads each selected expert once per token, so it loses on longer batches. It also shows that ~60-token turns at 135K are not bound by the host-synced MoE loop alone.

## Options

| Option | What it removes | Evidence | Status |
|---|---|---|---|
| Grouped MoE GEMM kernel | the per-expert GEMM launches and oneDNN primitive setup | Upstream [#29245](https://github.com/ggml-org/llama.cpp/pull/29245) "sycl: add grouped MoE XMX GEMM" (open as of 2026-09-25) dispatches only IQ types (IQ4_NL, IQ3_S, IQ4_XS, IQ3_XXS, IQ2_*, IQ1_*) and keeps the host wait. Author reports +34% prompt processing and +0.95% decode | UNVERIFIED. Our experts are K-quants and Q5_1 (43 of 48 `down_exps` are Q5_1), with no IQ types, so it would not engage as posted. Extended locally on 2026-09-26 to Q4_K, Q5_K, Q5_1 and Q8_0 experts (patches 0012-0015): see [grouped GEMM](../experiments/2026-09-26-grouped-moe-xmx-gemm.md) |
| Device-side routing | the 144 host syncs per micro-batch | Compute per-expert token counts and offsets on the GPU, then a grouped GEMM that reads them from device memory. No SYCL implementation exists | design only |
| Async split copies and device-side events | the blocking hop at every split; a prerequisite for pipeline parallelism to overlap cards | Feature request [#29398](https://github.com/ggml-org/llama.cpp/issues/29398) (open); a de-dpct / out-of-order queue PR (#29190) was closed unmerged | not started |
| Host trims | part of the 28.5% spin, 16% oneDNN and 14% libc | Cache one oneDNN primitive per expert shape; avoid per-GEMM allocation and f16 dequantize buffers; replace the spin wait | ideas; gains unmeasured |
| Sparse flash attention for longer batches | about 1.6-1.9% per sparse layer on ~60-token turns | Patch 0006 stops at 32 query rows and at `T*n_kv_g*2 <= n_kv` | small; untested |
| Larger micro-batch | fewer syncs per prompt token | `-ub 1536` done; the fullest card has about 3.5 GiB free | done |

## Claims to verify

| Claim | Reported number | Their setup | Status |
|---|---|---|---|
| #29245 speeds up MoE prompt processing | +34% pp, +0.95% decode | author, IQ-quant MoE | UNVERIFIED |
| Pipeline parallelism speeds prefill on multi-GPU SYCL | – | llama.cpp default when no overrides | measured 0% here |
| Larger `-ub` speeds prefill | – | – | verified-here: +4.5-6% at 1536 |
| Device-side routing plus a grouped kernel would remove most prompt time | – | our inference from the profile | UNVERIFIED |

## Relevance

Agent sessions reuse a cached prefix, so the prompt turns that matter add a few dozen to a few hundred tokens at 130K+ context and take about 1-3.5 s. Cold reads of a 135K prompt take 405-472 s (on the 2026-09-25 builds; 283-286 s on 2026-09-26, see [cold read by build](../benchmarks/2026-09-26-cold-read-135k-by-build.md)). A fix that only helps large batches (grouped GEMM) shortens cold reads; per-turn latency also depends on the dense attention and pooling costs in the table above.

## Measurement plan for any fix

- Prefill tok/s at ~10K, ~39K and ~219K with `tools/fnbench.py`, at least 3 runs.
- Per-turn prompt time for ~60- and ~400-token turns at 135K with `tools/lcbench.py`.
- `GGML_SYCL_OP_PROFILE` shares for MUL_MAT_ID and CONT.
- Main-thread `perf` split and per-card busy (`tools/gpuprof.py`).
- `test-backend-ops -o MUL_MAT_ID` for n > 8 on q4_K, q5_K and q5_1.
- Output check with `tools/klprobe.py` (top-1 agreement and KL over 128 positions) against the current build.

## Actions

- [x] Pipeline-parallel A/B (reverted)
- [x] `-ub 1536` (kept)
- [x] Per-token fused MUL_MAT_ID up to 512 tokens (reverted)
- [ ] Device-side expert routing in `ggml_sycl_mul_mat_id` for n > 8
- [x] Grouped MoE GEMM for Q4_K / Q5_K / Q5_1 experts, building on #29245 (built on 2026-09-26 as patches 0012-0015: [experiment](../experiments/2026-09-26-grouped-moe-xmx-gemm.md))
- [ ] Cache oneDNN primitives per expert shape (cheap host trim)
