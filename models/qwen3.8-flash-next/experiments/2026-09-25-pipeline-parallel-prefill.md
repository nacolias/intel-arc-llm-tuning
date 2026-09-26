# Pipeline-parallel prefill (drop the PLE `-ot`, `-b 4096`)

| | |
|---|---|
| **Date** | 2026-09-25 |
| **Model / checkpoint** | huihui-ai abliterated `UD-Q4_K_XL` + unsloth MTP draft shared-Q8_0 + mmproj BF16 |
| **Host** | [quad-b70-5800x-pex88096](../../../hardware/hosts/quad-b70-5800x-pex88096.md) |
| **Status** | reverted (no gain, more memory) |
| **Baseline** | build `20260924-7e5cb8f13`, `-ub 1024 -ts 13,13,13,10`, clock floor on ([raw/2026-09-25-ubatch-compare.csv](../benchmarks/raw/2026-09-25-ubatch-compare.csv), ub 1024 column) |
| **Result** | [raw/2026-09-25-production-fnbench.csv](../benchmarks/raw/2026-09-25-production-fnbench.csv), row `pp-test` |
| **Related** | [research: prefill bottleneck](../research/2026-09-25-prefill-bottleneck.md), [finding: MoE prefill host sync](../../../findings/llama-cpp-sycl-moe-prefill-host-sync.md) |

## Hypothesis

Prompt processing runs at about 550-630 tok/s at 10k-39k. With `-sm layer` the cards take turns, so overlapping ubatches across cards (llama.cpp's pipeline parallelism) should raise prefill speed. Two facts from the source made this a flag-only test:

- `per_layer_token_embd` (the n-gram table) is created with `TENSOR_READ_LAZY` and lands in the CPU buffer type before `-ot` overrides are checked (`llama-model-loader.cpp`, around line 1211). So `-ot per_layer_token_embd=CPU` is redundant.
- Any `-ot` disables pipeline parallelism (`src/llama-context.cpp:428-433`, the `!model.has_tensor_overrides()` condition).

## Change

```diff
- -ot 'per_layer_token_embd=CPU' -b 2048 -ub 1024
+                                -b 4096 -ub 1024
```

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
| Tensor / pipeline parallel | layer split across 4 cards, `-ts 13,13,13,10 -fit off`; this change turns on llama.cpp pipeline parallelism |
| GPU clock floor | 2800 MHz |

## Procedure

Restart with the change, check the log for "pipeline parallelism enabled" and the graph split count, record VRAM, GTT and driver-held host RAM after load, then run [`tools/fnbench.py`](../../../tools/fnbench.py). A 10-second `perf` sample of the main thread was taken during a 30k-token prefill.

## Results

| Metric | Baseline | This change | Delta |
|---|---|---|---|
| Prompt tok/s, ~10k (9,749 tokens) | 552-630 | 552.3 | none |
| Prompt tok/s, ~39k (39,119 tokens) | 553-560 | 556.2 | none |
| Single-stream tok/s, short / ~10k / ~39k | 50 / 64 / 49 | 49.9 / 55.3 / 48.0 | ~10k run lower (acceptance 177/234) |
| VRAM per card after load (MiB) | 27,809 / 26,195 / 26,445 / 24,846 | 30,498 / 29,117 / 29,384 / 27,767 | +2.6 to +2.9 GiB per card |
| GTT per card (GiB) | 1.4 / 4.4 / 4.4 / 4.9 | 3.9 / 9.8 / 10.1 / 10.1 | |
| Driver-held host RAM | 16 GiB | 27 GiB | +11 GiB |
| Compute buffers (MiB) | - | SYCL0 5,927; SYCL1-3 6,167; host 3,114 | |
| Aggregate at c8 / c16 | not applicable (one slot) | | |

The log confirmed "pipeline parallelism enabled" with 5 graph splits.

Why it gained nothing, from the source and the profile:

- The SYCL backend sets `cpy_tensor_async = NULL` (`ggml-sycl.cpp:6230` in the stock `6fcaa16` tree; our patches shift the line), so the scheduler synchronises the source device at every split. Buffer `cpy_tensor` and `set_tensor` wait on all queues, and `event_wait` blocks the host.
- The larger cost: for batches over 8 tokens, `ggml_sycl_mul_mat_id` copies the expert ids to the host and waits, then runs one GEMM per expert. That is 144 host syncs per ubatch and about 73.7k per-expert GEMMs (48 layers x 3 projections x about 512 experts). Each GEMM dequantizes its expert to f16 and builds a new oneDNN primitive descriptor.
- Main-thread `perf` during prefill: kernel 28.5% (mostly `sched_yield` spinning), Level Zero driver 23.7% (poll loop 6.5%), oneDNN 16% (LRU cache lookup 4.1%), libc 14% (memcpy, malloc, free), libsycl 4.3%, ggml-sycl 3.5%. The main thread sat at 100%, and per-card compute busy was 17-28% (95% summed over the four cards).

## Correctness

Not assessed beyond the sanity question: no kernel changed, only scheduling.

## Decision

Reverted. Prefill is bound by the MoE path's host synchronisation, not by the cards taking turns. The change cost about 2.8 GiB of VRAM per card and 11 GiB of driver-held host RAM for no gain. The redundant `-ot` stays in the config so that pipeline parallelism stays off.

## Follow-ups

- [ ] A grouped multi-token `MUL_MAT_ID` for SYCL (one kernel reading each expert once for its routed tokens, ids kept on the device). See [findings/llama-cpp-sycl-moe-prefill-host-sync.md](../../../findings/llama-cpp-sycl-moe-prefill-host-sync.md).
- [ ] Implement `cpy_tensor_async` for SYCL; retest pipeline parallelism after that.
