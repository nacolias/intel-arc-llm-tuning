# Flash-Next decode on four B70s: bottleneck analysis and ranked speed plan

| | |
|---|---|
| **Date read** | 2026-09-24 (outcomes added 2026-09-25) |
| **Source** | our own analysis: source reading of llama.cpp [PR #28243](https://github.com/ggml-org/llama.cpp/pull/28243) at `6fcaa16` (`ggml/src/ggml-sycl/`, `src/models/qwen4exp.cpp`), `perf` samples of the running server, a GGUF tensor dump, and the upstream PRs linked below |
| **Author / org** | this repository (AI-assisted analysis) |
| **Type** | source-code review and profiling analysis |
| **Applies to** | Qwen3.8-Flash-Next UD-Q4_K_XL GGUF, llama.cpp SYCL backend with PR #28243, `-sm layer` across 4x Arc Pro B70, host [quad-b70-5800x-pex88096](../../../hardware/hosts/quad-b70-5800x-pex88096.md) |

## Summary

On 2026-09-24, decode was limited by the host, not by memory bandwidth. The main CPU thread ran at about 95% of one core while each GPU was busy only 15-18% of the time. Layer split runs the four cards one after another, and every token launches about 4,000 kernels. The worst single defect was the MTP (multi-token prediction) verify pass, which took a host-synchronised slow path for MoE (mixture-of-experts) matmuls. Fixing that path, plus two small patches and a GPU clock floor, took short-context decode from about 37 to 50 tok/s.

Nothing in the plan section was measured when it was written. Gains in the plan are estimates. The "What happened" column and the results table are measured.

## Where the time went (2026-09-24)

**The main thread was mostly waiting, not making syscalls.** Two re-analyses of the same `perf` capture (MTP decode, sample IP joined to call chain) agree:

| Main-thread bucket | Share | Label |
|---|---|---|
| Kernel-launch path (Level Zero driver `libze` about 21% of it) | about 35% | verified-here |
| Spinning on GPU completion inside blocking USM (unified shared memory) memcpy, including `sched_yield` | about 25-30% | verified-here |
| `gettid` syscalls from the dpct helper layer | about 6.5% | verified-here |
| ggml-sycl host code | about 9% | verified-here |
| DRM ioctls | about 0.03% | verified-here |

Direct submission (ULLS, ultra-low-latency submission) is already on with compute-runtime 26.35 on Xe2 and `xe`, so there is no ioctl per launch. An earlier reading of "20% ioctls" was the sampler labelling a user-mode poll as `entry_SYSCALL_64`.

**Launch count.** About 4,000 kernel launches per token at about 5 µs each. This is a hand count from source, ±20%, and the per-launch cost is derived from it (UNVERIFIED).

**The MTP verify slow path.** `--spec-draft-p-min` defaults to 0.00, so every step verifies 4 tokens. With more than one token (`ne12 != 1`), all 144 MUL_MAT_ID calls per pass (48 layers x gate, up, down) skip the fused kernel (`ggml-sycl.cpp:5044, 5050, 5153` at `6fcaa16`). Each call then:

1. copies the expert ids device-to-host and calls `stream->wait()` (`5159-5166`);
2. launches one matmul per distinct expert, about 39 of them (`5256-5282`).

That is roughly 11,000 extra launches per verify. A verify cost about 66-79 ms against about 30 ms for a single-token pass, which is why MTP gave only 1.1-1.2x. Mechanism verified in source; the wall-time share is an inference.

**SYCL graphs do nothing here.** `check_graph_compatibility` rejects `device_count > 1` and any MUL_MAT_ID (`ggml-sycl.cpp:6099-6121`). Measured 36.47 tok/s with `GGML_SYCL_ENABLE_GRAPH=1`, the same as without. See [the finding](../../../findings/llama-cpp-sycl-graphs-off-with-multiple-gpus.md).

**Measurement noise.** The same config gave 31-38 tok/s on 512-token short runs, with MTP acceptance at temperature 1.0 between 57% and 71%. A single run cannot resolve a 5% change. Later benchmarks used temperature 0 and 5 repeats.

## Roofline under layer split

A token reads 6.33 GB of weights (GGUF tensor dump). `-sm layer` runs the cards in sequence, so the reads do not overlap.

| Quantity | Value | Label |
|---|---|---|
| Weight bytes per token | 6.33 GB | verified-here (GGUF dump) |
| Gated DeltaNet state traffic per token | about 0.45 GB | derived from GGUF headers |
| Time per token at 608 GB/s, sequential cards (weights plus state, 6.78 GB) | 11.2 ms | derived |
| Decode cap without MTP (100% / 75% of bandwidth) | 90 / about 67 tok/s | estimate; 608 GB/s is UNVERIFIED |
| Decode cap with MTP (100% / 75%) | 127 / about 95 tok/s | estimate |

The dense Qwen3.8-27B reached 129 tok/s at TP4 on the same host ([benchmark](../../qwen3.8-27b/benchmarks/2026-09-23-tp4-quad-b70.md)) because tensor parallelism spreads its 15.5 GB read across four cards in parallel, about 6.4 ms per token. Flash-Next cannot use tensor parallelism in llama.cpp today:

- qwen4exp is excluded from `-sm tensor` (`llama-arch.cpp:1167`; re-enable proposed in [#28569](https://github.com/ggml-org/llama.cpp/pull/28569), open).
- The SYCL all-reduce only supports 2 devices (`ggml-sycl.cpp:6926-6974`).
- `-sm tensor` was reported 3x slower than one GPU on 2x B70 ([#26409](https://github.com/ggml-org/llama.cpp/issues/26409), UNVERIFIED here).
- `-sm row` cannot split MoE tensors (`ggml-sycl.cpp:5143`).

## Claims to verify

| Claim | Reported number | Their setup | Status |
|---|---|---|---|
| Fused per-token MUL_MAT_ID for 2-8-token batches speeds MTP decode | +30-60%, to 50-60 tok/s short | plan estimate | measured: +14-19% short with patches 0002 and 0001 in the same build ([experiment](../experiments/2026-09-24-fused-mul-mat-id-mtp-verify.md)) |
| Caching `gettid` removes the 6.5% syscall share | +2-5% | plan estimate | not measured on its own; shipped in the same build as the fused path |
| An earlier claim that caching `gettid` gives +20% | +20% | earlier chat estimate | refuted by the perf split above |
| [#28931](https://github.com/ggml-org/llama.cpp/pull/28931) rms_norm+scale fusion removes 72 launches per pass | +1-2% | plan estimate | not measured on its own; shipped in the same build |
| SYCL sparse flash attention ([#28796](https://github.com/ggml-org/llama.cpp/pull/28796)) helps long context | author: +52% at 28.6K, +143% at 88.4K, MTP off, q8_0 KV, one B70 | author's post on the PR | UNVERIFIED for single-token only; our multi-token extension measured 22.2 to 34.5 tok/s at 135K ([experiment](../experiments/2026-09-25-sparse-fa-multi-token.md)) |
| Q8_0 kernel PRs [#29186](https://github.com/ggml-org/llama.cpp/pull/29186), [#29337](https://github.com/ggml-org/llama.cpp/pull/29337) | author: +5.6-7.3% without MTP | author's model family | UNVERIFIED; not tried |
| Deeper MTP drafts pay off once verify is cheap | +25% over MTP off with n-max 7, p-min 0.75 | 2x B70, comment on PR #28243 | UNVERIFIED; our sweep found no gain ([experiment](../experiments/2026-09-24-mtp-draft-length-sweep.md)) |
| `GGML_SYCL_PRIORITIZE_DMMV=1` helps | n/a | community suggestion | refuted in source: Q8_0 DMMV is also 2 launches with `GGML_SYCL_F16` (`dmmv.cpp:2071-2089`), and the setting disables the shared-expert GLU fusion (`ggml-sycl.cpp:4880-4884`) |
| `EnableDirectSubmission=1` helps | n/a | community suggestion | likely no-op: direct submission is already on for Xe2 with `xe` vm_bind (compute-runtime `product_helper_xe2_and_later.inl:80-85`) |

## Ranked levers and what happened to each

The plan scored each lever as gain x confidence / effort, only to order the work.

| # | Lever | Plan estimate | What happened |
|---|---|---|---|
| 1 | Sync-free MUL_MAT_ID for 2-8-token batches (MTP verify) | +30-60% with MTP | Done as patch 0003 ([configs/patches](../configs/patches/)). **Kept.** Short 37 to 42-44 tok/s together with #3 and #5 ([experiment](../experiments/2026-09-24-fused-mul-mat-id-mtp-verify.md)). |
| 2 | Draft shaping (`--spec-draft-p-min`, `--spec-draft-n-max`) | 0-10% | Swept; no gain over n-max 3 ([experiment](../experiments/2026-09-24-mtp-draft-length-sweep.md)). |
| 3 | Cache `gettid` per thread | +2-5% | Done as patch 0002. **Kept**, in the same build as #1. |
| 4 | SYCL sparse flash attention, #28796 | long context only | Merged upstream as `cd74ef627`; cherry-picked as patch 0005. Single-token only, so it never engaged on MTP verify. Extended to multi-token batches in patch 0006. **Kept** ([experiment](../experiments/2026-09-25-sparse-fa-multi-token.md)). |
| 5 | Cherry-pick #28931 (`5e48b3100`) | +1-2% | Done as patch 0001. **Kept.** |
| 6 | Q8_0 kernel PRs #29186, #29337 (+ #29338) | ≤1-2% with MTP | Not tried. |
| 7 | Vulkan backend A/B | unknown | Not tried. |
| 8 | F32 GEMV instead of oneMKL for 264 small F32 matmuls per token | +2-6% | Not tried. |
| 9 | Level Zero / Unified Runtime / compute-runtime environment variables | 0-3% each | The v1 Level Zero adapter (`SYCL_UR_USE_LEVEL_ZERO_V2=0`) showed no effect. The others were not tried. |
| 10 | Requantize the Q8_0 trunk to Q6_K | +3-8% | Not tried (quality risk). |
| 11 | Async cross-GPU split copies ([#29398](https://github.com/ggml-org/llama.cpp/issues/29398)) | +2-8% | Not implemented. The pipeline-parallel test confirmed `cpy_tensor_async` is NULL in SYCL ([experiment](../experiments/2026-09-25-pipeline-parallel-prefill.md)). |
| 12 | qwen4exp-specific fusions (MoE gate+up+SwiGLU, hyper-connection chains) | +10-20% | Not tried. |
| 13 | Q5_1 MoE weight reorder (43 of 48 `down_exps` are Q5_1) | small | Not tried. |
| 14 | SYCL graphs with multiple GPUs and MoE | 0-15% | Not tried; the stock switch is a no-op ([experiment](../experiments/2026-09-24-sycl-graphs-multi-gpu.md)). |
| – | GPU clock floor at rp0 (not in the plan) | – | **Kept.** Short 42-44 to 50 tok/s, 10K 52-54 to 64-66 ([experiment](../experiments/2026-09-24-gpu-clock-floor.md), [finding](../../../findings/gpu-clock-floor-speeds-layer-split.md)). |
| – | Pooled QSA indexer-key cache (long-context plan) | – | **Kept.** 135K decode median 29.9 to 48.5 tok/s ([experiment](../experiments/2026-09-25-pooled-qsa-key-cache.md)). |

Dropped before testing: `-sm tensor` and `-sm row` (see roofline section), [#28213](https://github.com/ggml-org/llama.cpp/pull/28213) (superseded by #28796), [#28699](https://github.com/ggml-org/llama.cpp/pull/28699) (draft pooled-key cache with open bugs, including an M-RoPE assert that matters with a vision projector loaded), turning off CPU mitigations (security cost), and switching to vLLM ([vLLM deep dive](2026-09-25-vllm-xpu-deep-dive.md)).

## Results so far (MTP on, single stream)

| Stage | Short | ~10K | ~39K | ~219K | Source |
|---|---|---|---|---|---|
| Start of 2026-09-24 | about 37 | 43-47 | 33-38 | 16 | [baseline benchmark](../benchmarks/2026-09-24-baseline-262k-mtp.md) |
| + fused MUL_MAT_ID, `gettid` cache, #28931 | 42-44 | 52-54 | 44-45 | – | [experiment](../experiments/2026-09-24-fused-mul-mat-id-mtp-verify.md) |
| + GPU clock floor | 50 | 64-66 | 47-49 | – | [experiment](../experiments/2026-09-24-gpu-clock-floor.md) |
| Production build `20260925-f47a6a5f3` (all kept patches, `-ub 1536`, sparse FA, pooled key cache) | 48.6 | 60.6 | 53.6 | 40.6 | [production benchmark](../benchmarks/2026-09-25-production-build.md) |

Decode figures are tok/s. The plan's target after lever #1 was 50-60 tok/s short; the stack reached 50 with the clock floor.

## Relevance

- The remaining short-context limit is launch count and the roughly 5 blocking split hops per token. Fusions (#12) and async copies (#11) attack those.
- The bandwidth cap under layer split (about 95 tok/s with MTP at 75% efficiency) is far away, so kernel speed is not the next lever.
- Long context was a separate problem: dense attention over every KV cell plus indexer re-pooling. See the [long-context QSA design](2026-09-25-long-context-qsa-design.md).

## Actions

- [x] Fused MUL_MAT_ID for 2-8 tokens (patch 0003)
- [x] `gettid` cache (patch 0002), #28931 cherry-pick (patch 0001)
- [x] Draft-length sweep
- [x] Sparse FA single-token (0005) and multi-token (0006)
- [ ] qwen4exp fusions: MoE gate+up+SwiGLU, hyper-connection mix chains (about 31% of launches)
- [ ] F32 GEMV for the router, hyper-connection inject and Gated DeltaNet alpha/beta matmuls
- [ ] Async split-boundary copies (#29398)
- [ ] Q8_0 kernel PRs once merged upstream
