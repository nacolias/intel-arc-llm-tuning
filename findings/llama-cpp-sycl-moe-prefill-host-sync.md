# llama.cpp SYCL MoE matmuls sync the host and run one GEMM per expert for batches over 8 tokens

| | |
|---|---|
| **Date** | 2026-09-25 |
| **Confidence** | verified-here (measured on four B70s) and upstream-documented (source lines below) |
| **Applies to** | llama.cpp SYCL backend: PR #28243 at `6fcaa16`, and master at `171e8846b` (checked 2026-09-25, same code at shifted lines); any MoE model; measured on Qwen3.8-Flash-Next, 4x Arc Pro B70, `-sm layer` |
| **Area** | engine |

## Finding

For a MUL_MAT_ID (mixture-of-experts matmul) with more than one token, the SYCL backend copies the expert ids to the host, blocks until the queue drains, and then launches one ordinary matmul per expert that received tokens. On Qwen3.8-Flash-Next (48 MoE layers, 3 expert matmuls each, 512 experts) that is 144 host syncs and about 73,700 per-expert oneDNN GEMMs per micro-batch. It dominates prompt processing: the main CPU thread runs at 100% while each GPU is busy 17-28% of the time. Our patch 0003 moves 2-8-token batches (MTP verify) onto the fused path; larger batches still take the slow path.

## Evidence

**Source** (`ggml/src/ggml-sycl/ggml-sycl.cpp` at `6fcaa16`):

- `5153-5157`: only `ne12 == 1` (one token) tries the fused MMVQ kernel.
- `5159-5166`: `stream->memcpy(ids_host, ids_dev, …)` then `stream->wait()`.
- `5256-5282`: `for (i02 = 0; i02 < n_as; i02++)` over experts, calling `ggml_sycl_mul_mat` for each expert with rows. With `GGML_SYCL_DNN=ON` each call dequantizes the expert to f16 and runs a oneDNN GEMM with a freshly built primitive descriptor; the quantized MMQ path is disabled on SYCL.
- `6230`: `cpy_tensor_async = NULL`, so cross-device copies at split boundaries also block. This is why pipeline parallelism cannot overlap the cards.

On master `171e8846b` the same host copy and wait sit at lines 5224-5228 and the NULL `cpy_tensor_async` at 6303.

**Main-thread profile** during a 30K-token prefill ([prefill note](../models/qwen3.8-flash-next/research/2026-09-25-prefill-bottleneck.md)): kernel `sched_yield` spin 28.5%, Level Zero driver 23.7%, oneDNN 16% (4.1% in its primitive-cache lookup), libc memcpy/malloc/free 14%. Per-card compute busy 17-28%.

**Measured on Flash-Next, 4x B70:**

| Change | Result | Link |
|---|---|---|
| Fused per-token path for 2-8 tokens (patch 0003) | short-context MTP decode 37 to 42-44 tok/s, together with two small patches | [experiment](../models/qwen3.8-flash-next/experiments/2026-09-24-fused-mul-mat-id-mtp-verify.md) |
| Pipeline parallelism (drop `-ot`, `-b 4096`) | prefill 552.3 tok/s at 9.7K and 556.2 at 39K vs 552-630 / 553-560 baseline: 0% | [experiment](../models/qwen3.8-flash-next/experiments/2026-09-25-pipeline-parallel-prefill.md) |
| `-ub 1536` instead of 1024 | prefill +4.5% at 39K, +6% at 219K (the 219K comparison is against the unpatched build without the clock floor) | [experiment](../models/qwen3.8-flash-next/experiments/2026-09-25-ubatch-1536.md) |
| Fused per-token path up to 512 tokens | ~60-token turns unchanged, ~400-token turns slower (3.40-3.64 s vs 2.73-2.94 s) | [experiment](../models/qwen3.8-flash-next/experiments/2026-09-25-mmid-multitoken-prompt-turns.md) |

## Impact

- Stock MTP gains only 1.1-1.2x, because each 4-token verify takes the slow path: about 66-79 ms against about 30 ms for a one-token pass.
- Prefill on Flash-Next runs at about 570-645 tok/s at 10K-39K and 268 tok/s at 219K ([production benchmark](../models/qwen3.8-flash-next/benchmarks/2026-09-25-production-build.md)).
- Adding GPUs or pipeline parallelism does not help, because the host is the bottleneck.

## What to do

- Apply the 2-8-token fused path (patch 0003 in [configs/patches](../models/qwen3.8-flash-next/configs/patches/)) for any MoE model with MTP or other small-batch speculative decoding on SYCL.
- Use the largest `-ub` that fits in VRAM.
- Do not expect pipeline parallelism to help on SYCL until async copies land ([#29398](https://github.com/ggml-org/llama.cpp/issues/29398)).
- A real fix needs device-side expert routing plus a grouped MoE GEMM. Upstream [#29245](https://github.com/ggml-org/llama.cpp/pull/29245) adds a grouped XMX GEMM for IQ-type experts only and keeps the host wait (extended locally on 2026-09-26 to Q4_K, Q5_K, Q5_1 and Q8_0 experts as patches 0012-0015: see [grouped GEMM experiment](../models/qwen3.8-flash-next/experiments/2026-09-26-grouped-moe-xmx-gemm.md)).

## Still open

- Re-check when #29245 or a device-side routing change merges.
- Whether the Vulkan backend, which resolves expert ids on the GPU, prefills faster on the same cards is untested.
