# llama.cpp SYCL runs a MUL_MAT with a 3D activation as one product per slice, re-reading the weight each time

| | |
|---|---|
| **Date** | 2026-09-26 |
| **Confidence** | verified-here (the effect was measured on four B70s; the mechanism is as stated in the patch's commit message, source lines not recorded) |
| **Applies to** | llama.cpp SYCL backend, PR #28243 tree at `6fcaa16` plus our patches; any graph that feeds a 3D (or 4D) activation into `MUL_MAT` against a 2D quantized weight; seen on Qwen3.8-Flash-Next's MTP `eh_proj`. Upstream master and other backends: UNVERIFIED |
| **Area** | engine |

## Finding

When `MUL_MAT`'s activation (`src1`) has a third dimension and the weight (`src0`) is 2D, the SYCL backend runs one product per `src1` slice, each dequantizing and reading the whole weight again. For Qwen3.8-Flash-Next's MTP (multi-token prediction) head, the input to `eh_proj` is `[2*n_embd, hc, n_tokens]` because of the model's hyper-connection (`hc`) axis, so a 1,536-token micro-batch (ubatch) ran 1,536 small products against a 14 MB Q8_0 weight: 115 ms per ubatch, where a single 2D product would take a few milliseconds. Reshaping `src1` to `[2*n_embd, hc*n_tokens]` before the product and back to 3D after it removed the cost and took about 29 s off a 135K cold read against the typical baseline read (12-30 s given the baseline's run-to-run spread).

## Evidence

- Prefill op profile ([raw/2026-09-26-prefill-op-profile.csv](../models/qwen3.8-flash-next/benchmarks/raw/2026-09-26-prefill-op-profile.csv), `GGML_SYCL_OP_PROFILE` from [patch 0004](../models/qwen3.8-flash-next/configs/patches/0004-sycl-op-profile.diff)): `MUL_MAT mtp_eh_proj` at 115.21 ms per call over 10 calls on build `993baf141` (4.7% of listed op time) and 115.08 ms per call over 22 calls on build `85b26781b` (8.4%). One call per ubatch. The next most expensive plain `MUL_MAT` in the same profile, `Qcur_full`, takes 1.10 ms per call. The [op-profile benchmark](../models/qwen3.8-flash-next/benchmarks/2026-09-26-prefill-op-profile.md) has the full tables.
- The fix is a two-line reshape, [patch 0016](../models/qwen3.8-flash-next/configs/patches/0016-qwen4exp-mtp-eh-proj-2d-product.diff). Its "commit message": "concat is [2*n_embd, hc, n_tokens]; a 3D src1 makes the SYCL backend run one hc-column product per token, each reading the whole 14 MB Q8_0 weight: 115 ms per 1536-token prompt ubatch (8.5% of a cold read at ~100K). The weight is 2D, so [2*n_embd, hc*n_tokens] is the same product. 135K cold read 313 -> 284 s."
- Cold reads of the same 134,862-token context, client-side ([raw/2026-09-26-cold-read-windows.csv](../models/qwen3.8-flash-next/benchmarks/raw/2026-09-26-cold-read-windows.csv)): 312.71, 314.11 and 295.68 s in three reads of the build without the fix (`85b26781b`; the fastest had the op profiler on for 62 s of it), 283.66 s in the first read with it, 286.44 s in a second read with `-b 3072`. The server's end-of-prompt progress lines are 0.56-0.77 s below these ([raw/2026-09-26-cold-read-progress.csv](../models/qwen3.8-flash-next/benchmarks/raw/2026-09-26-cold-read-progress.csv)). Details in the [experiment](../models/qwen3.8-flash-next/experiments/2026-09-26-mtp-eh-proj-2d-product.md).
- Target model output unchanged within run-to-run noise: teacher-forced next-token check, top-1 agreement 115/128 positions, KL (Kullback-Leibler divergence) median 0.00245 against the previous build, with same-build noise floors of 115-116/128 and median 0.0021-0.0026 ([raw/2026-09-26-klprobe-comparisons.csv](../models/qwen3.8-flash-next/benchmarks/raw/2026-09-26-klprobe-comparisons.csv)). The probe sends one-token requests, so the MTP draft, where `eh_proj` runs, never proposes a token in it; that the draft's output is unchanged rests on the construction (the same product), not on a measurement.
- Where the loop is: the SYCL backend's generic matmul driver iterates over the `ne12 x ne13` slices of `src1` and calls the per-slice GEMM (general matrix multiply; with the quantized `src0` converted to f16 for each call on the oneDNN path). This is our reading of `ggml/src/ggml-sycl/ggml-sycl.cpp` (`ggml_sycl_op_mul_mat`); line numbers for the `6fcaa16` tree were not recorded: UNVERIFIED as a source citation. The commit message states the behaviour; the profile and the cold reads confirm it.

Arithmetic (estimate, derived from the commit message and the profile): `eh_proj` is `[5120 x 2560]`, 13.1 M weights, 13.9 MB in Q8_0 (34 bytes per 32 weights). 1,536 slices times 13.9 MB is about 21 GB of weight reads per ubatch; at 115 ms that is about 186 GB/s. The same product as one 2D GEMM reads the weight once.

## Impact

- 115 ms per 1,536-token ubatch on this model: 4.7-8.4% of listed prompt op time in the two profiles, "8.5% of a cold read at ~100K" (commit message).
- A 135K cold read went from about 313 to about 284 s (-9%) against the typical baseline read; the baseline's three reads spanned 296-314 s, so the gain is between 4% and 10%. Against the typical read the saving is about three times the profiled op time (about 10 s per read; estimate: 66 full ubatches at 115 ms plus 65 of 512 tokens at about a third of that), so the 3D path cost more in normal execution than its serialized op time shows. Not investigated.
- Decode and short prompt turns did not change measurably; the draft head runs `eh_proj` on 1-4 tokens per step there, so the loop runs 1-4 times.
- Any model whose graph feeds a batched activation into a 2D weight on the SYCL backend pays the same per-slice cost. Hyper-connection residual streams are one source; others were not checked (UNVERIFIED).

## What to do

- In model graph code for the SYCL backend: when the weight is 2D and the activation has extra leading dimensions, reshape the activation to 2D for the product and reshape the result back. Pattern from patch 0016:

  ```cpp
  res = build_lora_mm(w, ggml_reshape_2d(ctx0, x, k, hc * n_tokens), w_s);
  res = ggml_reshape_3d(ctx0, res, n, hc, n_tokens);
  ```

- In op profiles, compare each `MUL_MAT`'s ms per call with its weight size. A 14 MB weight over 1,536 columns at 115 ms is two orders of magnitude off; the fix is in the graph, not the kernel.
- The reshape is safe only when `x` is contiguous in its leading dimensions so the 2D view is the same memory; `ggml_reshape_2d` asserts contiguity.

## Still open

- Whether upstream master still loops per slice for quantized 2D weights, and whether the f16-weight batched path (`ggml_sycl_mul_mat_batched_sycl`) would take over for f16 weights: UNVERIFIED, not checked in source.
- Whether the CUDA, Vulkan or Metal backends batch this case (CUDA has a strided-batched GEMM path for f16 weights; for quantized weights UNVERIFIED). The finding is stated for SYCL only.
- A profile of the fixed build was not taken; the expectation that `mtp_eh_proj` drops to a few milliseconds per call (the session estimated about 2 ms; source: session log, not archived) is UNVERIFIED.
- Whether the backend should flatten this case itself, so model code does not have to know. Worth an upstream issue with the numbers above.
