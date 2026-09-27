# llama.cpp SYCL reorders Q4_K/Q5_K/Q6_K MoE expert weights lazily, so a prompt kernel meets two layouts

| | |
|---|---|
| **Date** | 2026-09-26 |
| **Confidence** | verified-here (`test-backend-ops` in both layouts on a B70) and upstream-documented (source below) |
| **Applies to** | llama.cpp SYCL backend with the reorder optimisation on (`GGML_SYCL_ENABLE_OPT=1` in the PR #28243 tree at `6fcaa16` we build from; the production launch script does not set the variable, and whether 1 is the compiled default is UNVERIFIED); any MoE model whose experts are Q4_K, Q5_K or Q6_K; any kernel that reads expert weights on the prompt path |
| **Area** | engine / quantization |

## Finding

The SYCL backend does not store Q4_K, Q5_K and Q6_K MoE (mixture-of-experts) expert tensors in one layout. They are loaded in the stored block layout (each block carries its own quants, scales and dequantization constants). The first time the fused single-token `MUL_MAT_ID` path runs on such a tensor, `opt_for_reorder_id` rewrites it in place into a per-expert SoA (structure-of-arrays) layout, `[qs][qh][scales][dm]` over all the expert's blocks, and marks the tensor as reordered. Until that moment, which is the first decode step after a prompt, the same tensor is in the stored layout. A kernel that reads expert weights on the prompt path therefore sees the stored layout during the first prompt and the SoA layout during every later prompt turn. It must decode both, or trigger the reorder itself and still decode both for the case where the optimisation is off, and it must be tested in both.

## Evidence

**Source** (llama.cpp PR #28243 at `6fcaa16`, plus our patches):

- `opt_for_reorder_id` in `ggml/src/ggml-sycl/ggml-sycl.cpp` (about line 4730 at `6fcaa16`; see the [engine notes](../engines/llama-cpp-sycl/README.md)) reorders on the fused decode path and exists only for Q4_K, Q5_K and Q6_K. The reorder functions are `reorder_qw_q*_k_moe`.
- Patch [0013](../models/qwen3.8-flash-next/configs/patches/0013-sycl-grouped-moe-gemm-q4k-q5k-q51-q80.diff) documents the layout the reorder produces and adds row views for it:

```cpp
// [qs][qh (q5_K only)][scales][dm] over the nb = M*K/QK_K blocks of the matrix
struct fg_q4_K_soa {};
struct fg_q5_K_soa {};
```

  and calls the reorder from the grouped prompt path so the kernel meets one layout once a prompt has run:

```cpp
// the decode path reorders Q4_K/Q5_K experts on first use; doing it here too means the
// grouped kernel only ever meets that layout once a prompt has run
opt_for_reorder_id(&ctx, src0);
const ggml_tensor_extra_gpu * src0_extra = static_cast<const ggml_tensor_extra_gpu *>(src0->extra);
const bool src0_reordered = src0_extra && src0_extra->optimized_feature.reorder;
```

  Its commit message: "Adds A stages for those, row views for the per-expert SoA layout reorder_qw_q*_k_moe leaves Q4_K/Q5_K experts in, and reorders on the grouped path too so it meets one layout."

- Q5_1 and Q8_0 have no reorder, so they are always in the stored layout; the grouped kernel's row view for them is the plain one. Q6_K has a reorder but no grouped decoder; a Q6_K expert tensor falls back to the per-expert loop (UNVERIFIED on a model; inferred from the type switch in 0013, which returns false for other types).

**Measured** on [quad-b70-5800x-pex88096](../hardware/hosts/quad-b70-5800x-pex88096.md), `test-backend-ops -o MUL_MAT_ID` on `SYCL0` against the CPU backend, 60 cases with Q4_K, Q5_K, Q5_1 and Q8_0 experts ([benchmark](../models/qwen3.8-flash-next/benchmarks/2026-09-26-mul-mat-id-kernel-perf.md), [experiment](../models/qwen3.8-flash-next/experiments/2026-09-26-grouped-moe-xmx-gemm.md)):

| Build | `GGML_SYCL_ENABLE_OPT=1` (reordered) | `GGML_SYCL_ENABLE_OPT=0` (stored layout) |
|---|---|---|
| `8793154f8` (patch 0013) | 60/60 | 60/60 |
| `517eb833d` (patch 0014) | 60/60 | 60/60 |
| `85b26781b` (patch 0015) | "60/60 with and without the reorder" (commit message) | same |

With `=1`, the grouped path reorders the test's expert tensor inside the test process before decoding it, so the SoA decoders are exercised; with `=0` the stored-layout decoders are. Pass counts for the first two builds are from the test-window outputs (source: session log, not archived). The model-level check, a teacher-forced KL (Kullback-Leibler divergence) probe over 128 positions after a 134,840-token prefix, was inside run-to-run noise for both builds ([raw](../models/qwen3.8-flash-next/benchmarks/raw/2026-09-26-klprobe-comparisons.csv)).

## Impact

- A prompt-path kernel written against the stored layout alone would read reordered bytes as blocks from the second prompt turn on. We did not run that failure; both decoders were written before the tests (UNVERIFIED as an observed failure).
- A kernel that decodes only the SoA layout would fail the first prompt after load, and always with `GGML_SYCL_ENABLE_OPT=0`.
- A `test-backend-ops` run in one layout proves nothing about the other. Half of the correctness evidence for patches 0013-0015 is the second run.
- The reorder is a one-time rewrite of every expert tensor. Its cost, and whether it is visible in the first prompt's time or the first decode step's, was not measured.

## What to do

- Any new SYCL kernel that reads Q4_K, Q5_K or Q6_K expert weights on the prompt path needs one row view per layout, as `fg_row<fg_q4_K_soa>` and `fg_row<fg_q5_K_soa>` in patch 0013, and a check of `optimized_feature.reorder` on the tensor's extra to pick it.
- Run `test-backend-ops -o MUL_MAT_ID` twice: `GGML_SYCL_ENABLE_OPT=1` and `GGML_SYCL_ENABLE_OPT=0`. Filter to the types and shapes the kernel handles.
- When timing a prompt kernel in the server, know that the first prompt after load either runs on the stored layout (stock) or pays the reorder (with 0013's early call).
- A Q6_K-expert model needs a Q6_K decoder before the grouped path covers it.

## Still open

- Checked at `6fcaa16` and our tree only. Upstream master may have moved the reorder trigger or added types; re-check when rebasing, and when [#29245](https://github.com/ggml-org/llama.cpp/pull/29245) (open on 2026-09-26) merges with its own layout handling.
- The one-time reorder cost is unmeasured.
- Whether the reorder ever runs for a tensor that only sees prompt batches (a server with `n_predict` 0) is untested; with 0013 the grouped path triggers it anyway.
