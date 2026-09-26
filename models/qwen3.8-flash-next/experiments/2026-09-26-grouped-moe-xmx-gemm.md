# Grouped MoE XMX GEMM for prompt batches: upstream #29245 extended to Q4_K, Q5_K, Q5_1 and Q8_0 experts

| | |
|---|---|
| **Date** | 2026-09-26 |
| **Model / checkpoint** | huihui-ai abliterated `UD-Q4_K_XL` + unsloth MTP draft shared-Q8_0 + mmproj BF16 |
| **Host** | [quad-b70-5800x-pex88096](../../../hardware/hosts/quad-b70-5800x-pex88096.md) |
| **Status** | kept |
| **Baseline** | build `20260926-2c88bdf19` (per-expert oneDNN loop for batches over 8 tokens); in the kernel test, the same binary with `GGML_SYCL_XMX_GATHER_TYPES=0` |
| **Result** | [benchmarks/2026-09-26-mul-mat-id-kernel-perf.md](../benchmarks/2026-09-26-mul-mat-id-kernel-perf.md); raw: [kernel perf](../benchmarks/raw/2026-09-26-mul-mat-id-kernel-perf.csv), [kernel diag](../benchmarks/raw/2026-09-26-mul-mat-id-kernel-diag.csv), [KL comparisons](../benchmarks/raw/2026-09-26-klprobe-comparisons.csv), [cold-read progress](../benchmarks/raw/2026-09-26-cold-read-progress.csv), [prefill op profile](../benchmarks/raw/2026-09-26-prefill-op-profile.csv) |
| **Related** | [research: where the 4 ms goes](../research/2026-09-26-grouped-moe-gemm-cost-breakdown.md), [research: prefill bottleneck](../research/2026-09-25-prefill-bottleneck.md), [finding: MoE prefill host sync](../../../findings/llama-cpp-sycl-moe-prefill-host-sync.md), [finding: K-quant experts are reordered lazily](../../../findings/llama-cpp-sycl-kquant-experts-reordered-lazily.md), patches [0012](../configs/patches/0012-upstream-pr29245-sycl-grouped-moe-xmx-gemm.diff), [0013](../configs/patches/0013-sycl-grouped-moe-gemm-q4k-q5k-q51-q80.diff), [0014](../configs/patches/0014-sycl-grouped-moe-gemm-vector-loads.diff), [0015](../configs/patches/0015-sycl-grouped-moe-gemm-in-place-io.diff), upstream [#29245](https://github.com/ggml-org/llama.cpp/pull/29245) |

One expert matmul over a 1536-token micro-batch went from about 11 ms to 3.8-4.4 ms for the Q4_K, Q5_K and Q5_1 experts and to 7.4 ms for Q8_0. A cold read of a 135K-token conversation went from 382 s to about 313 s (-18%), and short prompt turns at 135K from 0.90 s to 0.77 s. Next-token distributions stayed inside run-to-run noise. Four commits, kept, in production since 2026-09-26.

## Hypothesis

Prompt processing on this model is bound by the MoE (mixture-of-experts) matmul path. For a batch of more than 8 tokens, the SYCL `MUL_MAT_ID` op copies the expert ids to the host, waits for the queue, and runs one oneDNN GEMM (general matrix multiply) per expert: about 500 per call, 144 calls per micro-batch ([finding](../../../findings/llama-cpp-sycl-moe-prefill-host-sync.md)). On build `993baf141` the three MoE matmuls took 50% of the listed op time of prompt micro-batches at about 100-120K context (gate 17.8%, down 17.5%, up 14.7%, 7.3-8.8 ms per call; [raw op profile](../benchmarks/raw/2026-09-26-prefill-op-profile.csv)).

Upstream PR [#29245](https://github.com/ggml-org/llama.cpp/pull/29245) replaces the loop with one XMX (Xe Matrix Extensions) kernel launch that covers every expert and dequantizes the weights inside the matrix tiles. It only decodes IQ (i-quant) formats. This model keeps its experts in Q4_K and Q5_K (gate, up) and Q5_1 and Q8_0 (down), so as posted it would not engage. Adding decoders for those formats should cut each MoE call several-fold. The upstream author reports +34% prompt processing on Qwen3.8-Flash-Next `UD-IQ4_XS` ([PR description](https://github.com/ggml-org/llama.cpp/pull/29245)) and +51.5% pp512 on Qwen3-30B-A3B `UD-IQ3_XXS` on one Arc Pro B60 (commit message of 0012); UNVERIFIED here. With MoE at half of a micro-batch, a cold read could shrink by up to a third (estimate).

## Change

Four commits on top of `2c88bdf19`. They are the last kernel changes in the frozen production build `20260926-2b84213a4`.

| Step | Commit | Patch | Change |
|---|---|---|---|
| 1 | `020bf5940` | [0012](../configs/patches/0012-upstream-pr29245-sycl-grouped-moe-xmx-gemm.diff) | Upstream #29245 at `950f1c4aa`, cherry-picked. One launch per `MUL_MAT_ID` node. The host lays out tiles of 32 routed rows from the per-expert row offsets it already has; the weights are dequantized inside the XMX A tiles; the activations are packed once into VNNI (Vector Neural Network Instructions layout) f16. `GGML_SYCL_XMX_GATHER_TYPES` selects formats as a bitmask, bits 0-8 for the nine IQ formats. IQ formats only, so it does not engage on this model. |
| 2 | `8793154f8` | [0013](../configs/patches/0013-sycl-grouped-moe-gemm-q4k-q5k-q51-q80.diff) | A-stage decoders for Q4_K, Q5_K, Q5_1 and Q8_0. Q4_K and Q5_K get two row views: the stored block layout and the per-expert SoA (structure-of-arrays) layout that the decode path's reorder leaves them in. The grouped path calls `opt_for_reorder_id` too, so it meets one layout once a prompt has run. Bits 9-12 of `GGML_SYCL_XMX_GATHER_TYPES`. 26 new `test-backend-ops` cases. |
| 3 | `517eb833d` | [0014](../configs/patches/0014-sycl-grouped-moe-gemm-vector-loads.diff) | Vector loads in the new A stages. Before, a lane read the quant bytes of a k step one byte at a time (32 to 64 loads per lane per step); a sub-group's 16 lanes read 16 different rows, so every load was a gather. Now Q4_K and Q5_K read them as two 16-byte loads (plus two for the Q5_K high bits), Q5_1 as two 8-byte loads and one 4-byte load, Q8_0 in 2-byte pairs. |
| 4 | `85b26781b` | [0015](../configs/patches/0015-sycl-grouped-moe-gemm-in-place-io.diff) | The B pack reads each routed row straight from `src1` through the row mapping and moves a 32 x 64 block through local memory; the GEMM writes each routed row straight into `dst`. The contiguous gather and scatter buffers and their copy kernels ("157 MB written for the gate/up gather, 157 MB read and written for the down scatter", commit message) are only set up for the per-expert loop. |

The dispatch after 0015 (`ggml-sycl.cpp`, from patch 0015):

```cpp
// one launch for every expert, reading src1 rows and writing dst rows in place
bool grouped = false;
if (ggml_is_contiguous(src0) && src1->type == GGML_TYPE_F32 &&
    dst->type == GGML_TYPE_F32 && dst->op_params[0] == GGML_PREC_DEFAULT &&
    nb10 == sizeof(float) && nb0 == sizeof(float)) {
    // the decode path reorders Q4_K/Q5_K experts on first use; doing it here too means the
    // grouped kernel only ever meets that layout once a prompt has run
    opt_for_reorder_id(&ctx, src0);
    ...
    grouped = ggml_sycl_grouped_dequant_gemm_f16(src0->type, src0_original, nb02, src0_reordered,
                                                 src1_original, ne11, nb11, nb12, dst_original, nb1, nb2,
                                                 dev_row_mapping.get(), expert_row_offsets.data(), ...);
}
if (grouped) {
    return;
}
```

The expert ids still go to the host once per call (`stream->memcpy(ids_host...)` then `stream->wait()`, unchanged from stock): the host counts rows per expert and builds the tile schedule. Patch 0012's comment says why the wait stays: the grouped GEMM enqueues an async copy out of the host-side tile schedule, and the next node must not overwrite it while the device reads it.

Runtime switch, `GGML_SYCL_XMX_GATHER_TYPES` (decimal bitmask, default all bits set):

| Bits | Formats | Source |
|---|---|---|
| 1, 2, 4, 8, 16, 32, 64, 128, 256 | IQ4_NL, IQ3_S, IQ4_XS, IQ3_XXS, IQ2_XXS, IQ2_XS, IQ2_S, IQ1_S, IQ1_M | 0012 (upstream) |
| 512, 1024, 2048, 4096 | Q4_K, Q5_K, Q5_1, Q8_0; grouped `MUL_MAT_ID` path only, plain `MUL_MAT` keeps the IQ list | 0013 |
| 0 | every path off: the per-expert loop, the baseline to measure against | 0012 |

Precision: the XMX kernel takes f16 A and B tiles and accumulates in f32 (`joint_matrix<float, use::accumulator>` in patch 0012). Upstream's message notes a precision trade against a default build, where the per-expert oneDNN GEMM computes in f32. Our build has `GGML_SYCL_F16=ON`, where the loop already dequantized experts to f16 for oneDNN ([prefill note](../research/2026-09-25-prefill-bottleneck.md)). The KL (Kullback-Leibler divergence) probe below is the check.

## Environment

| Component | Version / value |
|---|---|
| Engine and image (tag and digest) | llama.cpp, no container. PR #28243 at `6fcaa16` + #28931 + 0002-0004 + #28796 + 0006-0011, then the steps above. Kernel tests ran the development tree at `8793154f8`, `517eb833d` and `85b26781b`, the lcbench windows at the last two; the frozen production build `20260926-2b84213a4` includes all four commits plus 0016 and 0017 |
| Build flags | `GGML_SYCL=ON GGML_SYCL_TARGET=INTEL GGML_SYCL_F16=ON GGML_SYCL_DNN=ON GGML_SYCL_GRAPH=ON GGML_SYCL_SUPPORT_LEVEL_ZERO_API=ON GGML_NATIVE=ON`, Release, icx/icpx from oneAPI 2026.1 |
| compute-runtime / IGC | 26.35.39758.11 / IGC 2.41.9; Level Zero loader 1.32.0 (as recorded for the [2026-09-25 production benchmark](../benchmarks/2026-09-25-production-build.md); no driver or runtime change is recorded between the two dates; not re-read on 2026-09-26) |
| Kernel / `xe` driver / GuC firmware | 7.0.0-34-generic / in-tree `xe` / GuC 70.58.0 (same source and caveat) |
| oneCCL | not used |
| PyTorch XPU | not used |
| Power cap per card | 230 W |
| Tensor / pipeline parallel | layer split across 4 cards, `-ub 1536 -ts 12,13,13,11 -fit off`, MTP (multi-token prediction) draft n-max 3 on `SYCL3`, f16 KV (key-value) cache; kernel tests on `SYCL0` alone with the service stopped |
| GPU clock floor | 2800 MHz while the service runs; not recorded for the kernel-test windows |
| Other switches | `GGML_SYCL_SPARSE_FA=1`, pooled QSA (Qwen sparse attention) key cache, draft QSA and score expansion by broadcast on (patches 0006-0011) |

## Procedure

1. Build in the development worktree. Stop the service under a hold so the recovery unit does not switch profiles. Confirm the cards are empty.
2. `test-backend-ops` on `SYCL0` against the CPU backend, filter `type_a=(q4_K|q5_K|q5_1|q8_0),.*n_mats=(512|4),n_used=(10|4),`: correctness with `GGML_SYCL_ENABLE_OPT=1` (Q4_K/Q5_K experts reordered) and `=0` (stored layout), then with `GGML_SYCL_XMX_GATHER_TYPES=0` (the loop). Then `perf` for the four one-micro-batch shapes (512 experts, 10 used, 1536 tokens), loop and grouped. The exact commands are in the [benchmark](../benchmarks/2026-09-26-mul-mat-id-kernel-perf.md).
3. Point the service at the candidate tree and restart it with the production slot held (`slot-cache.sh hold`, a 41,364-token save; [raw](../benchmarks/raw/2026-09-26-slot-cache-timings.csv)). Run [`tools/lcbench.py`](../../../tools/lcbench.py): a cold read of the 134,866-token C++ context, then 6 turns of 256 greedy tokens. Summarise with [`tools/lcsum.py`](../../../tools/lcsum.py).
4. Teacher-forced next-token check with [`tools/klprobe.py`](../../../tools/klprobe.py): 128 positions after the same 134,840-token prefix, top-20 log-probabilities, compared with `kl-poolA` (a run of build `20260925-f47a6a5f3`, the reference every later build is compared with) and with the previous step.
5. Restore the production build and release the slot hold.

## Results

**Kernel**, one `-ub 1536` micro-batch, 512 experts, 10 used, one B70 ([raw](../benchmarks/raw/2026-09-26-mul-mat-id-kernel-perf.csv); N=1 `test-backend-ops perf` pass per cell, each the mean over the tool's own repetitions):

| Expert type (m, n, k) | Per-expert loop | 0013 grouped | 0014 vector loads | 0015 in-place I/O | Loop to final |
|---|---|---|---|---|---|
| Q4_K gate/up (640, 1536, 2560) | 11,314.47 us | 5,561.61 | 4,858.96 | 3,927.61 | 2.9x |
| Q5_K gate/up (640, 1536, 2560) | 11,943.55 | 7,807.55 | 5,305.17 | 4,436.68 | 2.7x |
| Q5_1 down (2560, 1536, 640) | 11,017.13 | 4,673.80 | 4,442.04 | 3,775.97 | 2.9x |
| Q8_0 down (2560, 1536, 640) | 10,935.39 | 7,970.92 | 7,981.60 | 7,395.54 | 1.5x |

Every cell is 50.33 GFLOP, so the final column is 12.8 / 11.3 / 13.3 / 6.8 TFLOPS achieved. The 0015 cells were measured in a foreground window whose stdout was not archived; the CSV row is the record, and the 0015 commit message quotes the same values rounded ("q4_K 4.9 -> 3.9 ms, q5_K 5.3 -> 4.4, q5_1 4.4 -> 3.8, q8_0 8.0 -> 7.4"). Q8_0 gained least at every step; its A stage still issues 16 loads per lane per k step after 0014, against 2 for Q4_K ([cost breakdown](../research/2026-09-26-grouped-moe-gemm-cost-breakdown.md)).

**Model**, agent session at 135K, 6 turns:

| Metric | `2c88bdf19` (loop) | `517eb833d` (0012-0014) | `85b26781b` (+0015) |
|---|---|---|---|
| Cold read of 134,866 tokens, lcbench wall | 382.49 s | 312.58 s | 312.71 s |
| Same cold reads, server end-of-prompt progress line ([raw](../benchmarks/raw/2026-09-26-cold-read-progress.csv), runs started 23:50, 00:33, 01:04 and 01:17 UTC) | 381.74 s | 311.84 s | 311.98 s, 313.34 s |
| Prompt speed over the cold read | 353 tok/s | 432 tok/s | 432 / 430 tok/s |
| Decode, median of 6 turns (min-max) | 43.9 tok/s (39.9-54.2) | 48.3 (40.8-52.7) | 48.7 (36.3-51.5) |
| MTP step time | 67.1 ms | 68.4 ms | 68.1 ms |
| Draft acceptance | 66.0% (1014/1537) | 72.2% (1045/1447) | 70.7% (1036/1466) |
| Per-turn prompt time, median (max) | 0.91 s (2.34) | 0.90 s (1.69) | 0.77 s (1.67) |

The lcbench lines (wall times, decode, step time, acceptance, prompt medians) are transcribed from the three test-window outputs in [raw/2026-09-26-cold-read-windows.csv](../benchmarks/raw/2026-09-26-cold-read-windows.csv), rows `blbgwttcj`, `bhc79ftn4` and `b2pfxuo2i`. The archived record is the server's own progress lines in the cold-read CSV: after the grouped kernel, three cold reads of the same context finished in 311.84-313.34 s (N=3, one of them the 01:17 profiling run), against 381.74 s before it; a fourth read, the 01:26 profiling run on `85b26781b`, ended at 294.94 s. Which build each journal run used is inferred from the run time against the commit sequence (0015 was committed between the 00:33 and the 01:04 run).

- The cold read gain is -18%. The kernel gain is larger (2.7-2.9x per call) because MoE was half of a micro-batch's GPU time, and because the context grows from zero during the read, so attention and the indexer take a growing share.
- Prompt turns of about 60 tokens improved only with 0015 (0.90 to 0.77 s). What 0015 removed at that batch size is two copy kernels and two pool allocations per call, 288 launches per micro-batch; whether that or the pack path explains the 0.13 s was not isolated (UNVERIFIED).
- Decode medians differ by acceptance on different generated text (66.0-72.2%); the step times (67.1-68.4 ms) are the same within noise. The grouped path does not run during decode (1-token and 2-8-token batches take the fused MMVQ (quantized matrix-vector) path).
- Op profile on `85b26781b`, prompt micro-batches at about 100-120K ([raw](../benchmarks/raw/2026-09-26-prefill-op-profile.csv)): the three MoE matmuls fell to 35.5% of the listed op time (down 12.1%, gate 11.8%, up 11.6%; 3.42-3.57 ms per call, averaged over 1536- and 512-token micro-batches), and dense prompt attention became the largest op at 32.8%. The profiler waits per op, so use the shares.

## Correctness

- `test-backend-ops -o MUL_MAT_ID` (SYCL0 against CPU) on the filter above: 60/60 pass with `GGML_SYCL_ENABLE_OPT=1` (reordered Q4_K/Q5_K experts) and 60/60 with `=0` (stored layout), at `8793154f8` and again at `517eb833d`; 60/60 with the loop (`GGML_SYCL_XMX_GATHER_TYPES=0`) at `8793154f8`. At `85b26781b` the commit message reports "test-backend-ops 60/60 with and without the reorder". The 60 cases are 26 added by 0013 (for each of the four types: the model's gate/up shape with 9 and 64 tokens, and 4 experts x 4 used at 32, 33, 64 and 65 tokens, which cover one full tile, a one-row second tile, two full tiles and the fall-back to the loop above the 64-column gate; plus the down shape for Q5_1 and Q8_0) and 34 pre-existing upstream Q4_K and Q8_0 cases the filter also matches (1 to 129 tokens, with and without broadcast). Pass counts are from the test-window outputs (source: session log, not archived).
- Teacher-forced KL, 128 positions after a 134,840-token prefix ([raw](../benchmarks/raw/2026-09-26-klprobe-comparisons.csv)):

| Comparison | Top-1 agreement | KL mean | KL median | KL p95 | KL max |
|---|---|---|---|---|---|
| `kl-mmid` (`517eb833d`) vs `kl-poolA` (reference build) | 118/128 | 0.11805 | 0.00242 | 0.31631 | 4.95057 |
| `kl-mmid18` (`85b26781b`) vs `kl-poolA` | 117/128 | 0.11847 | 0.00194 | 0.64569 | 3.46509 |
| `kl-mmid18` vs `kl-mmid` | 118/128 | 0.11027 | 0.00195 | 0.51959 | 4.36541 |
| Previous step: `kl-ident` (`2c88bdf19`) vs `kl-poolA` | 116/128 | 0.10726 | 0.00208 | 0.37733 | 4.22825 |
| Noise floor: `kl-poolC` vs `kl-poolA`, two runs of one build | 116/128 | 0.09916 | 0.00264 | 0.39624 | 4.86846 |

The grouped builds differ from the reference by no more than two runs of one build differ from each other. Greedy output is not token-identical run to run on this model even without the change (MTP batching changes the summation order; the QSA top-k breaks ties in an unspecified order), so the KL probe is the gate, as in the earlier experiments.

## Decision

Kept. Each MoE prompt call is 2.7-2.9x faster for the K-quant and Q5_1 experts and 1.5x for Q8_0; the 135K cold read fell from 382 s to about 313 s and short prompt turns from 0.90 to 0.77 s; next-token distributions are inside noise. In production since 2026-09-26 as part of build `20260926-2b84213a4`. `GGML_SYCL_XMX_GATHER_TYPES=0` restores the loop without a rebuild.

## Follow-ups

- [x] The MTP head's input projection (`eh_proj`) ran one product per token (115 ms per 1536-token micro-batch, 8.4% of listed op time on `85b26781b`); fixed in the next commit, `a890bf8b0` (patch 0016), which took the cold read to about 284 s ([raw](../benchmarks/raw/2026-09-26-cold-read-progress.csv), run started 01:36 UTC: 283.10 s). Documented separately.
- [ ] Share the routing work and the B pack between the gate and up calls of a layer: the same ids and the same activations are processed twice. About 2% of a cold read (estimate, session log); see the [cost breakdown](../research/2026-09-26-grouped-moe-gemm-cost-breakdown.md).
- [ ] More kernel work (the A-stage loads, fewer K-splits, wider tiles): about 5-8% of a cold read (estimate, session log).
- [ ] Q8_0 gets the smallest gain; a 32-byte A-stage load like Q4_K's is the obvious next step. Only 5 of the 48 `down_exps` are Q8_0 (43 are Q5_1, [prefill note](../research/2026-09-25-prefill-bottleneck.md)), so the model-level effect is small.
- [ ] Upstream #29245 was still open on 2026-09-26 (head `950f1c4aa`, updated 2026-09-24). Our decoders sit on top of it and would need rebasing if its layout or dispatch changes.
