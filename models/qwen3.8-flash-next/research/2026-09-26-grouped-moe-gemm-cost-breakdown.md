# Where the grouped MoE GEMM's 4 ms goes: cost breakdown and remaining levers

| | |
|---|---|
| **Date read** | 2026-09-26 |
| **Source** | our own measurements on [quad-b70-5800x-pex88096](../../../hardware/hosts/quad-b70-5800x-pex88096.md) ([kernel perf](../benchmarks/raw/2026-09-26-mul-mat-id-kernel-perf.csv), [kernel diag](../benchmarks/raw/2026-09-26-mul-mat-id-kernel-diag.csv), [prefill op profile](../benchmarks/raw/2026-09-26-prefill-op-profile.csv)) and source reading of patches [0012](../configs/patches/0012-upstream-pr29245-sycl-grouped-moe-xmx-gemm.diff) (upstream [#29245](https://github.com/ggml-org/llama.cpp/pull/29245)), [0013](../configs/patches/0013-sycl-grouped-moe-gemm-q4k-q5k-q51-q80.diff), [0014](../configs/patches/0014-sycl-grouped-moe-gemm-vector-loads.diff) and [0015](../configs/patches/0015-sycl-grouped-moe-gemm-in-place-io.diff) |
| **Author / org** | this repository (AI-assisted analysis) |
| **Type** | profiling analysis and source-code review |
| **Applies to** | the grouped `MUL_MAT_ID` path of llama.cpp SYCL (PR #29245 plus our decoders) on Arc Pro B70, with the expert shapes of Qwen3.8-Flash-Next `UD-Q4_K_XL` (512 experts, 10 used; gate/up 640 x 2560, down 2560 x 640) |

## Summary

The grouped MoE (mixture-of-experts) GEMM (general matrix multiply) is bound by loads and dequantization, not by the XMX (Xe Matrix Extensions) units. Of 4.05 ms for one Q4_K gate call over a 1536-token micro-batch, about 1.31 ms dequantizes weights into the A tiles, 0.37 ms is the XMX multiply-add, 0.32 ms loads B tiles, 0.96 ms is the rest of the kernel (tile stores, the K-split reduction, barriers and the output writes), 0.57 ms packs the activations and 0.53 ms is spent outside the kernels (expert ids to the host, the host wait, the counting sort and the tile schedule copy). The call is 50.33 GFLOP, so 4.05 ms is 12.4 TFLOPS achieved; the multiply-add stage alone is about 9% of the time. The next levers are sharing the routing and the activation pack between the gate and up calls, wider loads for Q8_0, and fewer K-splits or wider tiles. None is measured; the estimates below are labelled.

## Measurements

**Per call, one 1536-token micro-batch, one B70** ([raw](../benchmarks/raw/2026-09-26-mul-mat-id-kernel-perf.csv); the [benchmark](../benchmarks/2026-09-26-mul-mat-id-kernel-perf.md) has every step):

| Type | Per-expert loop | Grouped, build `85b26781b` | Achieved TFLOPS |
|---|---|---|---|
| Q4_K gate/up | 11,314.47 us | 3,927.61 us | 12.81 |
| Q5_K gate/up | 11,943.55 | 4,436.68 | 11.34 |
| Q5_1 down | 11,017.13 | 3,775.97 | 13.33 |
| Q8_0 down | 10,935.39 | 7,395.54 | 6.81 |

**Attribution by subtraction** on `85b26781b` ([raw](../benchmarks/raw/2026-09-26-mul-mat-id-kernel-diag.csv)). A temporary `GGML_SYCL_FG_DIAG` bitmask skipped one part of the kernel at a time; the results of those runs are wrong by design. Skipping a stage changes what the compiler and scheduler do around it, so treat each figure as an estimate.

| Part | Q4_K gate | Q5_K gate | Q5_1 down | Q8_0 down |
|---|---|---|---|---|
| A stage: dequantize 16 weight rows into the XMX A tile | 1,312 us (32%) | 1,807 (40%) | 1,247 (32%) | 4,675 (63%) |
| XMX `joint_matrix_mad` | 368 (9%) | 291 (6%) | 314 (8%) | 203 (3%) |
| B tile loads from the packed activations | 321 (8%) | 372 (8%) | 148 (4%) | 212 (3%) |
| Rest of the GEMM kernel: `joint_matrix_store`, K-split reduce, barriers, `dst` writes | 952 (24%) | 972 (21%) | 1,527 (39%) | 1,714 (23%) |
| B pack: routed `src1` rows to VNNI (Vector Neural Network Instructions layout) f16 | 567 (14%) | 559 (12%) | 91 (2%) | 75 (1%) |
| Outside the kernels: ids copy to host, wait, counting sort, tile schedule copy | 527 (13%) | 542 (12%) | 542 (14%) | 557 (7.5%) |
| Total (full kernel in the diagnostic run) | 4,046 | 4,542 | 3,869 | 7,437 |

The full-kernel time of the diagnostic run is 3% above the perf run of the same build (4,045.93 against 3,927.61 us for Q4_K); that is the spread of this test.

## Roofline arithmetic

- Work per call: 2 x 640 x 2560 x 1536 x 10 = 50.33 GFLOP for the gate/up shape, and the same for the down shape (2 x 2560 x 640 x 1536 x 10). `test-backend-ops` prints the same figure.
- Achieved: 50.33 GFLOP / 4.046 ms = 12.4 TFLOPS in the diagnostic run, 12.8 TFLOPS at 3.928 ms in the perf run (Q4_K); 6.8 TFLOPS for Q8_0.
- The B70's XMX f16 peak is not measured in this repository, and no vendor figure is used here, so no fraction of peak is stated. What the subtraction does show is that the multiply-add stage is 0.20-0.37 ms of 3.9-7.4 ms: the matrix units are idle for 90% or more of each call (estimate).
- Bytes per call, computed from the shapes. Expert weights, read once each because random routing gives every expert rows: Q4_K 512 x 640 x 2560 x 144/256 B = 472 MB; Q5_K 577 MB (176 B per 256 weights); Q5_1 512 x 2560 x 640 x 24/32 B = 629 MB; Q8_0 891 MB (34 B per 32 weights). Routed activation rows read by the B pack: 15,360 x 2560 x 4 B = 157 MB for gate/up (the "157 MB" of the 0015 commit message), 39 MB for down. Output written: 39 MB for gate/up, 157 MB for down. At 3.93 ms the Q4_K weight stream alone is 120 GB/s; the card's memory bandwidth is not measured in this repository, so no ratio is given.

## Why the A stage is load-bound

The kernel maps one lane to one weight row (`static_assert(FG_SG_ROWS == WARP_SIZE, "the A stage maps one lane to one row")`, patch 0012). A sub-group of 16 lanes therefore decodes 16 different rows at once, and every load instruction in the A stage is a gather across 16 rows. A k step is 32 weights (`FG_BK`). Loads per lane per k step, read from the decoders in 0013 (before) and 0014 (after):

| Type | Before 0014 (patch 0013) | After 0014 | Grouped time before, after |
|---|---|---|---|
| Q4_K | 32 one-byte loads of `qs` (both nibble halves of 32 bytes, one half used per step) | two 16-byte loads | 5.56 ms, 4.86 ms |
| Q5_K | 64 one-byte loads (32 `qs`, 32 `qh`) | four 16-byte loads | 7.81, 5.31 |
| Q5_1 | 16 one-byte loads of `qs` plus one 4-byte `qh` | two 8-byte loads plus one 4-byte | 4.67, 4.44 |
| Q8_0 | 32 one-byte loads | 16 two-byte loads | 7.97, 7.98 |

Three observations line up with load count rather than with bits per weight:

1. Before 0014 the grouped times ranked Q5_1 (17 loads) fastest, then Q4_K and Q8_0 (32), then Q5_K (64) slowest. The 0014 commit message says the same: "patch-16 times ranked by load count".
2. Vectorising Q5_K from 64 loads to 4 cut 2.5 ms; Q4_K from 32 to 2 cut 0.7 ms; Q8_0 from 32 to 16 cut nothing.
3. After 0014, the A stage is 4.7 ms for Q8_0 against 1.3 ms for Q4_K, and Q8_0 keeps eight times the loads per step.

The A stage also converts every weight to f16 with a multiply and a subtract, and for the K-quants decodes the 6-bit scale and min of each step. That arithmetic is common to all four and is not separated here.

## Where the rest goes

- **Outside the kernels (0.53 ms).** Unchanged from the stock loop: `stream->memcpy(ids_host, ...)` and `stream->wait()`, then the host counts rows per expert, builds the row mapping and the tile schedule, and copies both to the device (patch 0012; the comment on the wait says the grouped GEMM's async copy out of the host-side schedule is why the wait must stay). The wait also stops the host from launching ahead, a cost `test-backend-ops` does not see: 144 waits per micro-batch in the server.
- **B pack (0.57 ms for gate/up).** Since 0015 it reads each routed row straight from `src1` through the row mapping and moves a 32 x 64 block through local memory so reads and packed writes are both contiguous. The gate and up calls of one layer pack the same activations with the same routing.
- **Rest of the GEMM kernel (0.95-1.7 ms).** A work-group is `FG_KSPLIT` = 4 sub-groups that each cover a quarter of K for the same 16 x 32 output tile; each stores its accumulators to local memory (`FG_KSPLIT x 16 x 32` floats) and the work-group reduces them before writing `dst` (patch 0012). The down shape has four times as many output tiles as the gate shape (2560 / 16 against 640 / 16) at a quarter of the K depth, and its share here is largest (1.5-1.7 ms): the fixed per-tile cost dominates when K is short (estimate; not isolated).
- **Tiles and re-dequantization.** Tiles are 32 columns wide (`FG_BN`), and "every FG_BN columns dequantize A again" (comment in patch 0012). In the synthetic test every expert receives about 30 rows, so most experts are one tile and the weights are decoded once or twice. Real routing is skewed: a hot expert with 200 rows is seven tiles and decodes its weights seven times. The rows-per-expert histogram at 135K is not measured (UNVERIFIED how much this costs in the server).

## Levers

None of these is measured. Percentages of a cold read are estimates from the session that built the kernel; per-call figures are estimates from the subtraction table.

| Lever | What it removes | Size (estimate) | Cost |
|---|---|---|---|
| Share the routing between gate and up (and down): one ids copy, wait, sort and schedule per layer instead of three | up to 2 x 0.53 ms of the roughly 12 ms the three calls take per layer | about 2% of a cold read (session log) | graph or backend change: the three `MUL_MAT_ID` nodes of a layer share one `ids` tensor, but the backend sees them one at a time |
| Share the B pack between gate and up: same `src1`, same routing | 0.57 ms per layer | included in the row above if done together | keep the packed buffer alive across two nodes |
| Wider A-stage loads for Q8_0 (32 bytes as two 16-byte loads, as Q4_K does) | part of Q8_0's 4.7 ms A stage | Q8_0 toward Q5_1's 3.8 ms per call; only 5 of 48 `down_exps` are Q8_0 ([prefill note](2026-09-25-prefill-bottleneck.md)), so small at model level | one decoder |
| Fewer K-splits (2 instead of 4) or wider tiles (64 columns) | part of the 0.95-1.7 ms of stores, reduce and barriers; fewer A re-dequantizations for hot experts | bounded above by that 0.95-1.7 ms per call plus the unmeasured re-dequantization share | wider tiles waste columns on experts with few rows; fewer splits halve the sub-groups per tile unless tiles grow |
| More kernel work in total | the levers above together | about 5-8% of a cold read (session log) | several days |
| Device-side routing | the ids copy and host wait, and the serialisation of host and device | not estimated; design only ([prefill note](2026-09-25-prefill-bottleneck.md)) | a GPU counting sort and a device-built schedule |

For scale: on `85b26781b` the three MoE matmuls are 35.5% of the listed op time of prompt micro-batches at 90-110K (down 12.1%, gate 11.8%, up 11.6%; [raw](../benchmarks/raw/2026-09-26-prefill-op-profile.csv)), and MoE is a smaller share of a whole 135K cold read because the context, and with it attention, grows from zero. A further 2x on the kernel would therefore take well under 18% off a cold read (estimate).

## Claims to verify

| Claim | Reported number | Their setup | Status |
|---|---|---|---|
| #29245 speeds up MoE prompt processing | pp512 671.1 to 1016.9 tok/s (+51.5%), tg128 unchanged | one Arc Pro B60, Qwen3-30B-A3B UD-IQ3_XXS (commit message of 0012) | UNVERIFIED here; our experts are not IQ types |
| The kernel is "still about 3x slower than the hardware allows" | 3x | session log, 2026-09-26 | UNVERIFIED: no XMX peak measured here |
| The A stage is bound by load instructions | ranking and 0014 deltas above | our `test-backend-ops` runs | verified-here for the ranking; the mechanism is inferred from the code |
| Attribution by subtraction is accurate to within 10% | sum of parts equals the full run by construction | our diag runs | estimate; skipped stages change the surrounding schedule |

## Relevance

The grouped kernel moved the prompt bottleneck: MoE went from 50% to 35% of the listed op time of prompt micro-batches and dense prompt attention (32.8%) is now the largest op ([raw](../benchmarks/raw/2026-09-26-prefill-op-profile.csv)). The remaining MoE levers are worth a few percent each of a cold read; day-to-day agent turns are decode-bound and do not run this path (batches of 1 to 8 tokens take the fused MMVQ (quantized matrix-vector) path).

## Actions

- [ ] Measure the rows-per-expert histogram of real prompt micro-batches at 135K to size the tile lever.
- [ ] Try `FG_KSPLIT` 2 and `FG_BN` 64 variants under `test-backend-ops perf` on the four shapes; the down shape should move most.
- [ ] Two 16-byte loads for the Q8_0 A stage.
- [ ] Share the routing and the pack between the gate and up nodes of a layer.
- [ ] Measure the B70's XMX f16 rate with a plain f16 GEMM so the 12.8 TFLOPS can be placed.
