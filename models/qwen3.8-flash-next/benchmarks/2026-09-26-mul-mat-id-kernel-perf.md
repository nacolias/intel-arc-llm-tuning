# Grouped MoE GEMM (`MUL_MAT_ID`): test-backend-ops perf per kernel variant

| | |
|---|---|
| **Date** | 2026-09-26 |
| **Model / checkpoint** | none loaded: synthetic `MUL_MAT_ID` cases with the expert shapes of Qwen3.8-Flash-Next `UD-Q4_K_XL` (512 experts, 10 used; gate/up 640 x 2560, down 2560 x 640) and random routing |
| **Host** | [quad-b70-5800x-pex88096](../../../hardware/hosts/quad-b70-5800x-pex88096.md), one card (`SYCL0`), service stopped |
| **Config** | not a server run; `test-backend-ops` from the development tree at each commit below |
| **Experiment** | [grouped MoE XMX GEMM](../experiments/2026-09-26-grouped-moe-xmx-gemm.md) |
| **Raw data** | [raw/2026-09-26-mul-mat-id-kernel-perf.csv](raw/2026-09-26-mul-mat-id-kernel-perf.csv), [raw/2026-09-26-mul-mat-id-kernel-diag.csv](raw/2026-09-26-mul-mat-id-kernel-diag.csv) |

## Result

One MoE (mixture-of-experts) expert matmul over a 1536-token micro-batch takes 3.93 ms for Q4_K, 4.44 ms for Q5_K, 3.78 ms for Q5_1 and 7.40 ms for Q8_0 with the grouped XMX (Xe Matrix Extensions) GEMM (general matrix multiply) of patches 0012-0015, against 10.9-11.9 ms with the per-expert oneDNN loop: 2.9x, 2.7x, 2.9x and 1.5x. Each call is 50.33 GFLOP, so the grouped kernel reaches 12.8, 11.3, 13.3 and 6.8 TFLOPS. A diagnostic pass attributes the Q4_K time to weight dequantization (1.31 ms), the B pack (0.57 ms), work outside the kernels (0.53 ms), XMX multiply-add (0.37 ms), B tile loads (0.32 ms) and 0.95 ms of tile stores, K-split reduction, barriers and output writes.

## Environment

| Component | Version / value |
|---|---|
| Engine and image (tag and digest) | llama.cpp, no container. PR #28243 at `6fcaa16` + our patch series through 0011, then: `8793154f8` (0012 + 0013, "grouped v1"), `517eb833d` (+ 0014, vector loads), `85b26781b` (+ 0015, in-place I/O). Loop baseline: the same binary with `GGML_SYCL_XMX_GATHER_TYPES=0` |
| Build flags | `GGML_SYCL=ON GGML_SYCL_TARGET=INTEL GGML_SYCL_F16=ON GGML_SYCL_DNN=ON GGML_SYCL_GRAPH=ON GGML_SYCL_SUPPORT_LEVEL_ZERO_API=ON GGML_NATIVE=ON`, Release, icx/icpx from oneAPI 2026.1 |
| compute-runtime / IGC | 26.35.39758.11 / IGC 2.41.9; Level Zero loader 1.32.0 (as recorded for the [2026-09-25 production benchmark](2026-09-25-production-build.md); not re-read on 2026-09-26) |
| Kernel / `xe` driver / GuC firmware | 7.0.0-34-generic / in-tree `xe` / GuC 70.58.0 (same caveat) |
| oneCCL | not used |
| PyTorch XPU | not used |
| Power cap per card | 230 W |
| KV cache dtype / block size | not applicable |
| Speculative decoding | not applicable |
| Graph / compile mode | eager; `ZES_ENABLE_SYSMAN=1 UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1` |
| GPU clock floor | not recorded for these windows: the 2800 MHz floor is applied while the service runs, and the service was stopped (UNVERIFIED whether it held) |

## Workload

| | |
|---|---|
| Workload set | `test-backend-ops perf -o MUL_MAT_ID`; not one of the shared [workloads](../../../benchmarks/workloads/) |
| Cases | `type_a` in q4_K, q5_K (m=640, k=2560, the gate/up shape) and q5_1, q8_0 (m=2560, k=640, the down shape); `type_b` f32; `n_mats` 512; `n_used` 10; `n` 1536 tokens, one `-ub 1536` prompt micro-batch. Expert ids are random, so nearly every expert receives rows (about 30 on average) |
| Output | `dst` f32, 1536 x 10 rows |
| Warm-up | the tool's own |
| Repetitions | one perf pass per variant and type (N=1); each figure is the tool's mean over its own repetitions |

Commands (from the test-window script; oneAPI 2026.1 sourced first):

```bash
export ZES_ENABLE_SYSMAN=1 UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1
TBO=/path/to/llama.cpp/build/bin/test-backend-ops
P='type_a=(q4_K|q5_K|q5_1|q8_0),.*n_mats=(512|4),n_used=(10|4),'
PP='n_mats=512,n_used=10,b=0,m=[0-9]+,n=1536,'

# correctness, grouped path: reordered Q4_K/Q5_K experts, then the stored layout
GGML_SYCL_ENABLE_OPT=1 $TBO test -b SYCL0 -o MUL_MAT_ID -p "$P"
GGML_SYCL_ENABLE_OPT=0 $TBO test -b SYCL0 -o MUL_MAT_ID -p "$P"
# correctness and perf, per-expert loop (baseline)
GGML_SYCL_XMX_GATHER_TYPES=0 $TBO test -b SYCL0 -o MUL_MAT_ID -p "$P"
GGML_SYCL_XMX_GATHER_TYPES=0 $TBO perf -b SYCL0 -o MUL_MAT_ID -p "$PP"
# perf, grouped path
$TBO perf -b SYCL0 -o MUL_MAT_ID -p "$PP"
```

The diagnostic series ran the same `perf` command under `GGML_SYCL_FG_DIAG=<bits>`, a temporary switch that skipped parts of the grouped kernel. It is not in the patch series; the results of those runs are wrong by design and only the times are used.

## Results

Per variant, microseconds per call ([raw](raw/2026-09-26-mul-mat-id-kernel-perf.csv)). The loop baseline and the 0013 and 0014 columns come from archived test windows; the 0015 column was measured in a foreground window whose stdout was not archived, and its commit message quotes the same values rounded to 0.1 ms.

| Type (m, n, k) | Per-expert loop | 0013 grouped v1 | 0014 vector loads | 0015 in-place I/O | Loop to final | Final TFLOPS |
|---|---|---|---|---|---|---|
| q4_K (640, 1536, 2560) | 11,314.47 | 5,561.61 | 4,858.96 | 3,927.61 | 2.88x | 12.81 |
| q5_K (640, 1536, 2560) | 11,943.55 | 7,807.55 | 5,305.17 | 4,436.68 | 2.69x | 11.34 |
| q5_1 (2560, 1536, 640) | 11,017.13 | 4,673.80 | 4,442.04 | 3,775.97 | 2.92x | 13.33 |
| q8_0 (2560, 1536, 640) | 10,935.39 | 7,970.92 | 7,981.60 | 7,395.54 | 1.48x | 6.81 |

Step by step: 0013 took 27-58% off the loop (least for q8_0, most for q5_1); 0014 took a further 13% off q4_K and 32% off q5_K, 5% off q5_1 and nothing off q8_0; 0015 took 15-19% off the K-quant and q5_1 shapes and 7% off q8_0. The loop is flat at about 11 ms for all four types because it is launch-bound (one oneDNN GEMM per expert), not dequantization-bound.

**Diagnostic decomposition** ([raw](raw/2026-09-26-mul-mat-id-kernel-diag.csv)). Each row skips a part of the kernel; the difference between rows attributes time to the skipped part. Series 1 ran on `517eb833d` (bit 1 skips the GEMM launch, bit 2 the B pack; contiguous copies still present). Series 2 ran on `85b26781b` (bit 4 skips A-stage dequantization, 8 the XMX `joint_matrix_mad`, 16 the B tile loads, 32 the GEMM launch, 64 the B pack). Skipping a stage also changes what the compiler and the scheduler do around it, so these are attributions by subtraction (estimate), not measured stage times.

Measured times, microseconds:

| Series | Build | Bits | Skipped | q4_K gate | q5_K gate | q5_1 down | q8_0 down |
|---|---|---|---|---|---|---|---|
| 1 | `517eb833d` | 0 | nothing | 4,872.93 | 5,329.85 | 4,452.86 | 7,973.57 |
| 1 | `517eb833d` | 1 | GEMM launch | 2,061.84 | 2,034.91 | 1,450.17 | 1,449.52 |
| 1 | `517eb833d` | 3 | GEMM launch and B pack | 1,265.98 | 1,254.40 | 1,278.44 | 1,278.97 |
| 2 | `85b26781b` | 0 | nothing | 4,045.93 | 4,542.23 | 3,869.07 | 7,436.62 |
| 2 | `85b26781b` | 4 | A-stage dequantization | 2,733.94 | 2,735.23 | 2,622.31 | 2,761.55 |
| 2 | `85b26781b` | 8 | XMX mad | 3,677.70 | 4,251.16 | 3,554.83 | 7,233.56 |
| 2 | `85b26781b` | 12 | A stage and mad | 2,376.45 | 2,459.70 | 2,338.58 | 2,591.39 |
| 2 | `85b26781b` | 28 | A stage, mad and B tile loads | 2,055.94 | 2,087.85 | 2,190.39 | 2,379.29 |
| 2 | `85b26781b` | 32 | GEMM launch (pack and id sort remain) | 1,093.28 | 1,100.38 | 632.82 | 632.00 |
| 2 | `85b26781b` | 96 | GEMM launch and B pack (id sort and host sync remain) | 526.64 | 541.51 | 542.24 | 557.25 |
| 2 | `85b26781b` | loop | grouped path off (`GGML_SYCL_XMX_GATHER_TYPES=0`) | 11,453.18 | 11,896.22 | 10,811.17 | 10,709.78 |

Attribution by subtraction on `85b26781b`, microseconds (estimate):

| Part | How derived | q4_K gate | q5_K gate | q5_1 down | q8_0 down |
|---|---|---|---|---|---|
| A-stage dequantization into the XMX tiles | row 0 minus row 4 | 1,312 | 1,807 | 1,247 | 4,675 |
| XMX `joint_matrix_mad` | row 0 minus row 8 | 368 | 291 | 314 | 203 |
| B tile loads | row 12 minus row 28 | 321 | 372 | 148 | 212 |
| Rest of the GEMM kernel: tile stores, K-split reduce, barriers, `dst` writes | (row 0 minus row 32) minus the three above | 952 | 972 | 1,527 | 1,714 |
| B pack (routed `src1` rows to VNNI (Vector Neural Network Instructions layout) f16) | row 32 minus row 96 | 567 | 559 | 91 | 75 |
| Outside the kernels: ids copy to host, host wait, counting sort, tile schedule copy | row 96 | 527 | 542 | 542 | 557 |
| Total | row 0 | 4,046 | 4,542 | 3,869 | 7,437 |

On `517eb833d` the part outside the GEMM launch was 2,062 us for q4_K, of which the B pack was 796 us and the remainder 1,266 us. On `85b26781b` those are 567 and 527 us: the in-place changes of 0015 removed about 740 us of contiguous copies and 230 us of pack time per q4_K call (estimate across two builds, two runs).

## Observations

- The XMX multiply-add is 3-9% of each call. The kernels are bound by loads and dequantization, not by the matrix units. See the [cost breakdown](../research/2026-09-26-grouped-moe-gemm-cost-breakdown.md).
- q8_0's A stage is 4.7 ms of 7.4. After 0014 it still issues 16 two-byte loads per lane per 32-weight k step, against two 16-byte loads for q4_K; its 0014 gain was nil (7,970.92 to 7,981.60 us).
- The down shape (m=2560, k=640) has four times as many 16-row output tiles as the gate shape and a quarter of the K depth, and its "rest of the kernel" share is larger (1.5-1.7 ms against 0.95-0.97): tile stores and the K-split reduction weigh more when K is short (estimate; not isolated).
- The B pack is small for the down shape (75-91 us) because its `src1` rows are 640 wide instead of 2560.
- The full-kernel times differ by 3% between the perf run and the diagnostic run of the same build (q4_K 3,927.61 against 4,045.93 us); treat that as the run-to-run spread of this test.
- In the server the same calls average 3.42-3.57 ms per call at about 100-120K context under the op profiler, which mixes 1536- and 512-token micro-batches ([raw](raw/2026-09-26-prefill-op-profile.csv)).
- The cards were idle before each window; nothing else was resident (session log).
