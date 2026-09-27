# Benchmarks

One file per measured run, named `YYYY-MM-DD-<slug>.md`. Start from [`templates/benchmark.md`](../../../templates/benchmark.md) and follow the shared [methodology](../../../benchmarks/README.md). Raw output goes in [`raw/`](raw/).

All runs are single-slot (`-np 1`), so there is no aggregate throughput; the c16 column of other models does not apply. Two workloads are used:

- [`tools/fnbench.py`](../../../tools/fnbench.py): a short technical-prose prompt (512 tokens decoded), a repeated paragraph at 9,749 and 39,119 tokens, and optionally 219,217 tokens of structured records. Greedy. The repeated and structured prompts flatter MTP (multi-token prediction) acceptance.
- [`tools/lcbench.py`](../../../tools/lcbench.py): an agent-like session on a fixed 134,862-token C++ source context, 6 follow-up turns of 256 greedy tokens. Code workload. Its first read of the context by a freshly started server is the "cold read".

Abbreviations in the tables: QSA Qwen sparse attention, FA flash attention, KL Kullback-Leibler divergence, MoE mixture of experts, XMX Xe Matrix Extensions, GEMM general matrix multiply, ubatch micro-batch.

| Date | Benchmark | Config | Single-stream | Aggregate (c16) | Notes |
|---|---|---|---|---|---|
| 2026-09-24 | [Baseline 262K + MTP, unpatched](2026-09-24-baseline-262k-mtp.md) | `-ub 1024 -ts 13,13,13,10`, MTP n-max 3, stock unsloth checkpoint | 33.8-36.9 short; 43.3 at ~10k; 33.4 at ~39k; 16.1 at 219K | n/a | busiest card 27,420 MiB; MTP off: 33.1-33.5 / 31.4 / 23.4 / 7.2 |
| 2026-09-25 | [Agent session at 135K](2026-09-25-agent-session-135k.md) | production flags; dense, sparse FA and pooled-cache builds | 22-23 dense; 31-35 sparse FA; 39.1-48.5 with the pooled cache | n/a | MTP step 135 to 73 ms; prompt turns of ~60 tokens about 1.1 s on every build |
| 2026-09-25 | [Production build `20260925-f47a6a5f3`](2026-09-25-production-build.md) | [llama-server-262k-mtp.sh](../configs/llama-server-262k-mtp.sh) defaults, clock floor | 48.6-50.5 short; 60.6-67.2 at ~10k; 53.6-55.2 at ~39k; 40.6 at 219K | n/a | prompt 577-645 tok/s at 10k-39k, 268.3 at 219K; busiest card 28,957 MiB |
| 2026-09-26 | [Cold 135K read by build](2026-09-26-cold-read-135k-by-build.md) | production flags (`-ub 1536 -ts 12,13,13,11`); dev trees from the `LLAMA_MTP_WINDOW` tests (later `9f41d810e`) to `a890bf8b0`, production `993baf141`, and `2b84213a4` with `-b 3072` | 39-49 at 135K (medians; they move with acceptance, not with the build) | n/a | cold read 455 s (N=6, 436.8-462.0) to 382, about 313 and 284 s (-38%), one run per build after the baseline; ~60-token prompt turns 0.92-0.96 to 0.76 s; every journal read of 2026-09-25 20:41 to 2026-09-26 02:02 UTC listed; profiler runs, the 21:01 UTC 403 s run and `-b 3072` excluded from the medians |
| 2026-09-26 | [Grouped MoE GEMM kernel perf](2026-09-26-mul-mat-id-kernel-perf.md) | `test-backend-ops perf -o MUL_MAT_ID` on `SYCL0`, 512 experts / 10 used / 1536 tokens; builds `8793154f8`, `517eb833d`, `85b26781b` against the per-expert loop | n/a (kernel test) | n/a | loop 10.9-11.9 ms per call; grouped 3.93 / 4.44 / 3.78 / 7.40 ms for Q4_K / Q5_K / Q5_1 / Q8_0 (12.8 / 11.3 / 13.3 / 6.8 TFLOPS on 50.33 GFLOP); of 4.05 ms for Q4_K, A-stage dequantization 1.31, B pack 0.57, outside the kernels 0.53, XMX multiply-add 0.37, B tile loads 0.32, rest of the kernel 0.96 (attribution by subtraction, estimate) |
| 2026-09-26 | [Prefill op profile, builds `993baf141` and `85b26781b`](2026-09-26-prefill-op-profile.md) | production flags plus `GGML_SYCL_OP_PROFILE` (patch 0004) during 135K cold reads, windows at about 100-120K context (estimate) | n/a (op shares) | n/a | serialized op time, shares of the 30 listed ops: MoE matmuls 50.2% to 35.5% (7.30-8.84 to 3.42-3.57 ms per call), dense prompt attention 17.1% to 32.8% (34.07 to 38.77 ms per call), `mtp_eh_proj` 115 ms per call on both builds (fixed later by patch 0016), indexer `CONT (permuted)` 13.3% removed by patch 0011 |

## Raw files

| File | Contents |
|---|---|
| [raw/2026-09-24-baseline-fnbench.csv](raw/2026-09-24-baseline-fnbench.csv) | every fnbench row of the 2026-09-24 bring-up runs (6 configs) |
| [raw/2026-09-24-baseline-memory.csv](raw/2026-09-24-baseline-memory.csv) | peak VRAM per card and driver-held host RAM for those runs |
| [raw/2026-09-24-decode-profile.csv](raw/2026-09-24-decode-profile.csv) | per-card engine busy %, clocks and server CPU during decode, unpatched build |
| [raw/2026-09-24-speed-work.csv](raw/2026-09-24-speed-work.csv) | SYCL graphs, fused `MUL_MAT_ID`, draft-length sweep and clock floor results |
| [raw/2026-09-24-refusal-smoke-probe.csv](raw/2026-09-24-refusal-smoke-probe.csv) | 10-prompt refusal smoke test, stock against abliterated checkpoint (counts only) |
| [raw/2026-09-25-agent-session-135k.csv](raw/2026-09-25-agent-session-135k.csv) | per-turn lcbench timings and draft counts for 9 runs |
| [raw/2026-09-25-op-profile-135k.csv](raw/2026-09-25-op-profile-135k.csv) | `GGML_SYCL_OP_PROFILE` op shares at 135K, dense and sparse builds |
| [raw/2026-09-25-klprobe.csv](raw/2026-09-25-klprobe.csv) | teacher-forced KL comparisons for sparse FA and the pooled cache |
| [raw/2026-09-25-sparse-fa-kernel.csv](raw/2026-09-25-sparse-fa-kernel.csv) | `test-backend-ops perf`: dense against sparse flash attention |
| [raw/2026-09-25-ubatch-compare.csv](raw/2026-09-25-ubatch-compare.csv) | `-ub 1024` against `-ub 1536` |
| [raw/2026-09-25-production-fnbench.csv](raw/2026-09-25-production-fnbench.csv) | fnbench summaries per build and session on 2026-09-25, including the pipeline-parallel test and the ASPM trial |
| [raw/2026-09-25-draft-qsa-fnbench.csv](raw/2026-09-25-draft-qsa-fnbench.csv) | fnbench short / ~10k / ~39k on the draft-QSA build (commit `0f23f62c8`), with the per-card GPU memory after load in the header |
| [raw/2026-09-25-pairprobe-draft-qsa.csv](raw/2026-09-25-pairprobe-draft-qsa.csv) | [`tools/pairprobe.py`](../../../tools/pairprobe.py) paired A/B of draft QSA against the dense draft at 135K: 12 turns x 192 tokens per arm, per-turn draft counts and decode tok/s, totals |
| [raw/2026-09-26-cold-read-progress.csv](raw/2026-09-26-cold-read-progress.csv) | server prompt-processing progress lines (every 30,720 tokens; the end line only for runs from 23:27 UTC on) for all 25 reads in the service journal from 2026-09-25 20:41 to 2026-09-26 02:02 UTC, including partial and KL-probe reads, keyed by `run_start_utc` |
| [raw/2026-09-26-cold-read-windows.csv](raw/2026-09-26-cold-read-windows.csv) | one row per journal read: client-side cold-read totals and per-turn summaries (decode, prompt, acceptance, ms per MTP step) transcribed from each test window's output, the tree under test and what ran in the window |
| [raw/2026-09-26-lcbench-sessions-summary.csv](raw/2026-09-26-lcbench-sessions-summary.csv) | one row per archived lcbench or `accprobe.py` session file (18 files, from the 2026-09-25 agent-session runs to the `-b 3072` session in `lc-test.json`): turns, cold read, acceptance, ms per MTP step, decode overall and median, prompt median, as computed by [`tools/lcsum.py`](../../../tools/lcsum.py) |
| [raw/2026-09-26-lcbench-sessions-per-turn.csv](raw/2026-09-26-lcbench-sessions-per-turn.csv) | the same 18 session files per turn: new and cached prompt tokens, prompt ms, decoded tokens, decode ms and tok/s, draft counts |
| [raw/2026-09-26-klprobe-comparisons.csv](raw/2026-09-26-klprobe-comparisons.csv) | 11 teacher-forced KL comparisons (128 positions, top-20): sparse FA, pooled cache, broadcast score expansion (`kl-ident`), grouped GEMM (`kl-mmid`, `kl-mmid18`) and `eh_proj` (`kl-eh`) against their baselines, with same-build noise floors |
| [raw/2026-09-26-mul-mat-id-kernel-perf.csv](raw/2026-09-26-mul-mat-id-kernel-perf.csv) | `test-backend-ops perf`, one 1536-token `MUL_MAT_ID` per expert type (512 experts, 10 used): per-expert loop against the grouped kernel at patches 0013, 0014 and 0015 |
| [raw/2026-09-26-mul-mat-id-kernel-diag.csv](raw/2026-09-26-mul-mat-id-kernel-diag.csv) | `GGML_SYCL_FG_DIAG` stage-skipping perf passes (results wrong by design) that attribute the grouped kernel's time to dequantization, XMX multiply-add, B loads, pack and host work; two series, builds `517eb833d` and `85b26781b` |
| [raw/2026-09-26-prefill-op-profile.csv](raw/2026-09-26-prefill-op-profile.csv) | `GGML_SYCL_OP_PROFILE` totals, calls, shares and ms per call of the 30 listed ops per build during 135K cold reads of builds `993baf141` and `85b26781b` (3 and 4 print windows summed); the `85b26781b` build label contains a comma and is quoted |
| [raw/2026-09-26-fa-selection-union.csv](raw/2026-09-26-fa-selection-union.csv) | `GGML_SYCL_SPARSE_FA_DEBUG=2` (patch 0017): cells selected together by tiles of 1, 16, 32, 64 and 128 consecutive prompt rows, one QSA layer, per ubatch of a 41,419-token cold read (40 ubatches) and its 11 follow-up turns |
| [raw/2026-09-26-slot-cache-timings.csv](raw/2026-09-26-slot-cache-timings.csv) | slot save/restore with the MTP draft (patch 0010): save sizes and times at 135,760 / 136,528 / 41,364 tokens, the 88 s restart, first-turn times, the re-signed restore, the cold read it replaced, refused cross-build restores |

Generated text is not stored; only timings, counts and summary statistics.
