# Benchmarks

One file per measured run, named `YYYY-MM-DD-<slug>.md`. Start from [`templates/benchmark.md`](../../../templates/benchmark.md) and follow the shared [methodology](../../../benchmarks/README.md). Raw output goes in [`raw/`](raw/).

All runs are single-slot (`-np 1`), so there is no aggregate throughput; the c16 column of other models does not apply. Two workloads are used:

- [`tools/fnbench.py`](../../../tools/fnbench.py): a short technical-prose prompt (512 tokens decoded), a repeated paragraph at 9,749 and 39,119 tokens, and optionally 219,217 tokens of structured records. Greedy. The repeated and structured prompts flatter MTP (multi-token prediction) acceptance.
- [`tools/lcbench.py`](../../../tools/lcbench.py): an agent-like session on a fixed 134,862-token C++ source context, 6 follow-up turns of 256 greedy tokens. Code workload.

| Date | Benchmark | Config | Single-stream | Aggregate (c16) | Notes |
|---|---|---|---|---|---|
| 2026-09-24 | [Baseline 262K + MTP, unpatched](2026-09-24-baseline-262k-mtp.md) | `-ub 1024 -ts 13,13,13,10`, MTP n-max 3, stock unsloth checkpoint | 33.8-36.9 short; 43.3 at ~10k; 33.4 at ~39k; 16.1 at 219K | n/a | busiest card 27,420 MiB; MTP off: 33.1-33.5 / 31.4 / 23.4 / 7.2 |
| 2026-09-25 | [Agent session at 135K](2026-09-25-agent-session-135k.md) | production flags; dense, sparse FA and pooled-cache builds | 22-23 dense; 31-35 sparse FA; 39.1-48.5 with the pooled cache | n/a | MTP step 135 to 73 ms; prompt turns of ~60 tokens about 1.1 s on every build |
| 2026-09-25 | [Production build `20260925-f47a6a5f3`](2026-09-25-production-build.md) | [llama-server-262k-mtp.sh](../configs/llama-server-262k-mtp.sh) defaults, clock floor | 48.6-50.5 short; 60.6-67.2 at ~10k; 53.6-55.2 at ~39k; 40.6 at 219K | n/a | prompt 577-645 tok/s at 10k-39k, 268.3 at 219K; busiest card 28,957 MiB |

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

Generated text is not stored; only timings, counts and summary statistics.
