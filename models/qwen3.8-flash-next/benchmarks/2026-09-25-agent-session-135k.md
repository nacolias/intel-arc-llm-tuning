# Agent-like session at 135K context: lcbench benchmark

| | |
|---|---|
| **Date** | 2026-09-25 |
| **Model / checkpoint** | huihui-ai abliterated `UD-Q4_K_XL` + unsloth MTP draft shared-Q8_0 + mmproj BF16 |
| **Host** | [quad-b70-5800x-pex88096](../../../hardware/hosts/quad-b70-5800x-pex88096.md) |
| **Config** | [llama-server-262k-mtp.sh](../configs/llama-server-262k-mtp.sh) (`-ub 1536 -ts 12,13,13,11`), GPU clock floor 2800 MHz |
| **Experiment** | [sparse FA](../experiments/2026-09-25-sparse-fa-multi-token.md), [pooled QSA cache](../experiments/2026-09-25-pooled-qsa-key-cache.md), [MUL_MAT_ID up to 512](../experiments/2026-09-25-mmid-multitoken-prompt-turns.md) |
| **Raw data** | [raw/2026-09-25-agent-session-135k.csv](raw/2026-09-25-agent-session-135k.csv) (per-turn timings), [raw/2026-09-25-op-profile-135k.csv](raw/2026-09-25-op-profile-135k.csv), [raw/2026-09-25-klprobe.csv](raw/2026-09-25-klprobe.csv) |

## Result

On the production build, decode at 135K context runs at about 39 tok/s, and at about 49 tok/s when the draft is accepted often (medians 39.1 and 48.5 in two runs). The MTP step takes about 73 ms. That is up from 22-23 tok/s and about 135 ms per step on the dense build the same morning. Short prompt turns (about 60 new tokens) take about 1.1 s on every build (as of 2026-09-25; 0.92 s median with the draft QSA the same evening, [raw](raw/2026-09-26-lcbench-sessions-summary.csv) run `lc-draftqsa`, and about 0.76 s on the 2026-09-26 builds with the grouped MoE GEMM and the 2D `eh_proj` product, [raw](raw/2026-09-26-cold-read-windows.csv) row `bcp5hgsyz`; see [grouped GEMM](../experiments/2026-09-26-grouped-moe-xmx-gemm.md) and [eh_proj](../experiments/2026-09-26-mtp-eh-proj-2d-product.md)).

## Environment

| Component | Version / value |
|---|---|
| Engine and image (tag and digest) | llama.cpp, no container. PR #28243 at `6fcaa16` plus patches; build per run in the table below. See [configs/patches/](../configs/patches/README.md) |
| compute-runtime / IGC | 26.35.39758.11 / IGC 2.41.9; Level Zero loader 1.32.0; oneAPI 2026.1 |
| Kernel / `xe` driver / GuC firmware | 7.0.0-34-generic / in-tree `xe` / GuC 70.58.0 |
| oneCCL | not used |
| PyTorch XPU | not used |
| Power cap per card | 230 W |
| KV cache dtype / block size | f16 K and V, 262,144 cells allocated at load |
| Speculative decoding | MTP draft, `--spec-draft-n-max 3`, draft on `SYCL3` |
| Graph / compile mode | eager |

## Workload

| | |
|---|---|
| Workload set | [`tools/lcbench.py`](../../../tools/lcbench.py): a fixed context of C++ source (the llama.cpp tree at `6fcaa16`), read once with `cache_prompt`, then 6 follow-up turns; not yet one of the shared [workloads](../../../benchmarks/workloads/) |
| Input length | cold read 134,862 tokens; turns append the previous answer plus a 120- or 1,500-character "tool output" snippet: 57-62 or 375-416 new tokens |
| Output length | 16 tokens on the cold read, 256 per turn, `ignore_eos` |
| Sampling | greedy (temperature 0) |
| Warm-up | the cold read (turn 0) is excluded from the decode median |
| Repetitions | 1 session per arm; A/B/A arms in one server session for the sparse and pooled tests |

Command:

```bash
BASE=http://127.0.0.1:8080 VLLM_API_KEY="$(cat "$API_KEY_FILE")" \
  LC_CORPUS=lc-corpus.txt LC_TOKENS=135000 LC_TURNS=6 LC_PREDICT=256 LC_OUT=run.json python3 tools/lcbench.py
```

This is a code workload with greedy decoding, so acceptance is higher than for sampled prose. Real client traffic uses temperature 1.0.

## Results

Decode is the median over turns 1-6. Step time is 256 tokens / decode tok/s / (draft_n / 3), the median over turns. Acceptance is accepted / drafted over turns 1-6.

| Run | Build | Attention | Pooled cache | Decode median (min-max) | Step | Acceptance | Prompt, ~60 tok | Prompt, ~400 tok |
|---|---|---|---|---|---|---|---|---|
| `lc-base` | `20260924-7e5cb8f13` | dense | - | 23.3 (17.9-27.0) | 135.4 ms | 66.9% | 1.12-1.17 s | 2.89-2.94 s |
| `lc-lc1` | sandbox (dense, pre-commit 0004) | dense | - | 22.3 (19.4-23.6) | 135.9 ms | 63.4% | 1.11-1.35 s | 2.73-2.79 s |
| `lc-mmid512` | same, `GGML_SYCL_MMID_MULTITOKEN_MAX=512` | dense | - | 21.1 (19.7-21.7) | 134.5 ms | 59.6% | 1.08-1.31 s | 3.40-3.64 s |
| `lc-spA` | sandbox through 0006 | sparse | - | 34.5 (28.2-36.5) | 94.9 ms | 71.8% | 1.12-1.36 s | 2.95-3.02 s |
| `lc-dense` | same session, sparse switched off | dense | - | 22.2 (19.3-24.3) | 135.4 ms | 63.1% | 1.11-1.14 s | 2.66-2.79 s |
| `lc-spC` | same session, sparse on again | sparse | - | 31.3 (29.5-34.1) | 94.7 ms | 64.6% | 1.09-1.11 s | 2.64-2.78 s |
| `lc-poolA` | sandbox through 0007 | sparse | on | 48.5 (45.2-51.0) | 72.6 ms | 82.7% | 1.06-1.32 s | 2.85-2.88 s |
| `lc-pooloff` | same session, cache switched off | sparse | off | 29.9 (28.0-31.7) | 95.8 ms | 60.6% | 1.09-1.12 s | 2.65-2.80 s |
| `lc-poolC` | same session, cache on again | sparse | on | 39.1 (36.8-43.8) | 73.4 ms | 62.5% | 1.04-1.08 s | 2.59-2.74 s |

| Concurrency | Aggregate tok/s | Per-stream tok/s | TTFT p50 | TPOT p50 | Acceptance |
|---|---|---|---|---|---|
| 1 (production code path, `lc-poolA` / `lc-poolC`) | 48.5 / 39.1 | 48.5 / 39.1 | 1.1 s for ~60 new tokens on a cached prefix | 20.6 / 25.6 ms | 82.7% / 62.5% |
| 2 and up | not applicable: one slot | | | | |

| Context length | TTFT (cold) | Single-stream decode tok/s |
|---|---|---|
| 134,862 (cold read) | 405.9-472.2 s (dense builds), 442.6 s (sandbox through 0007); all on the 2026-09-25 builds. 283-286 s on the 2026-09-26 code (N=2: 283.10 s on tree `a890bf8b0`, which is build `2b84213a4` without its diagnostic patch 0017, and 285.88 s on `2b84213a4` with `-b 3072`, [raw](raw/2026-09-26-cold-read-progress.csv); see [cold read by build](2026-09-26-cold-read-135k-by-build.md)) | see the table above |

## Observations

- Real traffic matches the dense-build numbers. On 2026-09-25, one coding-agent session at 126K-140K tokens on build `20260924-7e5cb8f13` reused the cached prefix on every turn: 23-555 new tokens per turn in 1-3.5 s, 19-33 tok/s (two readings of the 2026-09-25 journal; for example 30 tokens in 1.14 s, 280 tokens in 2.47 s, and 2,080 tokens decoded at 25.5 tok/s at 137,411 tokens of context).
- Step time is the stable measure; tok/s also moves with acceptance, which changes with the generated text. `lc-poolA` and `lc-poolC` have the same step time and 82.7% against 62.5% acceptance.
- Op profile of decode turns on the dense build: the 12 QSA attention calls took about 25% of op time (about 2.0 ms each), the pooling slice copies 8.2% and the MTP draft's attention 7.3%. With sparse FA, the QSA calls left the top rows; MoE down/gate/up took 11.7/8.0/7.7%, the pooling copies 10.5% and the draft attention 9.4% ([raw/2026-09-25-op-profile-135k.csv](raw/2026-09-25-op-profile-135k.csv); profile window not archived; shares as recorded on 2026-09-25). The profiler slowed decode only from 22.3 to 20.1 tok/s, so decode at 135K is GPU-bound.
- Prompt-only turns (~60 tokens) are MoE-bound: `MUL_MAT_ID` down/gate/up take 46% of op time; dense QSA attention takes about 4 ms per call because 60 rows exceed the sparse path's limit.
- Next-token distributions (teacher-forced, 128 positions) matched between arms within run-to-run noise; see [raw/2026-09-25-klprobe.csv](raw/2026-09-25-klprobe.csv).
- Runs `lc-dense`, `lc-spC`, `lc-pooloff` and `lc-poolC` started on a cached prefix, so their turn 0 re-read only 1,540 tokens.
