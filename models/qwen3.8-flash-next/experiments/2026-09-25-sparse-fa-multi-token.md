# Sparse flash attention for multi-token batches (MTP verify)

| | |
|---|---|
| **Date** | 2026-09-25 |
| **Model / checkpoint** | huihui-ai abliterated `UD-Q4_K_XL` + unsloth MTP draft shared-Q8_0 + mmproj BF16 |
| **Host** | [quad-b70-5800x-pex88096](../../../hardware/hosts/quad-b70-5800x-pex88096.md) |
| **Status** | kept |
| **Baseline** | dense attention in the same server session (runtime off-switch), and build `20260924-7e5cb8f13` |
| **Result** | [benchmarks/2026-09-25-agent-session-135k.md](../benchmarks/2026-09-25-agent-session-135k.md), [raw/2026-09-25-sparse-fa-kernel.csv](../benchmarks/raw/2026-09-25-sparse-fa-kernel.csv), [raw/2026-09-25-klprobe.csv](../benchmarks/raw/2026-09-25-klprobe.csv) |
| **Related** | [research: long-context QSA design](../research/2026-09-25-long-context-qsa-design.md), [patch 0006](../configs/patches/0006-sycl-sparse-fa-multi-token.diff), upstream [#28796](https://github.com/ggml-org/llama.cpp/pull/28796) |

## Hypothesis

Each of the 12 QSA (Qwen sparse attention) layers attends only the cells its indexer selects: top-k 2048 tokens, which llama.cpp expands to a width of 2051 cells. The dense flash-attention kernel still scans the whole cache with a mask. At 135K, those 12 attention calls took about 25% of decode op time, about 2.0 ms per call at 4 query rows ([raw/2026-09-25-op-profile-135k.csv](../benchmarks/raw/2026-09-25-op-profile-135k.csv)). Upstream #28796 (merged, `cd74ef627`) adds a sparse path for SYCL, but only for single-token queries (`Q->ne[1] == 1`). With MTP (multi-token prediction) on, the target model always verifies 4 tokens, so the upstream path never engages. Gathering each query row's selected cells should cut the attention share to a few ms per step: an estimated 18-20 ms per step at 135K.

## Change

```diff
+ git cherry-pick cd74ef627      # upstream #28796, single-token sparse FA
+ configs/patches/0006-sycl-sparse-fa-multi-token.diff
+ Environment=GGML_SYCL_SPARSE_FA=1
```

Patch 0006 extends the sparse path to up to `GGML_SYCL_SPARSE_FA_MAX_Q=32` query rows:

- per-row atomic compaction of the finite mask cells;
- per-row gather of K, V and the mask into `n_kv_g = pad(n_kv_max + 256, 256)` slots (2,560 for this model);
- each row re-dispatched as its own sequence (`ne[3] = T`) to the unchanged TILE kernel, so Q and dst need no copy.

It engages only while `T * n_kv_g * 2 <= n_kv`. With 4 query rows that is from about 20.5K cells of context on; prompt turns of about 60 tokens stay dense.

## Environment

| Component | Version / value |
|---|---|
| Engine and image (tag and digest) | llama.cpp, no container. Build `20260925-60a598ed8`: PR #28243 at `6fcaa16` + #28931 + 0002 + 0003 + 0004 + #28796 + 0006 |
| Build flags | `GGML_SYCL=ON GGML_SYCL_TARGET=INTEL GGML_SYCL_F16=ON GGML_SYCL_DNN=ON GGML_SYCL_GRAPH=ON GGML_SYCL_SUPPORT_LEVEL_ZERO_API=ON GGML_NATIVE=ON`, Release, icx/icpx from oneAPI 2026.1 |
| compute-runtime / IGC | 26.35.39758.11 / IGC 2.41.9; Level Zero loader 1.32.0 |
| Kernel / `xe` driver / GuC firmware | 7.0.0-34-generic / in-tree `xe` / GuC 70.58.0 |
| oneCCL | not used |
| PyTorch XPU | not used |
| Power cap per card | 230 W |
| Tensor / pipeline parallel | layer split across 4 cards, `-ub 1536 -ts 12,13,13,11 -fit off`, MTP n-max 3 on `SYCL3`, f16 KV |
| GPU clock floor | 2800 MHz |

## Procedure

1. `test-backend-ops -o FLASH_ATTN_EXT -b SYCL0` against CPU, including the new qwen4-shaped multi-row cases. Confirm with `GGML_SYCL_SPARSE_FA_DEBUG=1` that the sparse path engaged.
2. `test-backend-ops perf` for decode (1 row) and MTP verify (4 rows) shapes at 32K, 131K and 262K cells.
3. Model A/B/A in one server session at 135K with [`tools/lcbench.py`](../../../tools/lcbench.py) (fixed C++ context, 6 turns of 256 greedy tokens). Arm A sparse, arm B dense via the file `/tmp/ggml-sycl-sparse-fa-off`, arm C sparse again.
4. Teacher-forced next-token check with [`tools/klprobe.py`](../../../tools/klprobe.py): 128 positions after the same 134,840-token prefix, top-20 log-probabilities, compared by top-1 agreement and KL divergence.
5. Deploy as a frozen build and run [`tools/fnbench.py`](../../../tools/fnbench.py).

## Results

Kernel (hsk = hsv = 256, 2 KV heads, GQA 12, f16 KV, `n_kv_max` 2051):

| KV cells | Query rows | Dense us/run | Sparse us/run | Speed-up |
|---|---|---|---|---|
| 32,768 | 1 | 451.97 | 53.67 | 8.4x |
| 32,768 | 4 | 510.49 | 200.88 | 2.5x |
| 131,072 | 1 | 1,728.39 | 54.47 | 31.7x |
| 131,072 | 4 | 2,004.42 | 206.05 | 9.7x |
| 262,144 | 1 | 3,421.66 | 54.48 | 62.8x |
| 262,144 | 4 | 3,969.57 | 209.18 | 19.0x |

Model, agent session at 135K ([raw/2026-09-25-agent-session-135k.csv](../benchmarks/raw/2026-09-25-agent-session-135k.csv), runs `lc-spA`, `lc-dense`, `lc-spC`):

| Metric | Dense (B) | Sparse (A / C) | Delta |
|---|---|---|---|
| Single-stream decode, median of 6 turns | 22.2 tok/s | 34.5 / 31.3 tok/s | +41% to +55% |
| MTP step time, median (256 tokens / tok/s / (draft_n/3)) | 135.4 ms | 94.9 / 94.7 ms | -30% |
| Acceptance, turns 1-6 | 63.1% | 71.8% / 64.6% | |
| Per-turn prompt time, median | 1.14 s | 1.36 / 1.11 s | none (prompt turns stay dense) |
| Aggregate at c8 / c16 | not applicable (one slot) | | |

Deployed build `20260925-60a598ed8`, `tools/fnbench.py`: short 49.3 tok/s (45.7-51.1), 49.7 at ~39k, 63.1 at ~10k; prompt 590.8 / 582.4 tok/s at 10k / 39k. That is no change at short and ~10k context, where the sparse path does not engage, and no gain visible at ~39k in this single run (52.6 on the previous build). The gain shows at agent-session depth.

## Correctness

- `test-backend-ops` (SYCL0 against CPU): 8/8 qwen4-shaped sparse cases pass, including 5 new multi-row cases (32,768 cells with 2, 4 and 6 rows at `n_kv_max` 2051; 32,768 cells with 16 rows at 512; 16,384 cells with 3 rows, permuted).
- The debug line confirmed engagement: 4 query rows, 32,768 cells, `n_kv_max` 2051, `n_kv_g` 2560, at most 2,051 finite cells per row.
- Teacher-forced KL, 128 positions ([raw/2026-09-25-klprobe.csv](../benchmarks/raw/2026-09-25-klprobe.csv)):

| Comparison | Top-1 agreement | KL mean | KL median | KL p95 | KL max |
|---|---|---|---|---|---|
| sparse A vs sparse C (noise floor) | 115/128 | 0.213 | 0.00196 | 0.311 | 11.67 |
| sparse A vs dense | 112/128 | 0.108 | 0.00178 | 0.747 | 2.81 |
| sparse C vs dense | 113/128 | 0.145 | 0.00242 | | |

Sparse against dense differs no more than two sparse runs differ from each other. Greedy output is not token-identical run to run on this model even without the change: MTP batching changes summation order and the QSA top-k breaks ties in an unspecified order. The sparse path changes only the fp16 accumulation order and tile partition, not the set of attended cells.

## Decision

Kept. +41% to +55% decode at 135K, with next-token distributions inside the run-to-run noise. Deployed as build `20260925-60a598ed8`; the launch config sets `GGML_SYCL_SPARSE_FA=1`.

## Follow-ups

- [x] The QSA indexer's pooling chain became the next-largest attention-side cost (CONT pooling 10.5% of decode op time): [pooled QSA key cache](2026-09-25-pooled-qsa-key-cache.md).
- [ ] The MTP draft layer still attends densely over the whole context (9.4% of decode op time at 135K, 1.87 ms per call).
- [ ] Prompt turns of about 60 tokens are above the 32-row limit and still use dense attention (about 4 ms per call). A union-of-rows gather might cover them.
