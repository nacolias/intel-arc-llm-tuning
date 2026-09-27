# Sparse attention for prompt batches: what a tile-union gather would save

| | |
|---|---|
| **Date read** | 2026-09-26 |
| **Source** | our own measurement on [quad-b70-5800x-pex88096](../../../hardware/hosts/quad-b70-5800x-pex88096.md) with the diagnostic in [patch 0017](../configs/patches/0017-sycl-sparse-fa-union-stats-debug.diff) (`GGML_SYCL_SPARSE_FA_DEBUG=2`), the [prefill op profile](../benchmarks/2026-09-26-prefill-op-profile.md) and the cold-read progress lines; raw data in [raw/2026-09-26-fa-selection-union.csv](../benchmarks/raw/2026-09-26-fa-selection-union.csv) |
| **Author / org** | this repository (AI-assisted analysis) |
| **Type** | diagnostic measurement and cost arithmetic |
| **Applies to** | Qwen3.8-Flash-Next (llama.cpp arch `qwen4exp`) on llama.cpp SYCL: the 12 QSA (Qwen sparse attention) layers during prompt processing at more than 2,051 cells of context |

## Summary

Consecutive prompt rows do not share most of their selected cells, so a tile-union sparse kernel would save less than hoped. Each QSA query row attends 2,051 cells of the context. At 34K cells of context, a tile of 16 consecutive rows keeps 7,649 cells together (3.7x one row), 64 rows keep 15,834 (7.7x) and 128 rows 21,499 (10.5x). Attending the union instead of the whole context would do 22% (16-row tiles) to 46% (64-row tiles) of dense attention's work at that context. Dense prompt attention was 32.8% of listed op time at the profiled context, but a smaller share of a whole cold read, because the context grows from zero during the read; the session estimated about 12% (source: session log, not archived). The session judged the change not worth building now: about 3-5% off a cold read by reusing the existing gather and kernel per tile, about 10% (25-30 s at 135K) with a custom kernel that takes several days (estimates; source: session log, not archived). **Not built.** How large attention's share of a whole read is remains unresolved and decides whether that ceiling is right.

## How the measurement works

[Patch 0006](../configs/patches/0006-sycl-sparse-fa-multi-token.diff) gathers each query row's selected cells and attends them, but only for batches of up to 32 rows ([sparse flash-attention (FA) experiment](../experiments/2026-09-25-sparse-fa-multi-token.md)). Prompt micro-batches (ubatches) of 512 or 1,536 rows take the dense TILE kernel with a mask over the whole context. A tile-union design would gather, once per tile of R consecutive rows, the cells that any row of the tile keeps, and run attention over that union.

Patch 0017 adds a diagnostic: with `GGML_SYCL_SPARSE_FA_DEBUG=2`, for a masked attention call too large for the per-row path, it counts on the GPU, for R in 1, 16, 32, 64 and 128, how many cells have a finite mask value in any row of each R-row tile, and logs the mean and maximum over the tiles. One call in 12 is logged. With 12 QSA layers per ubatch that is one line per ubatch, from one layer only (which layer is not recorded; the union sizes of the other 11 layers are assumed similar: UNVERIFIED). The data come from one `accprobe.py` session on a 41,419-token context ([eh_proj experiment](../experiments/2026-09-26-mtp-eh-proj-2d-product.md), session B): 40 lines from the cold read (`n_q` alternating 1,536 and 512, the two ubatch sizes of `-b 2048 -ub 1536`) and 11 from the follow-up turns (`n_q` 54-465).

## Measurements

Selected rows of [raw/2026-09-26-fa-selection-union.csv](../benchmarks/raw/2026-09-26-fa-selection-union.csv), 1,536-row ubatches. `n_kv_max` is 2,051 in every row from 3,584 cells on: one row's selection (top-k 2048 tokens, expanded to a width of 2,051 cells). Mean over the tiles of the ubatch; the maximum in brackets for R = 16 and 64.

| Context (cells) | R = 1 | R = 16 | R = 32 | R = 64 | R = 128 | 16 rows: x one row / % of dense | 64 rows: x one row / % of dense |
|---|---|---|---|---|---|---|---|
| 3,584 | 2,051 | 2,643 (3,256) | 2,728 | 2,800 (3,520) | 2,862 | 1.3x / 74% | 1.4x / 78% |
| 9,728 | 2,051 | 5,114 (6,028) | 6,062 | 6,995 (7,782) | 7,911 | 2.5x / 53% | 3.4x / 72% |
| 17,920 | 2,051 | 5,892 (7,492) | 7,536 | 9,571 (11,878) | 11,334 | 2.9x / 33% | 4.7x / 53% |
| 26,112 | 2,051 | 7,355 (9,142) | 10,047 | 13,757 (15,629) | 17,096 | 3.6x / 28% | 6.7x / 53% |
| 34,304 | 2,051 | 7,649 (9,251) | 10,863 | 15,834 (17,972) | 21,499 | 3.7x / 22% | 7.7x / 46% |
| 38,400 | 2,051 | 8,014 (10,344) | 11,754 | 17,445 (19,631) | 22,811 | 3.9x / 21% | 8.5x / 45% |

Below 2,051 cells every visible cell is selected (at 1,536 cells a row keeps 768 on average: the causal half).

The union grows slowly with context once the context is large: from 9,728 to 38,400 cells (4x), the 16-row union grows from 5,114 to 8,014 (1.6x). Relative to dense attention, the union therefore keeps shrinking: 53% at 9.7K, 22% at 34K, 21% at 38K. Above 45K cells nothing was measured; the trend suggests a 16-row union in the low tens of thousands of cells at 135K (extrapolation, UNVERIFIED).

**Agent turns.** The 11 follow-up turns of the session (54-465 new tokens at 41.7-45.8K cells) are the case of a real coding-agent client's prompt turns. For the ~60-row turns, one 64-row tile covers the whole turn:

| Turn rows (`n_q`) | Context (cells) | One row | Whole turn as one tile (R = 64) | % of dense |
|---|---|---|---|---|
| 59 | 41,728 | 2,051 | 14,734 | 35% |
| 75 | 42,752 | 2,051 | 9,154 | 21% |
| 56 | 43,008 | 2,051 | 12,018 | 28% |
| 54 | 43,776 | 2,051 | 8,778 | 20% |
| 62 | 44,032 | 2,051 | 11,461 | 26% |
| 56 | 45,056 | 2,051 | 11,291 | 25% |
| 64 | 45,312 | 2,051 | 10,651 | 24% |

So a single-tile union would cut a ~60-token turn's dense attention work by 65-80% at 42-45K. On earlier builds the 12 dense calls of such a turn took about 4 ms each at 135K ([prefill bottleneck](2026-09-25-prefill-bottleneck.md)), about 48 ms of a turn that now takes about 0.76 s: a saving of about 5% of the turn (estimate).

## Arithmetic

All figures in this section are estimates derived from the archived data as cited.

**Work.** Dense attention on a 1,536-row ubatch at `n_kv` cells does work proportional to `1536 x n_kv`. A tile-union kernel does `sum over tiles of R x U(R, n_kv)`, so the ratio is `U(R, n_kv) / n_kv`: 22% for R = 16 and 46% for R = 64 at 34,304 cells (table above). The per-row ideal, `2051 / n_kv`, is 6% there. A 16-row tile does 3.7x the ideal work, a 64-row tile 7.7x.

**Gather traffic of the cheap version.** Reusing patch 0006's gather per tile copies K and V of every union cell: 2 KV (key-value) heads x 256 dims x 2 bytes x (K and V) = 2,048 bytes per cell ([model card](../README.md): head dim 256, 2 KV heads, f16 KV). Per QSA layer and 1,536-row ubatch at 34,304 cells:

| R | Tiles per ubatch | Union cells per tile | Gather per layer | Gather per ubatch (12 layers) | Attention work vs dense |
|---|---|---|---|---|---|
| 16 | 96 | 7,649 | 1.5 GB | 18 GB | 22% |
| 64 | 24 | 15,834 | 0.78 GB | 9.3 GB | 46% |
| 128 | 12 | 21,499 | 0.53 GB | 6.3 GB | 63% |

Dense attention reads the whole K and V once per layer (34,304 cells x 2 KB = 70 MB per layer, 0.84 GB per ubatch), so the cheap version moves 7-20x more bytes than dense attention reads, and launches 12-96 kernels per layer instead of one. That is the "per-call overhead" that the session expected to eat most of the gain (3-5% of a cold read; source: session log, not archived). A custom kernel that reads K and V through the index list, as the reference implementations do for decode, avoids the copy.

**Share of a whole cold read.** At the profiled context (about 100-120K tokens), dense attention was 32.8% of listed op time on `85b26781b`, or 35.8% with the since-removed `eh_proj` taken out ([op profile](../benchmarks/2026-09-26-prefill-op-profile.md); serialized op time, listed ops only). Attention cost grows with context and the MoE (mixture-of-experts) cost does not, so attention's share of a read from zero is lower than at any late point. The session put attention at about 20% of a ubatch's GPU time at 100K and about 12% of a whole 135K cold read (source: session log, not archived).

A check against the archived progress lines of the 283 s read on the `eh_proj` build ([raw/2026-09-26-cold-read-progress.csv](../benchmarks/raw/2026-09-26-cold-read-progress.csv), run 01:36:43 UTC): the four 30,720-token segments took 46.5, 56.4, 65.8 and 78.9 s. A straight line through segment time against mid-segment context gives about 40.6 s of context-independent work per segment plus 0.35 s per 1K cells of context, which puts all context-dependent work at about 103 s, 36% of the read (one run, estimate). Attention is part of that; the QSA indexer's per-cell ops (the mask copy at 3.7%, `TOP_K` at 1.5%, `indexer_score_tokens`) are the rest. In the profile, attention is about 85% of those context-dependent ops. If that ratio holds across the read, attention is nearer 25-30% of a cold read than 12%, and the ceiling below roughly doubles. **This is unresolved**: the profile is at one unknown context, and the segment fit is one run.

**Ceiling.** The saving is `S x (1 - W)`, with S attention's share of the read and W the union's work relative to dense averaged over the read. With S = 12% and W falling from about 50% early in the read to about 20% late, the ceiling is 6-10%, which matches the session's "about 10% with a custom kernel, 25-30 s at 135K". With S = 25-30%, it is 15-24% (estimates).

## Claims to verify

| Claim | Reported number | Their setup | Status |
|---|---|---|---|
| 16 consecutive prompt rows select about 3.7x one row's cells at 34K; 64 rows about 7.7x | 7,649 and 15,834 vs 2,051 | our diagnostic, one layer, one session | verified-here |
| A 16-row union keeps shrinking relative to dense as context grows | 53% at 9.7K, 22% at 34K, 21% at 38K | same | verified-here up to 45K; above that UNVERIFIED |
| Union sizes are similar across the 12 QSA layers | – | assumption; one layer logged | UNVERIFIED |
| Attention is about 12% of a whole 135K cold read | ~12% | session estimate | UNVERIFIED; the segment fit puts all context-dependent work at ~36% |
| Cheap tile-union (reuse gather and kernel per tile) saves 3-5% of a cold read | 3-5% | session estimate | UNVERIFIED |
| A custom index-reading kernel saves about 10% (25-30 s at 135K) | ~10% | session estimate | UNVERIFIED; 15-24% if attention's share is 25-30% |
| Earlier plan estimate: 135K prefill from ~300 to ~450-500 tok/s with tile-union attention | +50-65% | plan of 2026-09-25, before the union was measured | superseded by the measurements above |

## Relevance

Cold reads of a 135K conversation take about 284 s on the production build ([eh_proj experiment](../experiments/2026-09-26-mtp-eh-proj-2d-product.md)), and the client re-reads cold only after a restart or a cache eviction. Normal agent turns are mostly decode (about 7 s of decode against about 0.8 s of prompt at 135K; source: session log, not archived), so decode work matters more day to day. That, the small measured overlap between rows, and the "several days" estimate for a custom kernel are why the change was not built on 2026-09-26. The single-tile case for ~60-token turns is cheaper than the prompt case and is a candidate if turn latency becomes the target.

## Actions

- [x] Measure selection-union sizes for tiles of consecutive prompt rows (patch 0017; this note)
- [ ] Pin attention's share of a whole cold read: profile at three contexts (about 30K, 90K and 130K) on the production build, or fit the progress lines of several reads. This decides whether the ceiling is ~10% or ~20%.
- [ ] Log the union statistics for all 12 QSA layers, and extend them to a 135K read (one `GGML_SYCL_SPARSE_FA_DEBUG=2` cold read).
- [ ] If the share is 25% or more: design the index-reading kernel (a per-tile variant of the reference decode kernel) and measure a 1,536-row call against the dense TILE kernel with `test-backend-ops perf` before touching the model graph.
- [ ] Single-tile union for prompt turns up to 64 rows, as a lower-effort first step; measure the ~60-token turn at 135K.
