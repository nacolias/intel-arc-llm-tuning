# QSA score expansion by broadcast

| | |
|---|---|
| **Date** | 2026-09-26 |
| **Model / checkpoint** | huihui-ai abliterated `UD-Q4_K_XL` + unsloth MTP draft shared-Q8_0 + mmproj BF16 |
| **Host** | [quad-b70-5800x-pex88096](../../../hardware/hosts/quad-b70-5800x-pex88096.md) |
| **Status** | kept; production since 2026-09-26 00:04 UTC as build `20260926-2c88bdf19`, carried into `20260926-2b84213a4` |
| **Baseline** | build `20260925-993baf141` (expansion by gather): [benchmarks/2026-09-26-cold-read-135k-by-build.md](../benchmarks/2026-09-26-cold-read-135k-by-build.md) |
| **Result** | same benchmark; [raw/2026-09-26-cold-read-progress.csv](../benchmarks/raw/2026-09-26-cold-read-progress.csv), [raw/2026-09-26-cold-read-windows.csv](../benchmarks/raw/2026-09-26-cold-read-windows.csv), [raw/2026-09-26-klprobe-comparisons.csv](../benchmarks/raw/2026-09-26-klprobe-comparisons.csv) |
| **Related** | [patch 0011](../configs/patches/0011-qwen4exp-qsa-score-expansion-by-broadcast.diff), [pooled QSA key cache](2026-09-25-pooled-qsa-key-cache.md), [research: long-context QSA design](../research/2026-09-25-long-context-qsa-design.md), [research: prefill bottleneck](../research/2026-09-25-prefill-bottleneck.md), [raw/2026-09-26-prefill-op-profile.csv](../benchmarks/raw/2026-09-26-prefill-op-profile.csv), [prefill op profile](../benchmarks/2026-09-26-prefill-op-profile.md) |

A cold read of the 134,862-token [`tools/lcbench.py`](../../../tools/lcbench.py) context took 382.49 s on this build (client-side; 381.74 s at the server's end line), against 455.35 s on the production build before it and 436.8-462.0 s over six runs of the same expansion code on three trees (-16% against the pooled median of 454.9 s). That is N=1 after the change. The archived reads before it span 403.26-472.24 s over 14 sessions with differing builds and settings ([raw/2026-09-26-lcbench-sessions-summary.csv](../benchmarks/raw/2026-09-26-lcbench-sessions-summary.csv), [raw/2026-09-26-cold-read-windows.csv](../benchmarks/raw/2026-09-26-cold-read-windows.csv)). So the step is somewhere between about 5% and 17% (estimate: against the same night's nine reads, 403.26-461.98 s; 19% against the slowest, 472.24 s on an earlier 2026-09-25 build). Decode and short prompt turns did not change. Next-token probabilities at 128 teacher-forced positions matched the pooled-cache build within run-to-run noise. The first build of the change crashed on an unused graph input; the fix is part of the patch.

## Hypothesis

QSA (Qwen sparse attention) scores 4-token blocks with an indexer, then every cell must carry its block's score so that the top-k over cells cuts on a block boundary. `build_qsa_top_k` did this with `ggml_get_rows` over a transposed copy of the `[n_blocks, n_tokens]` scores, indexed by a per-cell block table (`cell_blk`), and then transposed the `[n_kv, n_tokens]` result back: two `cont(permute)` copies of a cells-by-tokens F32 matrix per QSA layer and ubatch (micro-batch). Near the end of a 135K read with `-ub 1536` each copy is 134,862 x 1,536 x 4 bytes, about 830 MB (arithmetic, not measured). The op profile of a cold read on `20260925-993baf141` put `CONT (permuted)` at 13.3% of the profiled op time, 246 calls at 13.24 ms ([raw/2026-09-26-prefill-op-profile.csv](../benchmarks/raw/2026-09-26-prefill-op-profile.csv)); the commit message says "~13% of a cold 110K prefill".

The pooled key cache ([patch 0007](../configs/patches/0007-qwen4exp-pooled-qsa-key-cache.diff)) already builds a per-ubatch plan of the cache layout. In the production case, one conversation with no cache edits in the middle, every cached cell `j` holds position `j`, so cell `j` belongs to block `j / ratio`. Then the expansion is a broadcast add of the `[1, n_blocks, n_tokens]` scores onto the mask viewed as `[ratio, n_blocks, n_tokens]`: same values, no gather, no transposes, and no `cell_blk` upload. Expected: up to about 13% off a cold read; nothing on decode, where the MTP (multi-token prediction) verify batch has at most 4 tokens and the copies are small; nothing on ~60-token prompt turns.

## Change

[Patch 0011](../configs/patches/0011-qwen4exp-qsa-score-expansion-by-broadcast.diff), commit `2c88bdf19` (2026-09-26), on top of `993baf141`. No runtime switch.

- **Plan flag.** While building the pooled plan, `qsa_plan::ident` is set when `n_blocks * ratio == n_kv` and every non-empty cell `j` has `pos == j`.
- **Graph.** When `plan.fast && plan.ident && n_stream == 1`, the mask (cast to F32 if it is F16) is reshaped to `[r, n_blocks, n_tps]` and the scores to `[1, n_blocks, n_tps]`; `ggml_add` broadcasts the block score over the `r` cells of each block; the result is reshaped back to `[n_kv, n_tps, n_stream]`. Every other case (several streams, cells out of position order after cache edits, image positions) keeps the old gather path unchanged.
- **Graph reuse.** The reuse check compares `plan.ident` too, so a layout change rebuilds the graph.
- **Unused input.** `cell_blk` is still created as an input tensor, but the broadcast graph never reads it, so the graph allocator gives it no buffer. `set_input_qsa_fast` now writes it only `if (cell_blk->buffer)`.

```diff
-    GGML_ASSERT(ggml_backend_buffer_is_host(cell_blk->buffer));
-
-    memcpy(cell_blk->data, p.cell_blk.data(), n_kv*sizeof(int32_t));
+    // the ident broadcast (cells in position order) doesn't read cell_blk, so it has no buffer
+    if (cell_blk->buffer) {
+        GGML_ASSERT(ggml_backend_buffer_is_host(cell_blk->buffer));
+        memcpy(cell_blk->data, p.cell_blk.data(), n_kv*sizeof(int32_t));
+    }
```

**The crash of the first build.** The first build kept the unconditional assert above. With no buffer behind `cell_blk`, the server aborted with a `GGML_ASSERT(buffer)` failure as soon as a graph took the broadcast path. Production was brought back, the guard was added, and the test window was restarted: first attempt 23:37-23:45 UTC, second attempt from 23:47 UTC (source: session log, not archived). The commit message records the mechanism: "The broadcast leaves the cell_blk input out of the graph, so it gets no buffer; set_input_qsa_fast skips writing it then."

## Environment

| Component | Version / value |
|---|---|
| Engine and image (tag and digest) | llama.cpp, no container. Dev tree = `20260925-993baf141` + patch 0011, frozen after the test as build `20260926-2c88bdf19`: PR #28243 at `6fcaa16` + #28931 + 0002 + 0003 + 0004 + #28796 + 0006 + 0007 + 0008 + 0009 + 0010 + 0011 |
| Build flags | `GGML_SYCL=ON GGML_SYCL_TARGET=INTEL GGML_SYCL_F16=ON GGML_SYCL_DNN=ON GGML_SYCL_GRAPH=ON GGML_SYCL_SUPPORT_LEVEL_ZERO_API=ON GGML_NATIVE=ON`, Release, icx/icpx from oneAPI 2026.1 |
| compute-runtime / IGC | 26.35.39758.11 / IGC 2.41.9; Level Zero loader 1.32.0 |
| Kernel / `xe` driver / GuC firmware | 7.0.0-34-generic / in-tree `xe` / GuC 70.58.0 |
| oneCCL | not used |
| PyTorch XPU | not used |
| Power cap per card | 230 W |
| Tensor / pipeline parallel | layer split across 4 cards, `-ub 1536 -ts 12,13,13,11 -fit off`, MTP n-max 3 on `SYCL3`, f16 KV (key/value) cache, `GGML_SYCL_SPARSE_FA=1`, pooled key cache on, draft QSA on, `-np 1` |
| GPU clock floor | 2800 MHz |

## Procedure

1. Build the dev tree with the patch. Hold the saved production slot (`slot-cache hold`, see [configs/llama-server-slot-cache.sh](../configs/llama-server-slot-cache.sh)) and restart the service on the dev build; the saved slot is refused because the build signature differs.
2. Salted cold read (`LC_SALT` adds 4 tokens: 134,866 new tokens) plus 6 follow-up turns with [`tools/lcbench.py`](../../../tools/lcbench.py); summary with [`tools/lcsum.py`](../../../tools/lcsum.py).
3. Teacher-forced next-token probe with [`tools/klprobe.py`](../../../tools/klprobe.py), 128 positions, top-20, written as `kl-ident.json`; compared against `kl-poolA.json` (the pooled-cache sandbox of 2026-09-25) with the same-build pair `kl-poolC` vs `kl-poolA` as the noise floor.
4. Restore production; deploy the frozen build at 00:04 UTC. The whole window ran 2026-09-25 23:47 to 2026-09-26 00:04 UTC; the read's progress lines are at 23:50:17 to 23:55:31 in [raw/2026-09-26-cold-read-progress.csv](../benchmarks/raw/2026-09-26-cold-read-progress.csv).

## Results

Cold reads are single runs unless noted ([raw/2026-09-26-cold-read-windows.csv](../benchmarks/raw/2026-09-26-cold-read-windows.csv), [raw/2026-09-26-cold-read-progress.csv](../benchmarks/raw/2026-09-26-cold-read-progress.csv), [raw/2026-09-26-lcbench-sessions-summary.csv](../benchmarks/raw/2026-09-26-lcbench-sessions-summary.csv)). The full per-run table, with every run's start time and what else ran in that window, is in [benchmarks/2026-09-26-cold-read-135k-by-build.md](../benchmarks/2026-09-26-cold-read-135k-by-build.md).

| Metric | Gather (`993baf141` and the two dev trees before it) | Broadcast (`2c88bdf19` dev tree, 23:50 UTC) | Delta |
|---|---|---|---|
| Cold 135K read, client-side | 455.35 s (production `993baf141`, N=1); 436.8-462.0 s, median 454.9 s over N=6 runs of the same expansion code | 382.49 s (N=1) | -16% against the median; about -5% to -17% against the same night's reads (estimate, see below) |
| Server progress line at 122,880 tokens | 397.21 s (the `993baf141` run) | 337.12 s | -15% (one run each) |
| Server progress line at the end of the read | 447.01 s (only logged for the profile run) | 381.74 s | |
| Prompt speed over the whole read (134,862 tokens / client-side seconds) | 296 tok/s | 353 tok/s | +19% (one run each) |
| Per-turn prompt, ~60 new tokens, median | 0.96 s (2 turns, `993baf141`); 0.92 s (10 turns, draft-QSA tree) | 0.91 s (6 turns) | none |
| Single-stream decode, median (min-max) | 39.1 (37.7-39.1), 2 turns; 41.1 (39.6-44.7), 10 turns | 43.9 (39.9-54.2), 6 turns | within session-to-session variation |
| Draft acceptance | 63.1% (draft-QSA tree, 10 turns) | 66.0% | |
| MTP step time | 70.5 ms (draft-QSA tree); 70.3 ms (paired A/B, same tree) | 67.1 ms | -3 ms; see below |
| Aggregate at c8 / c16 | not applicable (one slot) | | |

The step time is 3 ms lower than in the two draft-QSA sessions before the change, and the three later builds that carry the change measured 68.0-68.4 ms per step. The expansion also runs in decode (4-token verify batches), so a small decode-side saving is plausible, but it was not tested with an A/B in one session: UNVERIFIED. Decode tok/s moves with acceptance, which differs between sessions because the generated text differs.

The commit message summarises the same window as "135K cold read: 447-461 s -> 382 s". Which six earlier sessions that range covers is not recorded; the clean runs archived here span 436.8-462.0 s and include two faster reads on the draft-QSA tree.

**Size of the step.** The broadcast build was read cold once (N=1). Before it, the archived 135K reads span 403.26-472.24 s over 14 sessions with differing builds and settings: 10 lcbench sessions in [raw/2026-09-26-lcbench-sessions-summary.csv](../benchmarks/raw/2026-09-26-lcbench-sessions-summary.csv), plus the pairprobe, host-timing, slot-test and profile windows in [raw/2026-09-26-cold-read-windows.csv](../benchmarks/raw/2026-09-26-cold-read-windows.csv). Against the same night's nine reads (403.26-461.98 s) 382.49 s is 5-17% faster (estimate). The -16% above is one point in that range. The later steps on top of this change are repeated and tight: server end lines of 311.84, 311.98 and 313.34 s with the grouped MoE (mixture-of-experts) GEMM (general matrix multiply) (two builds and one profiling run; a second profiling run ended at 294.94 s), then 283.10 and 285.88 s with the 2D `eh_proj` (the second with `-b 3072`) ([benchmarks/2026-09-26-cold-read-135k-by-build.md](../benchmarks/2026-09-26-cold-read-135k-by-build.md)).

## Correctness

Teacher-forced next-token distributions, 128 positions, top-20 ([raw/2026-09-26-klprobe-comparisons.csv](../benchmarks/raw/2026-09-26-klprobe-comparisons.csv)):

| Comparison | Top-1 agreement | KL mean | KL median | KL p95 | KL max |
|---|---|---|---|---|---|
| `kl-poolC` vs `kl-poolA` (two runs of one build, noise floor) | 116/128 | 0.09916 | 0.00264 | 0.39624 | 4.86846 |
| `kl-ident` vs `kl-poolA` (this change against the pooled-cache build) | 116/128 | 0.10726 | 0.00208 | 0.37733 | 4.22825 |

- At or inside the noise-floor pair on every column except the mean (0.107 against 0.099); the file's other same-build pair, `kl-spC` against `kl-spA`, has a mean of 0.207, so the mean is inside the same-build range too. KL is the Kullback-Leibler divergence over the top-20 log-probabilities per position.
- By construction the broadcast produces the same values as the gather: cell `j` of block `j / r` receives the block's score, and the mask is added in both paths. No token diff was used; greedy output is not token-identical run to run on this model even without the change (MTP batching, top-k tie order).
- The gather fallback is the previous code, unchanged. No archived run exercised it after the change (a conversation with cache edits in the middle, several streams, or image positions): UNVERIFIED.

## Decision

Kept. The cold read got faster by somewhere between about 5% and 17% (estimate; one read after the change, 382.49 s, against 403.26-461.98 s the same night; -16% or 72 s against the pooled median) for a pure graph change with no new kernels and no VRAM cost, and the probe stayed within noise. Deployed as production build `20260926-2c88bdf19` at 2026-09-26 00:04 UTC; the saved 41K-token production slot was re-signed and restored (see [slot save/restore](2026-09-25-slot-save-restore-and-ram-cache.md)). The build was superseded the same night by `20260926-2b84213a4`, which keeps this change.

## Follow-ups

- [ ] A/B the decode-side effect in one session (67.1 ms against 70.3-70.5 ms per step).
- [ ] Block-level top-k: select over the ~34K blocks instead of ~135K cells, which also removes the 513th-block deviation from the reference (see [research: long-context QSA design](../research/2026-09-25-long-context-qsa-design.md)).
- [ ] The remaining per-ubatch copies around the mask: `CPY attn_inp_kq_mask` was 2.0% of the profiled op time on `993baf141` and 3.7% on `85b26781b` ([raw/2026-09-26-prefill-op-profile.csv](../benchmarks/raw/2026-09-26-prefill-op-profile.csv)); the broadcast path still casts an F16 mask to F32.
- [ ] A test that drives the gather fallback after a `seq_rm` in the middle of the cache and checks `ident` flips off.
