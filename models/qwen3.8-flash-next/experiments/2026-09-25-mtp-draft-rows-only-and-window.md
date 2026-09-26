# MTP draft: output-rows-only attention and a sliding window

| | |
|---|---|
| **Date** | 2026-09-25 |
| **Model / checkpoint** | huihui-ai abliterated `UD-Q4_K_XL` + unsloth MTP draft shared-Q8_0 + mmproj BF16 |
| **Host** | [quad-b70-5800x-pex88096](../../../hardware/hosts/quad-b70-5800x-pex88096.md) |
| **Status** | rows-only attention: kept. `LLAMA_MTP_WINDOW`: reverted (the option stays in the code, off by default). `LLAMA_HOST_PROF`: kept as a diagnostic, off by default |
| **Baseline** | the dense draft attending every row, session `lc-mtpwin-0-allrows` in [raw/2026-09-26-lcbench-sessions-summary.csv](../benchmarks/raw/2026-09-26-lcbench-sessions-summary.csv) |
| **Result** | [raw/2026-09-26-lcbench-sessions-summary.csv](../benchmarks/raw/2026-09-26-lcbench-sessions-summary.csv) rows `lc-mtpwin-*`, per turn in [raw/2026-09-26-lcbench-sessions-per-turn.csv](../benchmarks/raw/2026-09-26-lcbench-sessions-per-turn.csv) |
| **Related** | [patch 0008](../configs/patches/0008-qwen4exp-mtp-draft-output-rows-only-attention.diff), [research: long-context QSA design](../research/2026-09-25-long-context-qsa-design.md) (design 4), [pooled QSA key cache](2026-09-25-pooled-qsa-key-cache.md) (its first follow-up), [QSA in the MTP draft head](2026-09-25-mtp-draft-qsa.md) (the next step) |

## Hypothesis

After sparse flash attention and the pooled key cache, the MTP (multi-token prediction) draft layer was the largest context-dependent cost left in decode. It attended densely over the whole context: 9.4% of decode op time at 135K, 1.87 ms per call on build `20260925-60a598ed8` ([raw/2026-09-25-op-profile-135k.csv](../benchmarks/raw/2026-09-25-op-profile-135k.csv)). Two cheap changes were tried ahead of a full QSA (Qwen sparse attention) draft:

1. **Attend only on rows that produce an output.** One MTP step runs the draft graph four times: three one-token draft decodes, and one replay of the target's verify batch into the draft context (the catch-up), whose rows have no outputs. Prompt chunks are replayed the same way. The attention output of those rows was computed and then discarded by `ggml_get_rows(cur, inp_out_ids)`, so skipping it is exact. Expected saving: about one of four dense attention calls per step, roughly 1.8 ms at 135K (estimate; source: session log, not archived).
2. **A sliding window for the draft** (`LLAMA_MTP_WINDOW=W`). The draft cache masks cells more than W positions back, so each draft attention call scans at most W cells instead of 135K. Expected saving at W = 2048: about 6.5-7 ms per step (estimate; source: session log, not archived). Expected cost: the draft loses far context, so acceptance may fall. With n-max 3 and p-min 0, tokens per step = 1 + 3a for acceptance a; at a = 0.62 and 72 ms per step that is 2.86 / 0.072, about 40 tok/s, so one acceptance point is worth about 1% of decode speed (arithmetic).

Neither change can alter the target's output. The target verifies every draft token; only acceptance and speed can move.

## Change

[Patch 0008](../configs/patches/0008-qwen4exp-mtp-draft-output-rows-only-attention.diff), commit message: "qwen4exp MTP draft: output-rows-only attention, optional LLAMA_MTP_WINDOW; LLAMA_HOST_PROF host timing".

- **Rows-only attention** (`src/models/qwen4exp.cpp`, `graph_mtp`). Every row still writes its K and V, which later draft steps read. When `n_outputs < n_tokens`, the query, the attention gate and the mask rows are gathered with `inp_out_ids` before `build_attn_mha`, so only the output rows attend; a ubatch with no outputs (catch-ups, prompt chunks) skips attention. The later `get_rows` on the attention output is skipped when the rows were already selected. One stream is asserted.
- **`LLAMA_MTP_WINDOW=W`** (`src/llama-model.cpp`, `create_memory`; default 0 = off). The draft's `llama_kv_cache` is created with `n_swa = W` and `swa_type = LLAMA_SWA_TYPE_STANDARD`, and the draft graph passes W to flash attention as the sparse-attention hint.
- **`LLAMA_HOST_PROF=<seconds>`** (`src/llama-context.cpp`, `src/llama-graph.cpp`; default 0 = off). Host time of each `process_ubatch` phase (`apply`, `build/reuse`, `set_inputs`, `compute`), keyed by context (`draft` or `target`) and ubatch size class (`t=1`, `t<=8`, `t>8`), plus `set_input` time per graph-input type. Printed as `[HOST-PROF]` lines every `<seconds>`.

```diff
+ Environment="LLAMA_MTP_WINDOW=2048"     # arm W=2048; 8192 for the other; unset or 0 for dense
+ Environment="LLAMA_HOST_PROF=20"        # diagnostic runs only
```

## Environment

| Component | Version / value |
|---|---|
| Engine and image (tag and digest) | llama.cpp, no container. Sandbox build of the patch series through 0008 (commit `9f41d810e`): PR #28243 at `6fcaa16` + #28931 + 0002 + 0003 + 0004 + #28796 + 0006 + 0007 + 0008. No frozen build directory was made for it; production stayed on `20260925-f47a6a5f3` between arms |
| Build flags | `GGML_SYCL=ON GGML_SYCL_TARGET=INTEL GGML_SYCL_F16=ON GGML_SYCL_DNN=ON GGML_SYCL_GRAPH=ON GGML_SYCL_SUPPORT_LEVEL_ZERO_API=ON GGML_NATIVE=ON`, Release, icx/icpx from oneAPI 2026.1 |
| compute-runtime / IGC | 26.35.39758.11 / IGC 2.41.9; Level Zero loader 1.32.0 |
| Kernel / `xe` driver / GuC firmware | 7.0.0-34-generic / in-tree `xe` / GuC 70.58.0 |
| oneCCL | not used |
| PyTorch XPU | not used |
| Power cap per card | 230 W |
| Tensor / pipeline parallel | layer split across 4 cards, `-ub 1536 -ts 12,13,13,11 -fit off`, MTP n-max 3 on `SYCL3`, f16 KV, `GGML_SYCL_SPARSE_FA=1`, pooled QSA key cache on |
| GPU clock floor | 2800 MHz |

## Procedure

1. One server session per arm, each with its own cold read of the 134,862-token context, then 10 turns of 256 greedy tokens with [`tools/lcbench.py`](../../../tools/lcbench.py) (`LC_TURNS=10`). Arms: `LLAMA_MTP_WINDOW=0` with every row attending (`lc-mtpwin-0-allrows`), `LLAMA_MTP_WINDOW=2048`, `LLAMA_MTP_WINDOW=8192`, then `LLAMA_MTP_WINDOW=0` with rows-only attention (`lc-mtpwin-0-rowsonly`). The all-rows and window arms ran on the window code before the rows-only change was added; the rows-only arm ran on the committed 0008 (order from the test-window outputs; source: session log, not archived).
2. Acceptance, milliseconds per MTP step, overall decode and the prompt median per session with [`tools/lcsum.py`](../../../tools/lcsum.py): steps = draft tokens / 3, so ms per step = decode ms / steps; overall tok/s = decoded tokens / decode ms over turns 1-10.
3. Production restored to `20260925-f47a6a5f3` after each window.

## Results

135K, 10 turns per arm ([raw/2026-09-26-lcbench-sessions-summary.csv](../benchmarks/raw/2026-09-26-lcbench-sessions-summary.csv); per-turn values, including the min-max and the ~400-token turns, from [raw/2026-09-26-lcbench-sessions-per-turn.csv](../benchmarks/raw/2026-09-26-lcbench-sessions-per-turn.csv)):

| Arm (session) | Cold read | Acceptance (accepted / drafted) | ms per MTP step | Decode overall | Decode median (min-max) | Prompt, ~60 tok (median) | Prompt, ~400 tok |
|---|---|---|---|---|---|---|---|
| Dense, all rows, W=0 (`lc-mtpwin-0-allrows`) | 403.26 s | 69.7% (1722 / 2472) | 73.4 | 42.3 tok/s | 42.3 (37.8-48.4) | 1.24 s | 2.50-2.82 s |
| Dense, rows only, W=0 (`lc-mtpwin-0-rowsonly`) | 454.37 s | 65.2% (1684 / 2582) | 72.7 | 40.9 tok/s | 41.4 (36.0-46.4) | 1.27 s | 2.66-2.94 s |
| All rows, W=2048 (`lc-mtpwin-2048`) | 458.60 s | 61.7% (1651 / 2678) | 68.4 | 41.9 tok/s | 42.3 (39.3-44.0) | 0.98 s | 2.38-2.64 s |
| All rows, W=8192 (`lc-mtpwin-8192`) | 461.98 s | 65.6% (1687 / 2572) | 69.2 | 43.1 tok/s | 43.0 (39.2-48.2) | 0.99 s | 2.40-2.67 s |

| Metric | Baseline (all rows) | Rows only | Delta | W=2048 | Delta | W=8192 | Delta |
|---|---|---|---|---|---|---|---|
| MTP step time | 73.4 ms | 72.7 ms | -0.7 ms (-1%) | 68.4 ms | -5.0 ms (-7%) | 69.2 ms | -4.2 ms (-6%) |
| Acceptance | 69.7% | 65.2% | -4.5 points | 61.7% | -8.0 points | 65.6% | -4.1 points |
| Single-stream decode, overall | 42.3 tok/s | 40.9 tok/s | -3% | 41.9 tok/s | -1% | 43.1 tok/s | +2% |
| Aggregate at c8 / c16 | not applicable (one slot) | | | | | | |

- **Rows only.** The step got 0.7 ms shorter, close to the estimate. Decode and acceptance moved by less than sessions of one configuration differ, so those differences are not resolved: in the [agent-session benchmark](../benchmarks/2026-09-25-agent-session-135k.md), two arms of the same build had the same step time and 82.7% against 62.5% acceptance (`lc-poolA`, `lc-poolC`), and another pair 71.8% against 64.6% (`lc-spA`, `lc-spC`), because the generated text differed ([raw/2026-09-26-lcbench-sessions-summary.csv](../benchmarks/raw/2026-09-26-lcbench-sessions-summary.csv)). N = 1 session per arm.
- **Window.** The step got 4.2-5.0 ms shorter, as estimated. Acceptance was 4.1-8.0 points lower, which is not resolved: it is within the same-configuration spread above, at N = 1 session per arm. Overall decode was 41.9 and 43.1 tok/s against 42.3 tok/s. That the draft loses useful far context is plausible, not shown. Not adopted.
- **Window prompt turns.** Both window arms took about 0.25 s less per ~60-token turn (0.98-0.99 s against 1.24-1.27 s). That is more than the draft's attention can explain. It was later attributed to the per-turn context checkpoint that the server saves: the plain draft cache copies its whole KV into every checkpoint, and a sliding-window cache holds fewer cells (UNVERIFIED for the window arm; the mechanism was confirmed for the QSA draft memory, see [QSA in the MTP draft head](2026-09-25-mtp-draft-qsa.md)).
- **Cold reads** vary from 403 to 462 s between these four sessions of near-identical code. Use the dedicated cold-read runs, not these, for prefill comparisons.

### `LLAMA_HOST_PROF`

The switch was added to see whether host-side input preparation limits prompt ubatches. One reading on a later build (`85b26781b`, ~135K cold read, 1,536-token ubatches): target `compute` 2,332.9 ms per ubatch, target `set_inputs` 31.9 ms (of which the QSA input 15.6 ms, the hybrid-memory input 16.2 ms and the attention KV input 15.4 ms), draft `compute` 249.5 ms, draft `set_inputs` 17.9 ms (source: session log, not archived). Host input preparation is about 1.4% of a prompt ubatch, so it is not the prefill limit. The `[HOST-PROF]` timings include queue waits on the host, so use them as shares, not as GPU times.

## Correctness

- Rows-only attention is exact by construction: the rows it no longer attends had their attention output discarded before, and every row still stores its K and V. The target verifies every draft token, so neither change can alter the target's output distribution; only draft acceptance and speed change.
- No teacher-forced KL (Kullback-Leibler divergence) probe was run for these arms (none in [raw/2026-09-26-klprobe-comparisons.csv](../benchmarks/raw/2026-09-26-klprobe-comparisons.csv)); given the argument above, none was needed. It would not have tested the draft anyway: [`tools/klprobe.py`](../../../tools/klprobe.py) sends one-token requests, so no MTP draft step and no multi-token verify batch runs in it, and its results show only the target model's next-token distribution.
- Greedy output is not token-identical between sessions on this model even without a change (MTP batching changes the summation order, QSA top-k breaks ties in an unspecified order), so no token diff was used.
- The window is not exact for the draft: it drops far context, which shows as the acceptance loss.

## Decision

Rows-only attention: kept. It is exact and saves about 0.7 ms per MTP step at 135K (N = 1 session per arm). It is in every later build.

`LLAMA_MTP_WINDOW`: reverted to off. Windows of 2048 and 8192 cut the step by 4-5 ms; acceptance was 4-8 points lower (not resolved at one session per arm) and overall decode did not improve beyond session-to-session noise. The option stays in the code for experiments; setting it also disables the QSA draft of [patch 0009](../configs/patches/0009-qwen4exp-mtp-draft-qsa.diff), which checks the same variable.

`LLAMA_HOST_PROF`: kept, off by default.

## Follow-ups

- [x] Give the draft head QSA with its own indexer, as the reference does: [QSA in the MTP draft head](2026-09-25-mtp-draft-qsa.md).
- [ ] Merge the catch-up replay with draft step 0, so one MTP step runs three draft forwards instead of four. Idea only; cost unmeasured.
- [ ] Confirm the window arms' 0.25 s prompt-turn gain came from smaller context checkpoints (UNVERIFIED).
