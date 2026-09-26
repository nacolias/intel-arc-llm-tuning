# QSA in the MTP draft head

| | |
|---|---|
| **Date** | 2026-09-25 |
| **Model / checkpoint** | huihui-ai abliterated `UD-Q4_K_XL` + unsloth MTP draft shared-Q8_0 + mmproj BF16 |
| **Host** | [quad-b70-5800x-pex88096](../../../hardware/hosts/quad-b70-5800x-pex88096.md) |
| **Status** | kept |
| **Baseline** | the dense draft with rows-only attention ([patch 0008](../configs/patches/0008-qwen4exp-mtp-draft-output-rows-only-attention.diff)): the same build with the runtime off-switch `/tmp/llama-mtp-qsa-off` (paired A/B), and sessions `lc-mtpwin-0-allrows` / `lc-mtpwin-0-rowsonly` in [raw/2026-09-26-lcbench-sessions-summary.csv](../benchmarks/raw/2026-09-26-lcbench-sessions-summary.csv) |
| **Result** | [raw/2026-09-25-pairprobe-draft-qsa.csv](../benchmarks/raw/2026-09-25-pairprobe-draft-qsa.csv) (paired A/B), row `lc-draftqsa` in [raw/2026-09-26-lcbench-sessions-summary.csv](../benchmarks/raw/2026-09-26-lcbench-sessions-summary.csv), [raw/2026-09-25-draft-qsa-fnbench.csv](../benchmarks/raw/2026-09-25-draft-qsa-fnbench.csv) |
| **Related** | [patch 0009](../configs/patches/0009-qwen4exp-mtp-draft-qsa.diff), [research: long-context QSA design](../research/2026-09-25-long-context-qsa-design.md) (design 4, "designed, not built" there), [rows-only attention and window](2026-09-25-mtp-draft-rows-only-and-window.md), [pooled QSA key cache](2026-09-25-pooled-qsa-key-cache.md) (the memory class this reuses), [sparse FA](2026-09-25-sparse-fa-multi-token.md) |

## Hypothesis

The draft was running off-spec. The reference implementation builds the MTP (multi-token prediction) layer, layer 48, as a full QSA (Qwen sparse attention) layer with its own trained indexer, and the draft GGUF ships those indexer weights (`blk.48.indexer.*`). The GGUF converter wrote compress ratio 0 for that layer, so llama.cpp attended densely over the whole context: 1.87 ms per call, 9.4% of decode op time at 135K on build `20260925-60a598ed8` ([raw/2026-09-25-op-profile-135k.csv](../benchmarks/raw/2026-09-25-op-profile-135k.csv)). Selecting the draft's ~2,051 cells like the trunk layers should cut each of the three draft attention calls per MTP step to about 1 ms of indexer work: 5-6 ms per step by the [research note's](../research/2026-09-25-long-context-qsa-design.md) estimate, or 0-2.9 ms on top of rows-only attention by the later plan (estimate; source: session log, not archived). Acceptance could move either way: below the 2048-token budget dense attention is a numerical superset of QSA, above it the trained indexer picks what the head was trained with.

What the reference does (vLLM v0.30.0 source, read during the session; the research note records that layer 48 is QSA there):

| Property | Reference | llama.cpp before 0009 | After 0009 |
|---|---|---|---|
| Layer 48 attention | QSA with its own indexer | dense over the whole context (compress ratio 0 from the converter) | QSA, ratio 4 copied from the trunk at load |
| Indexer parameters | the shared text config: ratio 4, budget 2048 tokens (512 blocks), 4 x 128 query heads, one 128-wide key head (values as on the [model card](../README.md)) | indexer weights in the GGUF, unused | the draft GGUF's own indexer weights |
| Caches | its own raw-key, pooled-key and top-k buffers, never the target's | plain `llama_kv_cache` | attention + indexer + pooled keys in an attention-only `llama_memory_hybrid_idx` |
| Selection | every draft step runs its own top-k; `index_share` is off for this checkpoint | none | every one-token draft decode selects |
| Coverage | the draft KV covers every position; selection ranks the whole prefix | same | same |
| Position grouping | reference position L = llama.cpp draft cell P - 1; cell 0 has no reference counterpart | not applicable | not implemented: blocks are grouped by llama.cpp positions (effect UNVERIFIED) |

The draft's indexer norm weights are not identity values, so the indexer was trained (read from the GGUF in the session; source: session log, not archived).

## Change

[Patch 0009](../configs/patches/0009-qwen4exp-mtp-draft-qsa.diff), commit message: "qwen4exp MTP draft: QSA with its own indexer (attention-only hybrid_idx draft memory, ratio override, sparse FA); /tmp/llama-mtp-qsa-off A/B switch". On by default.

- **Load** (`src/models/qwen4exp.cpp`, `load_arch_hparams`). When the trunk's non-recurrent layers share one non-zero compress ratio, every non-recurrent nextn layer with ratio 0 gets it. Logged as "MTP layer 48 attends with QSA (compress ratio 4)". `LLAMA_MTP_QSA=0` skips this and keeps the dense draft.
- **Memory** (`src/llama-model.cpp`, `create_memory`; `src/llama-memory-hybrid-idx.{h,cpp}`). The draft's memory becomes a `llama_memory_hybrid_idx` in a new attention-only mode (`set_attn_only()`): attention cache, indexer cache and pooled-key buffer for layer 48, zero recurrent layers. In that mode `seq_rm` clears the empty recurrent cache instead of asking it, because an empty recurrent cache refuses a partial `seq_rm`, which would make the server classify the draft context as full-state and abort on partial acceptance (traced in code during the session, not run; source: session log, not archived). `seq_pos_min` / `seq_pos_max` come from the attention cache. Not built when `LLAMA_MTP_WINDOW` is set.
- **Graph** (`graph_mtp`). A new `build_attn_inp_kv_hyb()` input reads the hybrid context's attention part (the trunk's `build_attn_inp_kv()` casts to the wrong context type). When `n_outputs > 0`, `n_tokens <= 8` and the pooled plan is fast, the draft runs the trunk's `build_qsa_top_k` and `build_attn_qsa` unchanged, so it selects its cells and attends sparsely. Otherwise it stores the raw indexer key (`index_k_proj`) through a new `llm_graph_input_idx_k` input, so later selections see a complete indexer cache, and attends densely on its output rows only (the 0008 path).
- **Why the `n_tokens <= 8` gate.** The draft reserves its prompt graph at 1,536 tokens. QSA temporaries scale with `n_tokens`: one `[262,144 x 1,536]` F32 tensor is 1.5 GiB (arithmetic), several of them would not fit `SYCL3`. Reserve graphs are never fast, so both stay dense.
- **Switches.** `LLAMA_MTP_QSA=0` at start. `/tmp/llama-mtp-qsa-off` at runtime: while the file exists (polled about once a second) the draft's plans are never fast, so it attends densely; the indexer cache keeps filling.
- **Memory cost.** 64 MiB of indexer cache (262,144 cells x 128 x f16, arithmetic) plus a 32 MiB pooled-key buffer on `SYCL3`: +96 MiB on that card after load ([raw/2026-09-25-draft-qsa-fnbench.csv](../benchmarks/raw/2026-09-25-draft-qsa-fnbench.csv), header).
- **Not implemented.** The reference position offset (L = P - 1) and masking of cell 0. The shipped draft groups blocks by llama.cpp positions, one off from the reference. Effect on acceptance UNVERIFIED.

```diff
+ configs/patches/0009-qwen4exp-mtp-draft-qsa.diff
  # A/B switches: LLAMA_MTP_QSA=0 (start), touch /tmp/llama-mtp-qsa-off (runtime)
```

## Environment

| Component | Version / value |
|---|---|
| Engine and image (tag and digest) | llama.cpp, no container. Sandbox build of the patch series through 0009 (commit `0f23f62c8`): PR #28243 at `6fcaa16` + #28931 + 0002 + 0003 + 0004 + #28796 + 0006 + 0007 + 0008 + 0009. Frozen and deployed the same day as `20260925-0f23f62c8` |
| Build flags | `GGML_SYCL=ON GGML_SYCL_TARGET=INTEL GGML_SYCL_F16=ON GGML_SYCL_DNN=ON GGML_SYCL_GRAPH=ON GGML_SYCL_SUPPORT_LEVEL_ZERO_API=ON GGML_NATIVE=ON`, Release, icx/icpx from oneAPI 2026.1 |
| compute-runtime / IGC | 26.35.39758.11 / IGC 2.41.9; Level Zero loader 1.32.0 |
| Kernel / `xe` driver / GuC firmware | 7.0.0-34-generic / in-tree `xe` / GuC 70.58.0 |
| oneCCL | not used |
| PyTorch XPU | not used |
| Power cap per card | 230 W |
| Tensor / pipeline parallel | layer split across 4 cards, `-ub 1536 -ts 12,13,13,11 -fit off`, MTP n-max 3 on `SYCL3`, f16 KV, `GGML_SYCL_SPARSE_FA=1`, pooled QSA key cache on |
| GPU clock floor | 2800 MHz |

## Procedure

1. Start-up checks in the server log: the ratio-override line, "MTP draft memory: attention + QSA indexer (attention-only hybrid)", and a 32 MiB QSA pool buffer on `SYCL3` (source: session log, not archived).
2. [`tools/fnbench.py`](../../../tools/fnbench.py) with three short repetitions: sanity question, short, ~10k and ~39k.
3. One session of [`tools/lcbench.py`](../../../tools/lcbench.py) at 135K, 10 turns of 256 greedy tokens (`lc-draftqsa`), summarised with [`tools/lcsum.py`](../../../tools/lcsum.py) and set against the two dense sessions of the same day.
4. Paired A/B with [`tools/pairprobe.py`](../../../tools/pairprobe.py) in one server session: a cold read of the same 134,822-token context, then 12 turns. Each turn appends a tool-output snippet and a fixed task line, decodes 192 tokens greedily with QSA on, then the same 192 tokens with `/tmp/llama-mtp-qsa-off` present, and continues the conversation with the QSA answer. Both arms see the same context and mostly the same text, so the comparison is not dominated by differences in the generated text, which is what moves acceptance between separate sessions. The tool's docstring puts that session-to-session noise at about 3 points; archived sessions of one configuration differ by more (see Results).
5. Adversarial code review of the memory mode and the graph branch.
6. Freeze the build, deploy, keep `20260925-f47a6a5f3` in the rollback list.

## Results

**Paired A/B at 135K**, 12 turns x 192 tokens per arm, one session ([raw/2026-09-25-pairprobe-draft-qsa.csv](../benchmarks/raw/2026-09-25-pairprobe-draft-qsa.csv)):

| Metric | Dense draft (off-file) | Draft QSA | Delta |
|---|---|---|---|
| Acceptance (accepted / drafted) | 65.7% (1515 / 2307) | 67.6% (1531 / 2264) | +1.9 points |
| MTP step time | 71.5 ms | 70.3 ms | -1.2 ms (-1.7%) |
| Single-stream decode, overall | 41.9 tok/s | 43.4 tok/s | +3.6% |
| Decode per turn, range | 39.4-45.0 tok/s | 39.9-48.1 tok/s | |
| Identical greedy outputs | 2 of 12 turns | | |
| Aggregate at c8 / c16 | not applicable (one slot) | | |

The prompt column of that file is not a prompt-turn measurement: the QSA arm processes each turn's new tokens, the dense arm re-decodes on the cached prefix.

**Separate lcbench sessions at 135K**, 10 turns each ([raw/2026-09-26-lcbench-sessions-summary.csv](../benchmarks/raw/2026-09-26-lcbench-sessions-summary.csv); per-turn values in [raw/2026-09-26-lcbench-sessions-per-turn.csv](../benchmarks/raw/2026-09-26-lcbench-sessions-per-turn.csv)):

| Session | Draft | Cold read | Acceptance | ms per MTP step | Decode overall | Decode median (min-max) | Prompt, ~60 tok (median) | Prompt, ~400 tok |
|---|---|---|---|---|---|---|---|---|
| `lc-mtpwin-0-allrows` | dense, all rows | 403.26 s | 69.7% | 73.4 | 42.3 tok/s | 42.3 (37.8-48.4) | 1.24 s | 2.50-2.82 s |
| `lc-mtpwin-0-rowsonly` | dense, rows only | 454.37 s | 65.2% | 72.7 | 40.9 tok/s | 41.4 (36.0-46.4) | 1.27 s | 2.66-2.94 s |
| `lc-draftqsa` | QSA | 436.80 s | 63.1% (1664 / 2637) | 70.5 | 41.3 tok/s | 41.1 (39.6-44.7) | 0.92 s | 2.34-2.58 s |

- **Step time** fell by 2.2 ms against rows-only in separate sessions and by 1.2 ms in the paired run: three dense 1.8 ms attention calls became roughly 1 ms of indexer work each. That is at the low end of the estimates.
- **Acceptance.** The paired run says +1.9 points, with the QSA arm ahead in 9 of 12 turns (per-turn differences -6.0 to +12.7 points, derived from [raw/2026-09-25-pairprobe-draft-qsa.csv](../benchmarks/raw/2026-09-25-pairprobe-draft-qsa.csv)). That paired A/B is the stronger evidence; it is still one session. The separate sessions (63.1% against 65.2% and 69.7%) point the other way, but differences of a few points between separate sessions are not resolved: sessions of one configuration differ by more, for example `lc-spA` 71.8% against `lc-spC` 64.6%, and `lc-poolA` 82.7% against `lc-poolC` 62.5% ([raw/2026-09-26-lcbench-sessions-summary.csv](../benchmarks/raw/2026-09-26-lcbench-sessions-summary.csv)). Acceptance between sessions moves with the generated text.
- **Prompt turns** got about 0.35 s faster: 0.92 s against 1.24-1.27 s for ~60 new tokens (-26% to -28%), and 2.34-2.58 s against 2.50-2.94 s for ~400 tokens. The draft's attention cannot explain that much. The cause is the server's per-turn context checkpoint. The plain draft `llama_kv_cache` ignores the `PARTIAL_ONLY` flag of `update_dft`, so every checkpoint copied the draft's whole KV. Checkpoint sizes logged at verbosity 4 grew from 112.6 MiB at 15 tokens to 542.7 MiB at 217,677 tokens, about 2,072 bytes per token, which is about 270 MiB per checkpoint at 135K (estimate; source: session log, not archived). The hybrid memory honours the flag, as it does for the target. The mechanism was verified by reading the llama.cpp source in the session, not by a separate measurement (source: session log, not archived).
- **Short context**, [`tools/fnbench.py`](../../../tools/fnbench.py) on this build ([raw/2026-09-25-draft-qsa-fnbench.csv](../benchmarks/raw/2026-09-25-draft-qsa-fnbench.csv)): short 46.3 tok/s (45.9-47.2, one run of three repetitions), ~10k 59.4 tok/s (acceptance 182 / 219), ~39k 54.9 tok/s (180 / 222); prompt 589.6 / 579.8 tok/s. The previous production build spans 48.6-50.5 / 60.6-67.2 / 53.6-55.2 over three sessions ([production build](../benchmarks/2026-09-25-production-build.md)). At ~10k and ~39k this is inside that spread. Short is 2 tok/s below the spread in a single run; whether that is a real cost is UNVERIFIED. Below the 2048-token budget QSA selects every block, so the attention itself is unchanged there and only the indexer work is added.

## Correctness

- The target verifies every draft token, so the target's output distribution cannot change; the change can only move acceptance and speed. Greedy outputs were identical in 2 of 12 paired turns: a different draft changes the verify batching and with it the fp16 summation order, as seen in every earlier MTP experiment on this model. No teacher-forced KL (Kullback-Leibler divergence) probe was run for this change (none in [raw/2026-09-26-klprobe-comparisons.csv](../benchmarks/raw/2026-09-26-klprobe-comparisons.csv)). Later builds that include 0009 were probed against the pooled-cache baseline and stayed within run-to-run noise (see that file, rows `kl-ident`, `kl-mmid`, `kl-mmid18`, `kl-eh`). That probe ([`tools/klprobe.py`](../../../tools/klprobe.py)) sends one-token requests (`n_predict` 1), so no MTP draft step and no multi-token verify batch runs in it. Its results show only that the target model's next-token distribution is unchanged, not that the draft is.
- Adversarial code review of the attention-only memory mode and the graph branch: no defects found. One limitation: after an image in the conversation (repeated M-RoPE positions), the pooled plan is not fast, so the draft attends densely for some positions and only stores its indexer keys. The same fast-path scope as [patch 0007](../configs/patches/0007-qwen4exp-pooled-qsa-key-cache.diff).
- The start-up log confirmed the ratio override, the attention-only draft memory and the 32 MiB pool on `SYCL3`; a sanity question was answered correctly (source: session log, not archived).
- Not done: draft-shaped `test-backend-ops` cases, a selection-parity check against the reference, an image-prompt test and a teacher-forced replay of the draft were planned and not run.

## Decision

Kept. In one session the draft with QSA decoded 3.6% faster than the dense draft at 135K (43.4 against 41.9 tok/s) with 1.9 points more acceptance and a 1.2 ms shorter step, and ~60-token prompt turns fell from about 1.25 s to 0.92 s because the draft's checkpoints stopped copying its whole KV. It costs 96 MiB on `SYCL3`. Deployed as build `20260925-0f23f62c8` on 2026-09-25 and part of every later build; patches 0010-0017 apply on top of it. `LLAMA_MTP_QSA=0` gives the old dense draft.

## Follow-ups

- [ ] Reference grouping: position offset L = P - 1 and masking of cell 0. Not implemented; the effect on acceptance is unmeasured.
- [ ] `index_share` (select at draft step 0, reuse for steps 1-2), an opt-in mode in vLLM that is off for this checkpoint. Try only if a paired A/B shows no acceptance loss.
- [ ] Bound the draft's QSA temporaries by `n_outputs` instead of the `n_tokens <= 8` gate, which would shrink the dense prompt reserve on `SYCL3`. Conflicts with reusing the trunk's QSA code verbatim.
- [ ] `common_speculative_impl_draft_mtp` never resets its pending hidden state on a new or diverging request, so one draft KV cell can be wrong after a checkpoint restore. Mechanism read in code; impact unmeasured; no patch.
- [ ] Run the planned draft-shaped `test-backend-ops` cases and a selection-parity check against the reference.
- [ ] The short-prompt reading of 46.3 tok/s (one run) against 48.6-50.5 for the previous build: repeat before calling it a cost.
