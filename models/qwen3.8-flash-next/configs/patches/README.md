# llama.cpp patches for Qwen3.8-Flash-Next on SYCL

The production build is upstream PR #28243 plus two upstream cherry-picks, one copied upstream PR and twenty of our own diffs. The twenty-one diffs (0002-0023, minus the two cherry-picks) are in this folder. They apply cleanly in the order below. We checked this twice by applying the diffs to the pre-image files and comparing the result with the frozen build's tree: on 2026-09-25 for 0002-0007 (identical except for two reworded code comments), and on 2026-09-26 for the whole stack 0002-0017 in order, which matched the production tree `2b84213a4` except for the same two reworded comments and three whitespace wraps. The wraps: three assignments in 0003 and 0004 were wrapped onto two lines so the repository's secret scanner does not mistake them for keys; the code is unchanged.

Build tag on our host: `20260926-5c258f538` (the commit hash of the last patch in our tree), in production since 2026-09-26 22:10 UTC. It holds 0002-0023; 0018, 0019, 0020 and 0023 are compiled in but do nothing unless their environment variable is set. The stack 0018-0023 applied to `2b84213a4` reproduces that tree exactly (checked 2026-09-26). 0020 is exported with one line of context and without git's function labels on the hunk headers, so the repository's secret scanner does not mistake an unchanged `..._multitoken_max = ...` line for a key; it applies the same way. The build before it, `20260926-2b84213a4` (0002-0017), ran from earlier on 2026-09-26. PR #2 of this repository documented the stack up to 0007, build `20260925-f47a6a5f3`. Intermediate frozen builds: `20260925-993baf141` (after 0010) and `20260926-2c88bdf19` (after 0011).

## Apply order

The `.diff` files are plain diffs. Their commit messages, which several experiment files quote, are kept in [COMMIT-MESSAGES.md](COMMIT-MESSAGES.md).

| Order | Change | Source | What it does |
|---|---|---|---|
| 0 | base | [ggml-org/llama.cpp PR #28243](https://github.com/ggml-org/llama.cpp/pull/28243) at `6fcaa16` | qwen4exp (Qwen3.8-Flash-Next) support plus the MTP (multi-token prediction) draft path. The PR was open when we used it. |
| 1 | upstream cherry-pick | `git cherry-pick 5e48b3100` ([#28931](https://github.com/ggml-org/llama.cpp/pull/28931), merged) | SYCL fusions: MMVQ GLU fusion for mixed quant types, rms_norm+scale, ssm_conv+silu. On this model the rms_norm+scale fusion fires on the Gated DeltaNet L2 norms. |
| 2 | ours | [0002-sycl-gettid-cache.diff](0002-sycl-gettid-cache.diff) | Caches `gettid()` per thread in dpct's `get_tid()`. The raw syscall ran on every op. |
| 3 | ours | [0003-sycl-fused-mul-mat-id-2-8-tokens.diff](0003-sycl-fused-mul-mat-id-2-8-tokens.diff) | Runs `MUL_MAT_ID` batches of 2 to 8 tokens (MTP verify passes, short prompts) through the fused single-token MMVQ path once per token, instead of the per-expert loop that copies expert ids to the host and waits first. |
| 4 | ours | [0004-sycl-op-profile.diff](0004-sycl-op-profile.diff) | Diagnostics only: `GGML_SYCL_OP_PROFILE` per-op timing and the `GGML_SYCL_MMID_MULTITOKEN_MAX` limit. Both are off by default. |
| 5 | upstream cherry-pick | `git cherry-pick cd74ef627` ([#28796](https://github.com/ggml-org/llama.cpp/pull/28796), merged) | SYCL sparse flash attention for single-token decode, enabled with `GGML_SYCL_SPARSE_FA=1`. |
| 6 | ours | [0006-sycl-sparse-fa-multi-token.diff](0006-sycl-sparse-fa-multi-token.diff) | Extends #28796 to batches of up to 32 query rows: per-row compaction and gather of the selected cells, each row re-dispatched as its own sequence to the unchanged TILE kernel. Adds qwen4-shaped `test-backend-ops` cases. |
| 7 | ours | [0007-qwen4exp-pooled-qsa-key-cache.diff](0007-qwen4exp-pooled-qsa-key-cache.diff) | Caches the pooled QSA (Qwen sparse attention) indexer key per 4-token position block, so a ubatch pools only the blocks it completes instead of every block in the cache. |
| 8 | ours | [0008-qwen4exp-mtp-draft-output-rows-only-attention.diff](0008-qwen4exp-mtp-draft-output-rows-only-attention.diff) | MTP draft: only output rows attend, so catch-up and prompt passes with no outputs skip the attention scan of the draft cache. Adds the optional `LLAMA_MTP_WINDOW=<tokens>` sliding window for the draft (off by default; tested and not adopted) and `LLAMA_HOST_PROF=<seconds>` host-side phase timing. |
| 9 | ours | [0009-qwen4exp-mtp-draft-qsa.diff](0009-qwen4exp-mtp-draft-qsa.diff) | QSA in the MTP draft head. The GGUF converter wrote compress ratio 0 for the draft layer (48), so it attended densely; at load it now gets the trunk's ratio (4). The draft context gets an attention-only `llama_memory_hybrid_idx` memory with its own indexer cache and pooled keys, and one-token draft decodes attend sparsely. `LLAMA_MTP_QSA=0` or the file `/tmp/llama-mtp-qsa-off` gives the dense draft. |
| 10 | ours | [0010-server-slot-save-restore-draft-context.diff](0010-server-slot-save-restore-draft-context.diff) | llama-server slot save/restore also saves and restores the speculative draft's context (`<file>.dft`), so a restored slot drafts with the prefix instead of starting the draft on whatever it held before. |
| 11 | ours | [0011-qwen4exp-qsa-score-expansion-by-broadcast.diff](0011-qwen4exp-qsa-score-expansion-by-broadcast.diff) | QSA per-cell score expansion by broadcast when every cached cell sits at its own position. Replaces a `get_rows` over a transposed copy plus a transpose back (two `cont(permute)` per QSA layer and ubatch). |
| 12 | upstream PR, copied | [0012-upstream-pr29245-sycl-grouped-moe-xmx-gemm.diff](0012-upstream-pr29245-sycl-grouped-moe-xmx-gemm.diff) | [ggml-org/llama.cpp PR #29245](https://github.com/ggml-org/llama.cpp/pull/29245) at `950f1c4aa` (open on 2026-09-26): a grouped MoE (mixture-of-experts) XMX (Xe Matrix Extensions) GEMM, one launch per `MUL_MAT_ID` node with the weights dequantized inside the XMX tiles. IQ formats only; selected by the `GGML_SYCL_XMX_GATHER_TYPES` bitmask. Copied because an open PR's head can be force-pushed; its two trivial conflicts against 0003 and 0004 are resolved by keeping both sides. |
| 13 | ours | [0013-sycl-grouped-moe-gemm-q4k-q5k-q51-q80.diff](0013-sycl-grouped-moe-gemm-q4k-q5k-q51-q80.diff) | Grouped GEMM decoders for the formats this model keeps its experts in: Q4_K and Q5_K (plain and lazily reordered SoA layouts), Q5_1 and Q8_0; reorders on the grouped path too; bits 9-12 of `GGML_SYCL_XMX_GATHER_TYPES`. |
| 14 | ours | [0014-sycl-grouped-moe-gemm-vector-loads.diff](0014-sycl-grouped-moe-gemm-vector-loads.diff) | Vector loads in the q4_K/q5_K/q5_1/q8_0 A stages (16-byte loads for the K-quants, 8-byte plus a dword for q5_1, pairs for q8_0). |
| 15 | ours | [0015-sycl-grouped-moe-gemm-in-place-io.diff](0015-sycl-grouped-moe-gemm-in-place-io.diff) | The B pack reads the routed `src1` rows in place through the row mapping via local memory and the GEMM writes `dst` rows in place; the contiguous copies remain only for the per-expert loop. |
| 16 | ours | [0016-qwen4exp-mtp-eh-proj-2d-product.diff](0016-qwen4exp-mtp-eh-proj-2d-product.diff) | MTP `eh_proj` as one 2D product. A 3D `src1` made SYCL `MUL_MAT` run one product per token, each reading the whole 14 MB Q8_0 weight: 115 ms per 1536-token ubatch ([raw op profile](../../benchmarks/raw/2026-09-26-prefill-op-profile.csv)). |
| 17 | ours | [0017-sycl-sparse-fa-union-stats-debug.diff](0017-sycl-sparse-fa-union-stats-debug.diff) | Diagnostics only: `GGML_SYCL_SPARSE_FA_DEBUG=2` logs how many cells tiles of 1, 16, 32, 64 and 128 consecutive query rows of a prompt batch select together; one call in 12. |
| 18 | ours, test hook (inert unless set) | [0018-mtp-spec-override-test-hook.diff](0018-mtp-spec-override-test-hook.diff) | `LLAMA_SPEC_OVERRIDE=<file>`: the MTP draft re-reads `n_max p_min` from the file on every draft call, capped at `--spec-draft-n-max`, so draft settings can be A/B-tested within one server session ([experiment](../../experiments/2026-09-26-mtp-draft-settings-sampled-replay.md)). Off unless the variable is set. |
| 19 | ours, diagnostics | [0019-prompt-cache-swap-timing-logs.diff](0019-prompt-cache-swap-timing-logs.diff) | Logs, for states of 64 MiB and more, how long a host-RAM prompt-cache swap spends allocating the host copy, walking the state and copying device to host and host to device. |
| 20 | ours, diagnostics (inert unless set) | [0020-sycl-op-devprof.diff](0020-sycl-op-devprof.diff) | `GGML_SYCL_OP_DEVPROF=<seconds>`: device-side op times from the GPU timestamps of a barrier submitted before each op, read after the graph finished, without host waits. Unlike `GGML_SYCL_OP_PROFILE` (0004) it does not serialize the ops. Creates the queues with profiling on only when the variable is set; runs while `/tmp/ggml-sycl-op-devprof` exists. |
| 21 | ours | [0021-server-prompt-cache-buffer-reuse.diff](0021-server-prompt-cache-buffer-reuse.diff) | The host-RAM prompt cache (`--cache-ram`) keeps the host buffers of the state it just loaded back into a slot and reuses them for the next save (matched by capacity; fresh buffers get 1/8 untouched slack). A fresh 4 GiB `std::vector` cost about 2 s in page faults and zero-fill per swap ([experiment](../../experiments/2026-09-26-prompt-cache-swap-speedup.md)). |
| 22 | ours | [0022-sycl-pinned-staged-set-tensor.diff](0022-sycl-pinned-staged-set-tensor.diff) | `ggml_backend_sycl_buffer_set_tensor` copies through two reusable pinned 32 MiB buffers per device, overlapping the host copy of one chunk with the device copy of the other, instead of a fresh `malloc` per call. Speeds up prompt-cache swap-ins (2.7 to 0.43 s at 135K), slot restores (3.2 to 1.0 s) and model loading (start to ready about 75 to 33 s). |
| 23 | ours, diagnostics (inert unless set) | [0023-sycl-op-devprof-coarse.diff](0023-sycl-op-devprof-coarse.diff) | Coarse mode for 0020: when `/tmp/ggml-sycl-op-devprof` starts with `coarse`, one barrier before a graph's first op and one after its last, so each graph reports one device span and its host submit time without slowing the step. |

```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
git fetch origin pull/28243/head && git checkout -b flash-next 6fcaa16
P=/path/to/configs/patches
git cherry-pick 5e48b3100                                            # #28931
git apply --index $P/0002-sycl-gettid-cache.diff              && git commit -qm "0002"
git apply --index $P/0003-sycl-fused-mul-mat-id-2-8-tokens.diff && git commit -qm "0003"
git apply --index $P/0004-sycl-op-profile.diff                && git commit -qm "0004"
git cherry-pick cd74ef627                                            # #28796
git apply --index $P/0006-sycl-sparse-fa-multi-token.diff     && git commit -qm "0006"
git apply --index $P/0007-qwen4exp-pooled-qsa-key-cache.diff  && git commit -qm "0007"
git apply --index $P/0008-qwen4exp-mtp-draft-output-rows-only-attention.diff && git commit -qm "0008"
git apply --index $P/0009-qwen4exp-mtp-draft-qsa.diff                    && git commit -qm "0009"
git apply --index $P/0010-server-slot-save-restore-draft-context.diff     && git commit -qm "0010"
git apply --index $P/0011-qwen4exp-qsa-score-expansion-by-broadcast.diff  && git commit -qm "0011"
git apply --index $P/0012-upstream-pr29245-sycl-grouped-moe-xmx-gemm.diff && git commit -qm "0012 (#29245 at 950f1c4aa)"
git apply --index $P/0013-sycl-grouped-moe-gemm-q4k-q5k-q51-q80.diff      && git commit -qm "0013"
git apply --index $P/0014-sycl-grouped-moe-gemm-vector-loads.diff         && git commit -qm "0014"
git apply --index $P/0015-sycl-grouped-moe-gemm-in-place-io.diff          && git commit -qm "0015"
git apply --index $P/0016-qwen4exp-mtp-eh-proj-2d-product.diff            && git commit -qm "0016"
git apply --index $P/0017-sycl-sparse-fa-union-stats-debug.diff           && git commit -qm "0017"
git apply --index $P/0018-mtp-spec-override-test-hook.diff               && git commit -qm "0018"
git apply --index $P/0019-prompt-cache-swap-timing-logs.diff             && git commit -qm "0019"
git apply --index $P/0020-sycl-op-devprof.diff                           && git commit -qm "0020"
git apply --index $P/0021-server-prompt-cache-buffer-reuse.diff          && git commit -qm "0021"
git apply --index $P/0022-sycl-pinned-staged-set-tensor.diff             && git commit -qm "0022"
git apply --index $P/0023-sycl-op-devprof-coarse.diff                    && git commit -qm "0023"
```

Commit after each diff: #28796 touches `ggml-sycl.cpp`, which 0003 and 0004 also change, and `git cherry-pick` refuses to run over uncommitted changes to the same file. Both cherry-picked commits are on upstream `master`, so a plain clone can reach them. 0012 is applied from the copied diff, not with `git cherry-pick 950f1c4aa`: the PR is open, so a plain clone does not have that commit and the head may move. The copy already carries the conflict resolutions against 0003 and 0004.

## Build

```bash
source /opt/intel/oneapi/2026.1/oneapi-vars.sh
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
  -DGGML_SYCL=ON -DGGML_SYCL_TARGET=INTEL -DGGML_SYCL_F16=ON -DGGML_SYCL_DNN=ON -DGGML_SYCL_GRAPH=ON \
  -DGGML_SYCL_SUPPORT_LEVEL_ZERO_API=ON -DGGML_NATIVE=ON
cmake --build build -j16 --target llama-server test-backend-ops llama-bench
```

These are the CMake options of our build, read from its `CMakeCache.txt`. The Level Zero development package (`libze-dev`, 1.32.0 on our host) must be installed, or the Level Zero allocator is compiled out.

## Runtime switches

| Variable or file | Default | Patch | Effect |
|---|---|---|---|
| `GGML_SYCL_MMID_MULTITOKEN=0` | 1 | 0003 | turn the fused 2-8-token `MUL_MAT_ID` path off |
| `GGML_SYCL_MMID_MULTITOKEN_MAX=<n>` | 8 | 0004 | largest batch sent through the per-token fused path. 512 was tested and was slower; see [the experiment](../../experiments/2026-09-25-mmid-multitoken-prompt-turns.md) |
| `GGML_SYCL_OP_PROFILE=<seconds>` | 0 | 0004 | per-op timing, printed every `<seconds>`, only while the file `/tmp/ggml-sycl-op-profile` exists. Each op waits on the queue, so use the shares, not the absolute times |
| `GGML_SYCL_SPARSE_FA=1` | 0 | #28796 | sparse flash attention for QSA layers |
| `GGML_SYCL_SPARSE_FA_MAX_Q=<n>` | 32 | 0006 | largest query batch gathered row by row |
| `/tmp/ggml-sycl-sparse-fa-off` | absent | 0006 | while this file exists, sparse FA stays off (A/B switch without a restart) |
| `LLAMA_QSA_POOLED=0` | on | 0007 | turn the pooled QSA key cache off |
| `/tmp/llama-qsa-pooled-off` | absent | 0007 | while this file exists, every ubatch takes the full re-pooling path; the cache catches up afterwards |
| `LLAMA_MTP_WINDOW=<tokens>` | unset (no window) | 0008 | the draft attends only the last `<tokens>` positions (a standard sliding-window mask; the window also goes to flash attention as the sparse hint). Setting it turns the draft QSA of 0009 off. Windows of 2048 and 8192 cut the MTP step from 73.4 to 68.4 and 69.2 ms but acceptance fell from 69.7% to 61.7% and 65.6%, so it stays off ([raw](../../benchmarks/raw/2026-09-26-lcbench-sessions-summary.csv), `lc-mtpwin-*`; [experiment](../../experiments/2026-09-25-mtp-draft-rows-only-and-window.md)) |
| `LLAMA_HOST_PROF=<seconds>` | 0 | 0008 | host-side time per `process_ubatch` phase (apply, build/reuse, set_inputs, compute) and per graph-input type, split by context (target or draft) and batch-size class, printed every `<seconds>` |
| `LLAMA_MTP_QSA=0` | on | 0009 | dense draft attention (the pre-0009 behaviour); read at load |
| `/tmp/llama-mtp-qsa-off` | absent | 0009 | while this file exists, the draft attends densely (A/B switch without a restart; [tools/pairprobe.py](../../../../tools/pairprobe.py) toggles it per turn) |
| `GGML_SYCL_XMX_GATHER_TYPES=<bitmask>` | all bits set | 0012, 0013 | which expert formats take the grouped XMX GEMM for `MUL_MAT_ID` batches over 8 tokens: bits 1-256 are the IQ formats of #29245; bit 9 (512) Q4_K, bit 10 (1024) Q5_K, bit 11 (2048) Q5_1, bit 12 (4096) Q8_0. `0` restores the per-expert oneDNN loop, the baseline to compare against |
| `GGML_SYCL_SPARSE_FA_DEBUG=2` | 0 | 0017 (extends the #28796 variable; `1` is upstream's debug logging) | for a masked prompt-batch attention call too large for the per-row gather, logs how many cells tiles of 1, 16, 32, 64 and 128 consecutive rows keep; one call in 12 ([raw](../../benchmarks/raw/2026-09-26-fa-selection-union.csv)) |

## Scope and limits

- 0007 takes its fast path only for one stream with one sequence, unique positions and the per-block bias. Anything else, including image (M-RoPE) positions from the vision projector, falls back to the full re-pooling path.
- 0007 keeps llama.cpp's top-k over expanded cells (width 2051). That attends up to 3 cells of a 513th block at 3 of every 4 positions, where the reference attends 512 whole blocks plus the tail. We did not change this. See [research/2026-09-25-long-context-qsa-design.md](../../research/2026-09-25-long-context-qsa-design.md).
- 0006 engages only while `T * n_kv_g * 2 <= n_kv`, where `T` is the number of query rows and `n_kv_g = pad(n_kv_max + 256, 256)` (2560 for this model). Prompt turns of about 60 tokens are above the 32-row limit and still use dense attention.
- 0009 takes the sparse draft path only for draft ubatches of at most 8 tokens that produce outputs and whose pooled plan (0007) is fast (`use_qsa = n_outputs > 0 && n_tokens <= 8 && ... .fast` in the diff). Catch-up passes with no outputs only store keys. Setting `LLAMA_MTP_WINDOW` disables it. After an image in the conversation, the draft falls back to dense attention for the affected positions, because 0007's fast path excludes M-RoPE positions. The draft groups cells by llama.cpp positions; the reference's exact grouping (position offset -1 with cell 0 masked) was not implemented.
- 0010 writes the draft state to `<file>.dft` next to the slot file and adds its size to the reported byte count. On restore, a missing or unreadable `.dft` clears the draft's sequence so the draft starts empty rather than on a stale prompt. A saved slot is only useful to an identical setup; [`llama-server-slot-cache.sh`](../llama-server-slot-cache.sh) compares a signature before restoring. Sizes and times are in [the raw timings](../../benchmarks/raw/2026-09-26-slot-cache-timings.csv). llama-server refuses to save a slot that holds image tokens (observed in production; UNVERIFIED in source).
- 0011's broadcast runs only when every cached cell `j` holds position `j` (`ident` in the diff: one conversation, no cache edits in the middle). Every other case keeps the `get_rows` expansion. In the `ident` case the `cell_blk` input is not in the graph and gets no buffer, so `set_input_qsa_fast` skips it; the first build asserted on the missing buffer (`GGML_ASSERT(buffer)`) until that skip was added (source: session log, not archived). It changes no values: "Same values, no gather, no transposes" (commit message); teacher-forced KL against the previous build was top-1 116/128, median 0.00208, the same as two runs of one build (116/128, 0.00264; [raw](../../benchmarks/raw/2026-09-26-klprobe-comparisons.csv)).
- 0012-0015 serve `MUL_MAT_ID` batches over 8 tokens (2-8 stay on 0003's per-token path; the limit is `GGML_SYCL_MMID_MULTITOKEN_MAX`) with contiguous f32 `src1` and `dst` at default precision; other shapes fall back to the per-expert loop. The grouped GEMM computes in f16 on the XMX units, a precision trade against the f32 per-expert library GEMM (0012 commit message). The decode path reorders Q4_K/Q5_K experts into a per-expert SoA layout on first use, so 0013 decodes both layouts and reorders on the grouped path too; plain `MUL_MAT` keeps the IQ list. Correctness: "test-backend-ops: 60/60 new MUL_MAT_ID cases pass with and without the reorder" (0013 commit message); teacher-forced KL against the previous build top-1 118/128, median 0.00242, and 117/128, 0.00194 after 0015 ([raw](../../benchmarks/raw/2026-09-26-klprobe-comparisons.csv)).
- Correctness evidence for each patch is in its experiment: [fused MUL_MAT_ID](../../experiments/2026-09-24-fused-mul-mat-id-mtp-verify.md), [sparse FA](../../experiments/2026-09-25-sparse-fa-multi-token.md), [pooled cache](../../experiments/2026-09-25-pooled-qsa-key-cache.md), [draft rows-only and window](../../experiments/2026-09-25-mtp-draft-rows-only-and-window.md), [draft QSA](../../experiments/2026-09-25-mtp-draft-qsa.md), [slot save/restore](../../experiments/2026-09-25-slot-save-restore-and-ram-cache.md), [score expansion by broadcast](../../experiments/2026-09-26-qsa-score-expansion-broadcast.md), [grouped MoE GEMM](../../experiments/2026-09-26-grouped-moe-xmx-gemm.md), [eh_proj](../../experiments/2026-09-26-mtp-eh-proj-2d-product.md).

## License

llama.cpp is MIT-licensed. Our diffs are released under the same MIT license. The two cherry-picks are upstream llama.cpp commits under its MIT license and are not copied here. 0012 is a copy of upstream PR #29245 at `950f1c4aa`, also MIT; it is copied because an open PR's head can be force-pushed and vanish.
