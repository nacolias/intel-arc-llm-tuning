# llama.cpp patches for Qwen3.8-Flash-Next on SYCL

The production build is upstream PR #28243 plus two upstream cherry-picks and five of our own diffs. Our diffs are in this folder. They apply cleanly in the order below; we checked this on 2026-09-25 by applying them to the pre-image files and comparing the result with the frozen build's tree (identical except for two reworded code comments). Afterwards, three assignments in 0003 and 0004 were wrapped onto two lines so the repository's secret scanner does not mistake them for keys; the code is unchanged, and the wrapped diffs were re-checked to apply in order.

Build tag on our host: `20260925-f47a6a5f3` (the commit hash of the last patch in our tree).

## Apply order

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
```

Commit after each diff: #28796 touches `ggml-sycl.cpp`, which 0003 and 0004 also change, and `git cherry-pick` refuses to run over uncommitted changes to the same file. Both cherry-picked commits are on upstream `master`, so a plain clone can reach them.

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

## Scope and limits

- 0007 takes its fast path only for one stream with one sequence, unique positions and the per-block bias. Anything else, including image (M-RoPE) positions from the vision projector, falls back to the full re-pooling path.
- 0007 keeps llama.cpp's top-k over expanded cells (width 2051). That attends up to 3 cells of a 513th block at 3 of every 4 positions, where the reference attends 512 whole blocks plus the tail. We did not change this. See [research/2026-09-25-long-context-qsa-design.md](../../research/2026-09-25-long-context-qsa-design.md).
- 0006 engages only while `T * n_kv_g * 2 <= n_kv`, where `T` is the number of query rows and `n_kv_g = pad(n_kv_max + 256, 256)` (2560 for this model). Prompt turns of about 60 tokens are above the 32-row limit and still use dense attention.
- Correctness evidence for each patch is in its experiment: [fused MUL_MAT_ID](../../experiments/2026-09-24-fused-mul-mat-id-mtp-verify.md), [sparse FA](../../experiments/2026-09-25-sparse-fa-multi-token.md), [pooled cache](../../experiments/2026-09-25-pooled-qsa-key-cache.md).

## License

llama.cpp is MIT-licensed. Our diffs are released under the same MIT license. The two cherry-picks are upstream llama.cpp commits under its MIT license and are not copied here.
