# Pooled QSA indexer-key cache

| | |
|---|---|
| **Date** | 2026-09-25 |
| **Model / checkpoint** | huihui-ai abliterated `UD-Q4_K_XL` + unsloth MTP draft shared-Q8_0 + mmproj BF16 |
| **Host** | [quad-b70-5800x-pex88096](../../../hardware/hosts/quad-b70-5800x-pex88096.md) |
| **Status** | kept |
| **Baseline** | cache off in the same server session (runtime off-switch), sparse FA on in all arms |
| **Result** | [benchmarks/2026-09-25-agent-session-135k.md](../benchmarks/2026-09-25-agent-session-135k.md), [benchmarks/2026-09-25-production-build.md](../benchmarks/2026-09-25-production-build.md) |
| **Related** | [research: long-context QSA design](../research/2026-09-25-long-context-qsa-design.md), [patch 0007](../configs/patches/0007-qwen4exp-pooled-qsa-key-cache.diff), [sparse FA](2026-09-25-sparse-fa-multi-token.md) |

## Hypothesis

The QSA (Qwen sparse attention) indexer scores 4-token blocks. Each block's key is the mean of its 4 raw indexer keys, then RMS-normed and roped at the block's first position. PR #28243 recomputes that pooled key for every block in the cache on every ubatch: about 34K blocks per QSA layer at 135K context, gathered, split into slices, added, scaled, normed and roped. After sparse FA, the pooling slice copies alone were 10.5% of decode op time ([raw/2026-09-25-op-profile-135k.csv](../benchmarks/raw/2026-09-25-op-profile-135k.csv)). The reference implementations (Hugging Face `modeling_qwen4_exp.py`, vLLM v0.30.0, sglang) compute the same thing, and vLLM caches one row per complete block. Caching the rows and pooling only newly completed blocks should save an estimated 17-26 ms per MTP step at 135K.

## Change

[Patch 0007](../configs/patches/0007-qwen4exp-pooled-qsa-key-cache.diff), on by default; `LLAMA_QSA_POOLED=0` turns it off.

- **Storage.** One F32 tensor per QSA layer, `[128, kv_size/4 + 1, n_stream]`, about 96 MiB per GPU. Row `pos/4` holds the pooled key after norm and rope; the spare last row takes padding writes.
- **Watermark.** Per stream, every complete block below the watermark has a correct row. It is lowered in `apply()` (to the first position of the ubatch) and in `seq_rm` (covers MTP rollback). It is reset on clear, `seq_cp`, `seq_keep`, `seq_add`, `seq_div`, full `state_read` and `state_drop`. It is raised in `next()` after a successful ubatch.
- **Recompute.** Dirty complete blocks go through the same op chain as before and are written with `set_rows`; scoring reads a view of the `set_rows` result, so it runs after the write.
- **Fast path scope.** One stream, one sequence, unique positions, per-block bias. Anything else, including image (M-RoPE) positions, takes the old full path.

## Environment

| Component | Version / value |
|---|---|
| Engine and image (tag and digest) | llama.cpp, no container. Build `20260925-f47a6a5f3`: PR #28243 at `6fcaa16` + #28931 + 0002 + 0003 + 0004 + #28796 + 0006 + 0007 |
| Build flags | `GGML_SYCL=ON GGML_SYCL_TARGET=INTEL GGML_SYCL_F16=ON GGML_SYCL_DNN=ON GGML_SYCL_GRAPH=ON GGML_SYCL_SUPPORT_LEVEL_ZERO_API=ON GGML_NATIVE=ON`, Release, icx/icpx from oneAPI 2026.1 |
| compute-runtime / IGC | 26.35.39758.11 / IGC 2.41.9; Level Zero loader 1.32.0 |
| Kernel / `xe` driver / GuC firmware | 7.0.0-34-generic / in-tree `xe` / GuC 70.58.0 |
| oneCCL | not used |
| PyTorch XPU | not used |
| Power cap per card | 230 W |
| Tensor / pipeline parallel | layer split across 4 cards, `-ub 1536 -ts 12,13,13,11 -fit off`, MTP n-max 3 on `SYCL3`, f16 KV, `GGML_SYCL_SPARSE_FA=1` |
| GPU clock floor | 2800 MHz |

## Procedure

1. Adversarial code review of the watermark and invalidation paths (three review lenses plus a verification pass).
2. Model A/B/A in one server session at 135K with [`tools/lcbench.py`](../../../tools/lcbench.py): arm A cache on (cold read plus 6 turns), arm B cache off via the file `/tmp/llama-qsa-pooled-off`, arm C cache on again, which starts with a catch-up of all blocks.
3. Teacher-forced next-token check with [`tools/klprobe.py`](../../../tools/klprobe.py), 128 positions.
4. Freeze the build, deploy, run [`tools/fnbench.py`](../../../tools/fnbench.py) with `LONG_TOKENS=200000`.

## Results

Agent session at 135K ([raw/2026-09-25-agent-session-135k.csv](../benchmarks/raw/2026-09-25-agent-session-135k.csv), runs `lc-poolA`, `lc-pooloff`, `lc-poolC`):

| Metric | Cache off (B) | Cache on (A / C) | Delta |
|---|---|---|---|
| MTP step time, median (256 tokens / tok/s / (draft_n/3)) | 95.8 ms | 72.6 / 73.4 ms | -23% to -24% |
| Single-stream decode, median of 6 turns | 29.9 tok/s | 48.5 / 39.1 tok/s | +31% to +62% |
| Acceptance, turns 1-6 | 60.6% | 82.7% / 62.5% | |
| Per-turn prompt time, ~60 / ~400 new tokens | 1.09-1.12 / 2.65-2.80 s | 1.04-1.32 / 2.59-2.88 s | none |
| VRAM, GPU0 | 28,264 MiB (build without the pool; the runtime switch does not free it) | 28,360 MiB | +96 MiB per GPU |
| Aggregate at c8 / c16 | not applicable (one slot) | | |

The step time is the clean measure. Decode tok/s also moves with draft acceptance, which differs between runs because the generated text differs: arm A accepted 82.7% of drafts and arm C 62.5%, with the same step time. So a realistic figure for this workload is about 39 tok/s, with about 49 when the draft is accepted often.

Deployed build `20260925-f47a6a5f3`, [`tools/fnbench.py`](../../../tools/fnbench.py) ([benchmarks/2026-09-25-production-build.md](../benchmarks/2026-09-25-production-build.md)):

| Metric | `20260924-7e5cb8f13`, `-ub 1536` (dense) | `20260925-60a598ed8` (sparse FA) | `20260925-f47a6a5f3` (sparse FA + pooled cache) |
|---|---|---|---|
| Short decode | 50.3 | 49.3 | 48.6 (46.4-49.9) |
| ~10k decode | 63.6 | 63.1 | 60.6 (acceptance 182/219) |
| ~39k decode | 52.6 | 49.7 | 53.6 |
| 219K decode | 18.4 | not run | 40.6 (acceptance 89/112) |
| Prompt, 10k / 39k | 589.9 / 582.8 | 590.8 / 582.4 | 589.8 / 577.2 |
| Prompt, 219K | 270.0 | not run | 268.3 (816.9 s) |

At 219K the combined sparse FA and pooled cache took decode from 18.4 to 40.6 tok/s (2.2x). Peak VRAM during the 219K run was 28,541 / 28,280 / 28,530 / 28,957 MiB, and host MemAvailable stayed at or above 91.5 GiB.

## Correctness

- Code review: no invalidation or equivalence defects. One minor defect was found and fixed before the build was frozen: the pool size was reported as 0 under `no_alloc` (the `--fit` dry run).
- Teacher-forced KL, 128 positions ([raw/2026-09-25-klprobe.csv](../benchmarks/raw/2026-09-25-klprobe.csv)):

| Comparison | Top-1 agreement | KL mean | KL median |
|---|---|---|---|
| cache on A vs cache on C (noise floor) | 116/128 | 0.088 | 0.00247 |
| cache on A vs cache off | 116/128 | 0.136 | 0.00224 |

- On against off is inside the run-to-run noise. Greedy output is not token-identical run to run on this model even without the change (MTP batching, top-k tie order), so no token diff was used.
- Not changed, and still different from the reference: llama.cpp runs top-k over expanded cells with width 2051. When `(q + 1) % 4 != 3` it can attend up to 3 cells of a 513th block, where the reference attends 512 whole blocks plus the tail. The effect on output has not been measured.
- No image request was tested with the cache on. Image positions take the old path by design.

## Decision

Kept. The MTP step at 135K fell by about 23%, and decode at 219K more than doubled together with sparse FA. It costs 96 MiB of VRAM per GPU. Deployed as build `20260925-f47a6a5f3`, the production build.

## Follow-ups

- [ ] The MTP draft layer is now the largest attention cost: dense over the whole context, 9.4% of decode op time at 135K.
- [ ] Decide whether to fix the 513th-block deviation. It would change output relative to today and match the reference.
- [ ] Prompt turns (~60 tokens, about 1.1 s at 135K) are MoE-bound; see [the MUL_MAT_ID experiment](2026-09-25-mmid-multitoken-prompt-turns.md).
