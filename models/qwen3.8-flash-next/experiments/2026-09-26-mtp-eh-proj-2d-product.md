# MTP `eh_proj` as one 2D matrix product

| | |
|---|---|
| **Date** | 2026-09-26 |
| **Model / checkpoint** | huihui-ai abliterated `UD-Q4_K_XL` + unsloth MTP draft shared-Q8_0 + mmproj BF16 |
| **Host** | [quad-b70-5800x-pex88096](../../../hardware/hosts/quad-b70-5800x-pex88096.md) |
| **Status** | kept (in production build `20260926-2b84213a4` since 2026-09-26) |
| **Baseline** | sandbox build `85b26781b` (patches through [0015](../configs/patches/0015-sycl-grouped-moe-gemm-in-place-io.diff): grouped MoE (mixture-of-experts) GEMM (general matrix multiply) with in-place I/O) |
| **Result** | [raw/2026-09-26-cold-read-windows.csv](../benchmarks/raw/2026-09-26-cold-read-windows.csv) rows `b2pfxuo2i`, `bxftv5kuj`, `begg31572` (baseline) and `bcp5hgsyz` (this change), [raw/2026-09-26-prefill-op-profile.csv](../benchmarks/raw/2026-09-26-prefill-op-profile.csv), [raw/2026-09-26-cold-read-progress.csv](../benchmarks/raw/2026-09-26-cold-read-progress.csv), [raw/2026-09-26-klprobe-comparisons.csv](../benchmarks/raw/2026-09-26-klprobe-comparisons.csv), [raw/2026-09-26-lcbench-sessions-summary.csv](../benchmarks/raw/2026-09-26-lcbench-sessions-summary.csv) rows `acc-A` and `acc-B` |
| **Related** | [patch 0016](../configs/patches/0016-qwen4exp-mtp-eh-proj-2d-product.diff), [benchmark: prefill op profile](../benchmarks/2026-09-26-prefill-op-profile.md), [finding: SYCL MUL_MAT loops per slice](../../../findings/llama-cpp-sycl-batched-mul-mat-loops-per-slice.md), [`-b 3072` test on this build](2026-09-26-batch-3072.md) |

## Result

A cold 135K read went from about 313 s to about 284 s (-9%) by reshaping one tensor. The baseline build's three reads spanned 295.7-314.1 s and the new build was read once at the default flags, so the gain is somewhere between -4% and -10%. Short prompt turns and decode did not change. The target model's next-token probabilities stayed within run-to-run noise; that probe does not run the MTP draft, which this patch changes, so it does not show the draft is unchanged.

## Hypothesis

The MTP (multi-token prediction) draft head projects the concatenated embedding and hidden state through `eh_proj`, a 2D `[2*n_embd, n_embd]` weight (Q8_0, about 14 MB; "commit message" of patch 0016). Its input `concat` is 3D, `[2*n_embd, hc, n_tokens]`, because this model's residual stream has a hyper-connection (`hc`) axis. In the prefill op profile, `MUL_MAT mtp_eh_proj` took 115 ms per 1536-token micro-batch (ubatch) on two builds, 4.7% and 8.4% of listed op time ([raw/2026-09-26-prefill-op-profile.csv](../benchmarks/raw/2026-09-26-prefill-op-profile.csv)); the commit message puts it at "8.5% of a cold read at ~100K". A `[5120 x 2560]` product over 1536 tokens should take a few milliseconds, not 115.

The cause, from the commit message: "a 3D src1 makes the SYCL backend run one hc-column product per token, each reading the whole 14 MB Q8_0 weight". 1,536 slices times 13.9 MB is about 21 GB of weight reads per ubatch, which at 115 ms is about 186 GB/s (estimate, derived from the commit message's weight size and the profiled time). Since the weight is 2D, the same product can be written as one 2D product over `hc * n_tokens` columns. Expected: the 115 ms per ubatch disappears, so a 135K cold read gets about 8-10 s faster (estimate); nothing else changes.

## Change

[patch 0016](../configs/patches/0016-qwen4exp-mtp-eh-proj-2d-product.diff), `src/models/qwen4exp.cpp`, in the MTP graph:

```diff
-    ggml_tensor * res_hc = build_lora_mm(layer.nextn.eh_proj, concat, layer.nextn.eh_proj_s);
+    // one [2*n_embd, hc*n_tokens] product: with a 3D src1 the SYCL backend runs one hc-column
+    // product per token, each reading the whole weight
+    ggml_tensor * res_hc = build_lora_mm(layer.nextn.eh_proj,
+            ggml_reshape_2d(ctx0, concat, 2 * n_embd, hc * n_tokens), layer.nextn.eh_proj_s);
+    res_hc = ggml_reshape_3d(ctx0, res_hc, n_embd, hc, n_tokens);
```

No flags or environment variables change. The A/B session below used a temporary `LLAMA_MTP_EH3D=1` switch in the sandbox build to select the old 3D path; the committed patch has no such switch (source: session log, not archived).

## Environment

| Component | Version / value |
|---|---|
| Engine and image (tag and digest) | llama.cpp, no container. Sandbox build `a890bf8b0`: PR #28243 at `6fcaa16` + #28931 + 0002-0004 + #28796 + 0006-0016 ([configs/patches/](../configs/patches/)). Frozen as production build `20260926-2b84213a4` together with the diagnostic-only [patch 0017](../configs/patches/0017-sycl-sparse-fa-union-stats-debug.diff) |
| Build flags | `GGML_SYCL=ON GGML_SYCL_TARGET=INTEL GGML_SYCL_F16=ON GGML_SYCL_DNN=ON GGML_SYCL_GRAPH=ON GGML_SYCL_SUPPORT_LEVEL_ZERO_API=ON GGML_NATIVE=ON`, Release, icx/icpx from oneAPI 2026.1 |
| compute-runtime / IGC | 26.35.39758.11 / IGC 2.41.9; Level Zero loader 1.32.0 |
| Kernel / `xe` driver / GuC firmware | 7.0.0-34-generic / in-tree `xe` / GuC 70.58.0 |
| oneCCL | not used |
| PyTorch XPU | not used |
| Power cap per card | 230 W |
| Tensor / pipeline parallel | layer split across 4 cards, `-ub 1536 -ts 12,13,13,11 -fit off`, MTP n-max 3 on `SYCL3`, `GGML_SYCL_SPARSE_FA=1`, pooled QSA (Qwen sparse attention) key cache on, draft QSA on |
| GPU clock floor | 2800 MHz |

## Procedure

1. [`tools/lcbench.py`](../../../tools/lcbench.py) at 135K (`LC_TOKENS=135000 LC_TURNS=10 LC_PREDICT=256`): one cold read of the fixed 134,862-token C++ context, then 10 greedy turns of 256 tokens. Same command on the baseline build with 6 turns.
2. [`tools/klprobe.py`](../../../tools/klprobe.py): teacher-forced top-20 log-probabilities at 128 positions after the same 134,840-token prefix, compared with the baseline build's probe (`kl-eh` against `kl-mmid18`). The probe sends one-token requests (`n_predict` 1), so no MTP draft step and no multi-token verify batch runs in it.
3. [`tools/accprobe.py`](../../../tools/accprobe.py): two sessions of the same sandbox build on a 41,419-token context, 11 greedy turns of 192 tokens each (`ACC_TURNS=12 ACC_PREDICT=192`), session A with the 2D product and session B with the old 3D path (`LLAMA_MTP_EH3D=1`, plus `GGML_SYCL_SPARSE_FA_DEBUG=2` for the [selection-union statistics](../research/2026-09-26-sparse-prompt-attention-tile-union.md)). Summaries with [`tools/lcsum.py`](../../../tools/lcsum.py).

## Results

| Metric | Baseline `85b26781b` | This change (`a890bf8b0`) | Delta | Source |
|---|---|---|---|---|
| `MUL_MAT mtp_eh_proj`, ms per 1536-token ubatch | 115.08 (22 calls; 115.21 on `993baf141`, 10 calls) | not profiled; the session estimated about 2 ms (UNVERIFIED) | | [raw/2026-09-26-prefill-op-profile.csv](../benchmarks/raw/2026-09-26-prefill-op-profile.csv); session log |
| Cold read of 134,862 tokens, client-side (lcbench turn 0) | 312.71 / 314.11 / 295.68 s (3 reads; the 295.68 s read had the op profiler on for 62 s of it) | 283.66 s; 286.44 s with `-b 3072` | -29.05 s (-9.3%) against the median baseline read; -12.0 s (-4.1%) against the fastest | [raw/2026-09-26-cold-read-windows.csv](../benchmarks/raw/2026-09-26-cold-read-windows.csv) rows `b2pfxuo2i`, `bxftv5kuj`, `begg31572`, `bcp5hgsyz`, `bsw2jblow` |
| Cold read, server-side progress line at the last token | 311.98 / 313.34 / 294.94 s (runs 01:04:07, 01:17:44, 01:26:45 UTC) | 283.10 s (run 01:36:43); 285.88 s with `-b 3072` (run 02:02:06) | -29 s | [raw/2026-09-26-cold-read-progress.csv](../benchmarks/raw/2026-09-26-cold-read-progress.csv); build per run from the windows CSV; the "commit message" says "135K cold read 313 -> 284 s" |
| Prompt turn, median (~60 and ~400 new tokens) | 0.77 s (6 turns) | 0.76 s (10 turns) | none | [windows CSV](../benchmarks/raw/2026-09-26-cold-read-windows.csv) |
| Decode median, turns 1-N | 48.7 tok/s (36.3-51.5, 6 turns) | 43.7 tok/s (37.6-50.2, 10 turns) | within acceptance noise | [windows CSV](../benchmarks/raw/2026-09-26-cold-read-windows.csv) |
| MTP step time (from lcsum) | 68.1 ms | 68.0 ms | none | [windows CSV](../benchmarks/raw/2026-09-26-cold-read-windows.csv) |
| Acceptance, turns 1-N | 70.7% | 63.8% | different generated text | [windows CSV](../benchmarks/raw/2026-09-26-cold-read-windows.csv) |
| Peak VRAM per card | 28,748 / 28,486 / 28,752 / 29,256 MiB | 28,752 / 28,490 / 28,756 / 29,278 MiB | none | session log, not archived |
| Aggregate at c8 / c16 | not applicable (one slot) | | | |

The step time is the stable decode measure; tok/s moves with acceptance, which changes with the generated text ([agent-session benchmark](../benchmarks/2026-09-25-agent-session-135k.md)).

Against the typical baseline read, the saving is about three times the profiled `eh_proj` time. With `-b 2048 -ub 1536`, a 135K read runs 66 ubatches of 1536 tokens and 66 of 512, so 115 ms per full ubatch sums to about 10 s per read (estimate), yet the read got 29 s faster. The profiler waits after every op, so the 3D path's cost in normal execution (1,536 dependent small products with their host-side setup) was evidently larger than its serialized op time. This was not investigated.

The production README's "0.90 to 0.76 s" for short prompt turns spans two changes: patch 0015 took them from 0.90 to 0.77 s, and this change left them at 0.76 s. This patch can explain at most a few milliseconds of a turn: 115 ms per 1,536-token ubatch is about 0.075 ms per token, about 4.5 ms for a 60-token turn at the profiled rate (derived), against the 0.13 s drop from 0.90 to 0.77 s.

### Acceptance in two sessions at 41K (not a measured result)

73.5% against 66.3% is not a measured result: two sessions, different generated text (0 of 12 identical outputs), a 41K context rather than 135K, and a debug setting (`GGML_SYCL_SPARSE_FA_DEBUG=2`) only in session B. Two `accprobe.py` sessions of the same build, 11 turns each, on the same 41,419-token context ([raw/2026-09-26-lcbench-sessions-summary.csv](../benchmarks/raw/2026-09-26-lcbench-sessions-summary.csv), per turn in [raw/2026-09-26-lcbench-sessions-per-turn.csv](../benchmarks/raw/2026-09-26-lcbench-sessions-per-turn.csv)):

| Session | Path | Cold read of 41,419 tokens | Acceptance, turns 1-11 | Per-turn acceptance range | Step | Decode overall | Prompt median |
|---|---|---|---|---|---|---|---|
| `acc-A` | 2D product (this change) | 60.41 s | 73.5% (1440/1958) | 63.8-88.5% | 65.1 ms | 49.7 tok/s | 0.66 s |
| `acc-B` | old 3D path (`LLAMA_MTP_EH3D=1`), plus `GGML_SYCL_SPARSE_FA_DEBUG=2` | 62.07 s | 66.3% (1395/2103) | 60.6-72.8% | 64.0 ms | 47.1 tok/s | 0.65 s |

The two sessions produced identical turn outputs in 0 of 12 turns (source: session log, not archived), so they ran on different generated text and their acceptance does not compare like for like. The 7-point difference is within the per-turn spread inside each session (63.8-88.5% and 60.6-72.8%) and within the session-to-session spread seen on unchanged builds (for example 82.7% against 62.5% for `lc-poolA` and `lc-poolC` in one server session, [agent-session benchmark](../benchmarks/2026-09-25-agent-session-135k.md)). Step time, which does not depend on the text, differed by 1.1 ms. Read the acceptance difference as neither a gain nor a regression: it is not resolved. Mathematically the 2D product is the same product; only the floating-point summation order can differ.

The prompt turns of session A were 0.19-0.28 s slower than session B's on every long (~400-470-token) turn (1.20-1.33 s against 0.92-1.08 s), the opposite of what the change would predict. One session each; the cause was not investigated.

## Correctness

- Teacher-forced next-token check ([raw/2026-09-26-klprobe-comparisons.csv](../benchmarks/raw/2026-09-26-klprobe-comparisons.csv)): `kl-eh` against `kl-mmid18` (the baseline build) agrees on the top-1 token at 115/128 positions, KL (Kullback-Leibler divergence) median 0.00245, mean 0.179, p95 0.545, max 5.59. Same-build noise floors in the same file: `kl-poolC` against `kl-poolA` 116/128, median 0.00264; `kl-spC` against `kl-spA` 115/128, median 0.00207. (`kl-mmid18` against `kl-mmid`, 118/128, median 0.00195, compares two builds, `85b26781b` and `517eb833d`, not two runs of one.) The target model's next-token distribution is inside that noise. The probe sends one-token requests, so the MTP draft that this patch changes never proposes a token and no multi-token verify batch runs: the probe does not show the draft is unchanged. That rests on the construction (the same product; only the floating-point summation order can differ), not on a measurement.
- The shared greedy token-by-token check of [benchmarks/README.md](../../../benchmarks/README.md) was not run in that form. Greedy output on this model is not token-identical between runs even on one build (MTP batching changes summation order; the QSA top-k breaks ties in an unspecified order), so the KL probe is the check used throughout this model's experiments.
- No NaN, repetition or garbled output in the lcbench turns of this build (source: session log, not archived). The repetition seen in the later `-b 3072` session is discussed [there](2026-09-26-batch-3072.md).

## Decision

Kept. A one-line reshape removes 12-30 s (4-10%) from a 135K cold read, 29 s against the typical baseline read, with no measurable change to decode, prompt turns or the target model's next-token probabilities. Frozen as production build `20260926-2b84213a4` on 2026-09-26.

## Follow-ups

- [ ] Profile a cold read on the fixed build to confirm `mtp_eh_proj` is gone from the top of the list and to explain why the saving (29 s) is about three times the profiled op time (about 10 s per read).
- [ ] Pin the size of the gain with two or three more reads of each build at the default flags: the baseline's reads spanned 295.7-314.1 s, and the new build has one read at the default flags.
- [ ] Check the other `qwen4exp` products with a hyper-connection axis (`hc_gate`, `hc_inject`: 0.14-0.17 ms per call in the profile) for the same pattern; they look cheap enough to leave alone.
- [ ] Upstream: report the per-slice `MUL_MAT` behaviour of the SYCL backend, or make the backend flatten a 3D src1 against a 2D weight itself. See the [finding](../../../findings/llama-cpp-sycl-batched-mul-mat-loops-per-slice.md).
