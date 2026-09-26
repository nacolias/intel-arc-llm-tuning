# Fused MUL_MAT_ID for 2-8-token batches (MTP verify), plus gettid cache and upstream #28931

| | |
|---|---|
| **Date** | 2026-09-24 |
| **Model / checkpoint** | huihui-ai abliterated `UD-Q4_K_XL` + unsloth MTP draft shared-Q8_0 |
| **Host** | [quad-b70-5800x-pex88096](../../../hardware/hosts/quad-b70-5800x-pex88096.md) |
| **Status** | kept |
| **Baseline** | [benchmarks/2026-09-24-baseline-262k-mtp.md](../benchmarks/2026-09-24-baseline-262k-mtp.md) (config), unpatched build |
| **Result** | [raw/2026-09-24-speed-work.csv](../benchmarks/raw/2026-09-24-speed-work.csv) |
| **Related** | [research: decode bottleneck and speed plan](../research/2026-09-24-decode-bottleneck-and-speed-plan.md), patches [0002](../configs/patches/0002-sycl-gettid-cache.diff) and [0003](../configs/patches/0003-sycl-fused-mul-mat-id-2-8-tokens.diff), upstream [#28931](https://github.com/ggml-org/llama.cpp/pull/28931) |

## Hypothesis

MTP (multi-token prediction) verify passes carry 4 tokens, and every one of them takes the slow path of SYCL `MUL_MAT_ID`. With more than one token per call, the backend skips the fused kernel, copies the expert ids to the host, waits on the stream, then launches one matmul per distinct expert (about 39). There are 144 such calls per pass, so a verify costs roughly 11k extra launches and about 66-79 ms, against about 30 ms for a single-token pass. Running each token through the fused single-token path should remove the host waits and take short-context decode from about 37 to an estimated 50-60 tok/s.

The mechanism was read in the source; the size of the gain was an estimate. See the [research note](../research/2026-09-24-decode-bottleneck-and-speed-plan.md).

## Change

Three commits on top of PR #28243 at `6fcaa16`, frozen as build `20260924-7e5cb8f13`:

1. `git cherry-pick 5e48b3100` (upstream #28931: SYCL MMVQ GLU, rms_norm+scale and ssm_conv+silu fusions).
2. [0002-sycl-gettid-cache.diff](../configs/patches/0002-sycl-gettid-cache.diff): cache `gettid()` per thread. perf had put the raw syscall at about 6% of the decode main thread.
3. [0003-sycl-fused-mul-mat-id-2-8-tokens.diff](../configs/patches/0003-sycl-fused-mul-mat-id-2-8-tokens.diff). The core of it:

```diff
+    } else if (g_ggml_sycl_mmid_multitoken && ne12 <= MMVQ_MAX_BATCH_SIZE && ids->ne[1] == ne12 &&
+               ids->nb[0] == sizeof(int32_t) && ne13 == 1 && ids->ne[2] == 1) {
+        // run the fused single-token path on a per-token view of src1, ids and dst
+        for (int64_t t = 0; t < ne12; t++) {
+            ...
+            if (!ggml_sycl_mul_mat_id_mmvq_fused(ctx, src0, &src1_t, &ids_t, &dst_t)) { ... }
+        }
```

`GGML_SYCL_MMID_MULTITOKEN=0` switches the new path off. Launch flags are unchanged from the baseline config (`-ub 1024 -ts 13,13,13,10 -fit off`, MTP n-max 3).

## Environment

| Component | Version / value |
|---|---|
| Engine and image (tag and digest) | llama.cpp, no container. PR #28243 at `6fcaa16` + #28931 + 0002 + 0003 (build `20260924-7e5cb8f13`) |
| Build flags | `GGML_SYCL=ON GGML_SYCL_TARGET=INTEL GGML_SYCL_F16=ON GGML_SYCL_DNN=ON GGML_SYCL_GRAPH=ON GGML_SYCL_SUPPORT_LEVEL_ZERO_API=ON GGML_NATIVE=ON`, Release, icx/icpx from oneAPI 2026.1 |
| compute-runtime / IGC | 26.35.39758.11 / IGC 2.41.9; Level Zero loader 1.32.0 |
| Kernel / `xe` driver / GuC firmware | 7.0.0-34-generic / in-tree `xe` / GuC 70.58.0 |
| oneCCL | not used |
| PyTorch XPU | not used |
| Power cap per card | 230 W |
| Tensor / pipeline parallel | layer split across 4 cards, `-ts 13,13,13,10`; no tensor parallel |
| GPU clock floor | none (driver default) |

## Procedure

1. Build in a separate worktree; never rebuild the directory the service loads.
2. `test-backend-ops -o MUL_MAT_ID -b SYCL0` against the CPU backend.
3. Point the service at the candidate build, restart, and run [`tools/fnbench.py`](../../../tools/fnbench.py) (5 short runs of 512 tokens, then 9,749- and 39,119-token prompts, greedy).
4. Five simple capability checks through the chat endpoint.

## Results

| Metric | Baseline (unpatched) | This change | Delta |
|---|---|---|---|
| Single-stream tok/s, short | about 37 | 42-44 | +14% to +19% |
| Single-stream tok/s, ~10k | about 45 | 52-54 | +16% to +20% |
| Single-stream tok/s, ~39k | about 35 | 44-45 | +26% to +29% |
| Aggregate at c8 / c16 | not applicable (one slot) | | |

- The before values are the session's representative figures. The same config ranged 31-38 tok/s on short runs in that day's service journal, with acceptance from 57% to 71%, so a single run cannot resolve a 5% change.
- A reference run of this build from the [draft-length sweep](2026-09-24-mtp-draft-length-sweep.md), the same day: short median 43.6 tok/s (42.3-45.6), 53.6 at ~10k, 44.0 at ~39k.
- The gain is below the 50-60 tok/s estimate. Before the change, a profile of the unpatched build showed the server's main thread at 96% of one core and each card's compute engine busy only 16-18% ([raw/2026-09-24-decode-profile.csv](../benchmarks/raw/2026-09-24-decode-profile.csv)). Layer split runs the cards one after another, and each token still launches about 4,000 kernels, so the host stays the limit.
- The gettid cache and #28931 were not measured on their own. The research estimates were +2% to +5% and +1% to +2%.

## Correctness

- `test-backend-ops -o MUL_MAT_ID`: 929/929 cases pass (SYCL0 against CPU).
- 5/5 capability checks passed.
- Greedy output is not token-identical between two runs, even on the unmodified build. MTP batching changes the floating-point summation order, and the QSA top-k breaks ties in an unspecified order. So a token-by-token diff cannot be used as the gate for this model. The teacher-forced KL probe ([`tools/klprobe.py`](../../../tools/klprobe.py)) was written the next day and was not run for this change.

## Decision

Kept. It is the largest single decode gain on this model: +14% to +29% depending on context. It is in every later build.

## Follow-ups

- [x] Retry longer drafts now that verify is cheap: [draft-length sweep](2026-09-24-mtp-draft-length-sweep.md).
- [x] Raise the per-token path's batch limit for prompt turns: [GGML_SYCL_MMID_MULTITOKEN_MAX=512](2026-09-25-mmid-multitoken-prompt-turns.md), which was slower.
- [ ] A real multi-token grouped MoE kernel that reads each expert once for all tokens routed to it. That is what prefill and prompt turns need; see [findings/llama-cpp-sycl-moe-prefill-host-sync.md](../../../findings/llama-cpp-sycl-moe-prefill-host-sync.md).
