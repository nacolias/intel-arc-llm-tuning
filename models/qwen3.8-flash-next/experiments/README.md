# Experiments log

One file per change attempt, named `YYYY-MM-DD-<slug>.md`. Start from [`templates/experiment.md`](../../../templates/experiment.md).

All runs are on [quad-b70-5800x-pex88096](../../../hardware/hosts/quad-b70-5800x-pex88096.md): 4x Arc Pro B70, llama.cpp SYCL, layer split, 262,144 context, MTP draft, single slot. Decode is single-stream tok/s. "Short / 10k / 39k" are the [`tools/fnbench.py`](../../../tools/fnbench.py) points (19-, 9,749- and 39,119-token prompts); "135K" is the agent-session median from [`tools/lcbench.py`](../../../tools/lcbench.py).

## Log

| Date | Experiment | Change | Result | Status |
|---|---|---|---|---|
| 2026-09-24 | [Baseline: 262K + MTP](2026-09-24-baseline-llamacpp-mtp-262k.md) | `-ub 1024 -ts 13,13,13,10 -fit off`, MTP n-max 3 on the last card | fits 262K with MTP; 33.8-36.9 / 43.3 / 33.4 tok/s, 16.1 at 219K; `-ub 2048` with MTP thrashed at 0.6 tok/s | kept |
| 2026-09-24 | [Fused MUL_MAT_ID for MTP verify](2026-09-24-fused-mul-mat-id-mtp-verify.md) | patches #28931 + 0002 + 0003 | short 37 to 42-44, 10k 45 to 52-54, 39k 35 to 44-45; test-backend-ops 929/929 | kept |
| 2026-09-24 | [SYCL graphs](2026-09-24-sycl-graphs-multi-gpu.md) | `GGML_SYCL_ENABLE_GRAPH=1` | 36.47 tok/s with it, the same level as without; graphs are refused with more than one device | reverted |
| 2026-09-24 | [MTP draft length sweep](2026-09-24-mtp-draft-length-sweep.md) | n-max 4; n-max 6 + p-min 0.75 | no consistent gain; n-max 6 lost 19% at 39k | reverted |
| 2026-09-24 | [GPU clock floor](2026-09-24-gpu-clock-floor.md) | `min_freq = rp0` (2800 MHz) while serving | short 42-44 to 50, 10k 52-54 to 64-66, 39k 44-45 to 47-49; about 0 W at true idle | kept |
| 2026-09-25 | [Pipeline-parallel prefill](2026-09-25-pipeline-parallel-prefill.md) | drop the PLE `-ot`, `-b 4096` | prefill 552 / 556 tok/s, no gain; +2.6 to +2.9 GiB VRAM per card, +11 GiB host RAM | reverted |
| 2026-09-25 | [`-ub 1536`](2026-09-25-ubatch-1536.md) | `-ub 1536 -ts 12,13,13,11` | prefill +4.5% at 39k, +6% at 219K; fullest card 28.2 GiB | kept |
| 2026-09-25 | [Per-token MUL_MAT_ID up to 512 tokens](2026-09-25-mmid-multitoken-prompt-turns.md) | `GGML_SYCL_MMID_MULTITOKEN_MAX=512` | ~60-token turns unchanged, ~400-token turns 16-33% slower | reverted |
| 2026-09-25 | [Sparse FA for multi-token batches](2026-09-25-sparse-fa-multi-token.md) | #28796 + patch 0006, `GGML_SYCL_SPARSE_FA=1` | 135K decode 22.2 to 31.3-34.5 tok/s; KL inside run-to-run noise | kept |
| 2026-09-25 | [Pooled QSA key cache](2026-09-25-pooled-qsa-key-cache.md) | patch 0007 | 135K MTP step 95.8 to 72.6-73.4 ms (39.1-48.5 tok/s); 219K decode 18.4 to 40.6 with sparse FA | kept |

## Progress

| Stage | Short | ~10k | ~39k | 219K | 135K agent session |
|---|---|---|---|---|---|
| Baseline, unpatched (2026-09-24) | 33.8-36.9 | 43.3 | 33.4 | 16.1 | not measured |
| + fused MUL_MAT_ID, gettid cache, #28931 | 42-44 | 52-54 | 44-45 | not measured | not measured |
| + GPU clock floor | 50 | 64-66 | 47-49 | not measured | not measured |
| + `-ub 1536 -ts 12,13,13,11` | 50.3 | 63.6 | 52.6 | 18.4 | 22.3-23.3 |
| + sparse FA for multi-token batches | 49.3 | 63.1 | 49.7 | not measured | 31.3-34.5 |
| + pooled QSA key cache (production) | 48.6-50.5 | 60.6-67.2 | 53.6-55.2 | 40.6 | 39.1-48.5 |

The production row spans three fnbench runs of build `20260925-f47a6a5f3` on 2026-09-25; see [benchmarks/2026-09-25-production-build.md](../benchmarks/2026-09-25-production-build.md). Later rows do not improve short-context decode. The remaining limit there is most likely the host launch rate under layer split, about 4,000 kernel launches per token plus the blocking hops between cards; see [research: decode bottleneck](../research/2026-09-24-decode-bottleneck-and-speed-plan.md).
