# Experiments log

One file per change attempt, named `YYYY-MM-DD-<slug>.md`. Start from [`templates/experiment.md`](../../../templates/experiment.md).

All runs are on [quad-b70-5800x-pex88096](../../../hardware/hosts/quad-b70-5800x-pex88096.md): 4x Arc Pro B70, llama.cpp SYCL, layer split, 262,144 context, MTP (multi-token prediction) draft, single slot. Decode is single-stream tok/s. "Short / 10k / 39k" are the [`tools/fnbench.py`](../../../tools/fnbench.py) points (19-, 9,749- and 39,119-token prompts); "135K" is the agent-session median from [`tools/lcbench.py`](../../../tools/lcbench.py). A "cold read" is the first read of lcbench's 134,862-token context by a freshly started server.

Abbreviations in the tables: PLE per-layer n-gram embedding table, FA flash attention, KL Kullback-Leibler divergence, QSA Qwen sparse attention, KV key/value, MoE mixture of experts, XMX Xe Matrix Extensions, GEMM general matrix multiply.

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
| 2026-09-25 | [MTP draft: output-rows-only attention and a sliding window](2026-09-25-mtp-draft-rows-only-and-window.md) | patch 0008; `LLAMA_MTP_WINDOW=2048` / `8192`; `LLAMA_HOST_PROF` | rows-only: 135K MTP step 73.4 to 72.7 ms, exact; window: step -4.2 to -5.0 ms but acceptance -4.1 to -8.0 points, decode 41.9-43.1 vs 42.3 tok/s (a wash); N=1 session per arm | rows-only kept; window reverted |
| 2026-09-25 | [QSA in the MTP draft head](2026-09-25-mtp-draft-qsa.md) | patch 0009; `LLAMA_MTP_QSA=0`, `/tmp/llama-mtp-qsa-off` | paired A/B at 135K: acceptance 67.6% vs 65.7%, step 70.3 vs 71.5 ms, decode 43.4 vs 41.9 tok/s (+3.6%); ~60-token prompt turns 1.24-1.27 to 0.92 s (checkpoints stop copying the draft KV); +96 MiB on `SYCL3` | kept |
| 2026-09-25 | [Slot save/restore across restarts + RAM prompt cache](2026-09-25-slot-save-restore-and-ram-cache.md) | patch 0010 (draft `.dft` save), [`llama-server-slot-cache.sh`](../configs/llama-server-slot-cache.sh) with a setup signature, `--slot-save-path`, `--cache-ram 24576` | a 135,760-token conversation survives a restart: save 1.5 s (3.7 GB + 303 MB draft), restart 88 s, first turn 3.92 s then 0.78-0.80 s, instead of a 403-472 s cold re-read; 41K saves in 0.4 s; re-signed restore 0.89 s | kept |
| 2026-09-26 | [QSA score expansion by broadcast](2026-09-26-qsa-score-expansion-broadcast.md) | patch 0011: broadcast add of block scores onto the mask instead of `get_rows` plus two `cont(permute)` when cells sit in position order; the unused `cell_blk` input gets no buffer | cold 135K read 455 to 382 s (-16%; N=1 against a pooled baseline of 6 runs, 436.8-462.0 s); decode and ~60-token prompt turns unchanged; KL within noise (116/128, median 0.00208 vs 116/128, 0.00264); the first build crashed on the unused input, fixed in the patch | kept |
| 2026-09-26 | [Grouped MoE XMX GEMM for prompt batches](2026-09-26-grouped-moe-xmx-gemm.md) | patches 0012-0015: upstream #29245 (IQ types only) + Q4_K/Q5_K/Q5_1/Q8_0 decoders in both expert layouts, vector loads, in-place `src1`/`dst` | one MoE call per 1536-token ubatch (micro-batch) 11.3 to 3.9 ms (Q4_K), 11.9 to 4.4 (Q5_K), 11.0 to 3.8 (Q5_1), 10.9 to 7.4 (Q8_0); cold 135K read 382 to about 313 s (-18%); ~60-token prompt turns 0.90 to 0.77 s; `test-backend-ops` 60/60 reordered and plain; KL inside run-to-run noise | kept |
| 2026-09-26 | [MTP `eh_proj` as one 2D product](2026-09-26-mtp-eh-proj-2d-product.md) | patch 0016: reshape the 3D `concat` to 2D before the `eh_proj` product and back after it | cold 135K read about 313 to 283.66 s (-9% against the median baseline read, -4% against the fastest; baseline reads 295.7-314.1 s); prompt turns 0.77 to 0.76 s and MTP step 68 ms unchanged; KL within run-to-run noise | kept |
| 2026-09-26 | [`-b 3072`](2026-09-26-batch-3072.md) | `-b 3072 -ub 1536`: no 512-token remainder ubatches | cold 135K read 286.44 vs 283.66 s: no gain; that session's decode and acceptance (62.4 tok/s, 93.1%) are not valid, because the model repeated the same answer in every turn | reverted |

## Progress

| Stage | Short | ~10k | ~39k | 219K | 135K agent session |
|---|---|---|---|---|---|
| Baseline, unpatched (2026-09-24) | 33.8-36.9 | 43.3 | 33.4 | 16.1 | not measured |
| + fused MUL_MAT_ID, gettid cache, #28931 | 42-44 | 52-54 | 44-45 | not measured | not measured |
| + GPU clock floor | 50 | 64-66 | 47-49 | not measured | not measured |
| + `-ub 1536 -ts 12,13,13,11` | 50.3 | 63.6 | 52.6 | 18.4 | 22.3-23.3 |
| + sparse FA for multi-token batches | 49.3 | 63.1 | 49.7 | not measured | 31.3-34.5 |
| + pooled QSA key cache (`20260925-f47a6a5f3`) | 48.6-50.5 | 60.6-67.2 | 53.6-55.2 | 40.6 | 39.1-48.5 |
| + draft rows-only attention, draft QSA (`20260925-0f23f62c8`) | 46.3 (one run of 3 repetitions) | 59.4 | 54.9 | not measured | 41.1-43.4 (paired: 43.4 draft QSA vs 41.9 dense draft) |
| + QSA score expansion by broadcast (`20260926-2c88bdf19`) | not measured | not measured | not measured | not measured | 43.9 (6 turns) |
| + grouped MoE XMX GEMM (dev trees `517eb833d`, `85b26781b`) | not measured | not measured | not measured | not measured | 48.3-48.7 (6 turns each) |
| + MTP `eh_proj` as one 2D product (`a890bf8b0`, frozen as production `20260926-2b84213a4`) | not measured | not measured | not measured | not measured | 43.7 (10 turns) |

The pooled-cache row spans three fnbench runs of build `20260925-f47a6a5f3` on 2026-09-25; see [benchmarks/2026-09-25-production-build.md](../benchmarks/2026-09-25-production-build.md). Rows after the clock floor do not improve short-context decode. The remaining limit there is most likely the host launch rate under layer split, about 4,000 kernel launches per token plus the blocking hops between cards; see [research: decode bottleneck](../research/2026-09-24-decode-bottleneck-and-speed-plan.md).

The draft-QSA row is one fnbench run ([raw/2026-09-25-draft-qsa-fnbench.csv](../benchmarks/raw/2026-09-25-draft-qsa-fnbench.csv)); its short value below the previous row is UNVERIFIED as a real cost. Its 135K cell is the `lc-draftqsa` median and the paired A/B ([raw/2026-09-25-pairprobe-draft-qsa.csv](../benchmarks/raw/2026-09-25-pairprobe-draft-qsa.csv)).

The 2026-09-26 rows change prefill, not decode. Their 135K medians move with draft acceptance (63.8-72.2%) while the MTP step stays at 67.1-68.4 ms. What they changed is the cold read: 455 s (median of 6 runs before patch 0011) to 382 s (0011), about 313 s (0012-0015) and 284 s (0016), one run per build after the baseline; see [benchmarks/2026-09-26-cold-read-135k-by-build.md](../benchmarks/2026-09-26-cold-read-135k-by-build.md). fnbench was not re-run after `0f23f62c8`.
