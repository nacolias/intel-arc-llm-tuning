# llama.cpp (SYCL backend)

Tested on Arc Pro B70 since 2026-09-24, on host [quad-b70-5800x-pex88096](../../hardware/hosts/quad-b70-5800x-pex88096.md). It is the primary engine for [Qwen3.8-Flash-Next](../../models/qwen3.8-flash-next/), because upstream vLLM refuses that model on XPU ([deep dive](../../models/qwen3.8-flash-next/research/2026-09-25-vllm-xpu-deep-dive.md)).

## Build

| Item | Value |
|---|---|
| Tree | [ggml-org/llama.cpp PR #28243](https://github.com/ggml-org/llama.cpp/pull/28243) (qwen4exp + MTP, open) at `6fcaa16`, plus patches 0001-0023: two upstream cherry-picks, one copied upstream PR (#29245) and twenty of our diffs ([list and apply order](../../models/qwen3.8-flash-next/configs/patches/README.md)) |
| Frozen production build | `20260926-5c258f538` (through 0023) since 2026-09-26 22:10 UTC. Earlier: `20260926-2b84213a4` (through 0017, earlier on 2026-09-26), `20260925-f47a6a5f3` (patches through 0007, the build behind the fnbench numbers below), `20260925-993baf141` (through 0010), `20260926-2c88bdf19` (through 0011) |
| Compiler | Intel oneAPI 2026.1 (`icx` / `icpx` 2026.1.1) |
| Runtime on the host | compute-runtime (NEO) 26.35, Level Zero 1.32.0, kernel 7.0.0-34 with `xe`, GuC 70.58.0 |

CMake options read from our build's `CMakeCache.txt`:

```bash
source /opt/intel/oneapi/2026.1/oneapi-vars.sh
cmake -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
  -DGGML_SYCL=ON -DGGML_SYCL_TARGET=INTEL -DGGML_SYCL_F16=ON \
  -DGGML_SYCL_DNN=ON -DGGML_SYCL_GRAPH=ON -DGGML_NATIVE=ON
cmake --build build -j --target llama-server llama-bench test-backend-ops
```

`GGML_SYCL_SUPPORT_LEVEL_ZERO_API` and `GGML_SYCL_HOST_MEM_FALLBACK` were also `ON` in the cache. Run with `ZES_ENABLE_SYSMAN=1` and `UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1`. The full launch script is in [`configs/llama-server-262k-mtp.sh`](../../models/qwen3.8-flash-next/configs/llama-server-262k-mtp.sh).

## Multi-GPU split modes

Only layer split works for qwen4exp on four cards.

| Mode | Status on 4x B70 | Source |
|---|---|---|
| `-sm layer` | Works. The cards run one after another, so each card's compute engine is busy well under half the time (15-28% measured, by card and workload: 15-18% in decode, 17-28% in prefill) and speed is bounded by the sum of per-card time | [decode analysis](../../models/qwen3.8-flash-next/research/2026-09-24-decode-bottleneck-and-speed-plan.md), [prefill note](../../models/qwen3.8-flash-next/research/2026-09-25-prefill-bottleneck.md) |
| `-sm row` | Cannot split MoE (mixture-of-experts) tensors | `ggml-sycl.cpp:5143` at `6fcaa16` |
| `-sm tensor` | qwen4exp is excluded (`llama-arch.cpp:1167`; [#28569](https://github.com/ggml-org/llama.cpp/pull/28569) open). The SYCL all-reduce supports only 2 devices (`ggml-sycl.cpp:6926-6974`). [#26409](https://github.com/ggml-org/llama.cpp/issues/26409) reports it 3x slower than one GPU on 2x B70 (UNVERIFIED here) | source |
| Pipeline parallelism | Disabled by any `-ot` (`src/llama-context.cpp:428-433`). When enabled it gained 0% on prefill, because the SYCL backend has no asynchronous cross-device copy (`cpy_tensor_async = NULL`) | [experiment](../../models/qwen3.8-flash-next/experiments/2026-09-25-pipeline-parallel-prefill.md) |

`-ot per_layer_token_embd=CPU` is redundant for qwen4exp: that tensor is lazily read and placed in CPU memory before overrides are checked (`llama-model-loader.cpp` about line 1211). Leaving it in only disables pipeline parallelism.

## SYCL graphs

SYCL graphs never engage with more than one GPU. `check_graph_compatibility` returns false when `device_count > 1` and for any MUL_MAT_ID. `GGML_SYCL_ENABLE_GRAPH=1` measured no effect: 36.47 tok/s with it set, the same level as the same day's runs without it ([experiment](../../models/qwen3.8-flash-next/experiments/2026-09-24-sycl-graphs-multi-gpu.md)). See [the finding](../../findings/llama-cpp-sycl-graphs-off-with-multiple-gpus.md).

## MoE matmul (MUL_MAT_ID) paths

| Tokens in the batch | Path | Host sync |
|---|---|---|
| 1 | fused MMVQ (quantized matrix-vector) kernel for Q4_K, Q5_K, Q5_1 and Q8_0 experts | none |
| 2-8, stock | same slow path as more than 8 | yes |
| 2-8, with our patch 0003 | fused MMVQ once per token (`GGML_SYCL_MMID_MULTITOKEN=1`, the default in the patch) | none |
| more than 8, stock (or `GGML_SYCL_XMX_GATHER_TYPES=0`) | copies expert ids to the host and waits, then runs one oneDNN GEMM per expert. On Flash-Next: 144 host syncs and about 73.7K GEMMs per micro-batch | yes, per expert |
| more than 8, with patches 0012-0015 | grouped XMX (Xe Matrix Extensions) GEMM: one launch per `MUL_MAT_ID` node covers every expert, the weights are dequantized inside the XMX tiles, and the routed rows are read from `src1` and written to `dst` in place. Upstream #29245 (copied as 0012) handles IQ formats; 0013 adds this model's Q4_K, Q5_K, Q5_1 and Q8_0 experts. Computes in f16 on XMX (a precision trade against the f32 per-expert GEMM; teacher-forced KL against the loop build was within run-to-run noise, [raw](../../models/qwen3.8-flash-next/benchmarks/raw/2026-09-26-klprobe-comparisons.csv)) | one per call: the expert ids go to the host for the tile schedule |

One expert matmul over a 1536-token ubatch (`test-backend-ops perf`, 512 experts, 10 used, one B70; [raw](../../models/qwen3.8-flash-next/benchmarks/raw/2026-09-26-mul-mat-id-kernel-perf.csv), [experiment](../../models/qwen3.8-flash-next/experiments/2026-09-26-grouped-moe-xmx-gemm.md)):

| Expert type | Per-expert loop | 0013 grouped | 0014 vector loads | 0015 in-place I/O |
|---|---|---|---|---|
| q4_K (gate/up) | 11.31 ms | 5.56 ms | 4.86 ms | 3.93 ms |
| q5_K (gate/up) | 11.94 ms | 7.81 ms | 5.31 ms | 4.44 ms |
| q5_1 (down) | 11.02 ms | 4.67 ms | 4.44 ms | 3.78 ms |
| q8_0 (down) | 10.94 ms | 7.97 ms | 7.98 ms | 7.40 ms |

The per-expert loop dominated MTP verify passes (stock) and prompt processing; with the grouped path a cold 135K read went from 382 s to 312-313 s (N=3 server end lines, one from a profiling run; a second profiling read ended at 295 s; [raw](../../models/qwen3.8-flash-next/benchmarks/raw/2026-09-26-cold-read-progress.csv)). See [the prefill finding](../../findings/llama-cpp-sycl-moe-prefill-host-sync.md) and the [cost breakdown](../../models/qwen3.8-flash-next/research/2026-09-26-grouped-moe-gemm-cost-breakdown.md) of the remaining 3.9 ms. Reordered MoE weights exist only for Q4_K, Q5_K and Q6_K (`opt_for_reorder_id`, `ggml-sycl.cpp:4730`), and the decode path reorders them lazily on first use, so a new MoE kernel must decode both the plain and the reordered layout ([finding](../../findings/llama-cpp-sycl-kquant-experts-reordered-lazily.md)); IQ-type experts use the fused path without reordering. A 3D `src1` makes SYCL `MUL_MAT` loop one product per slice: the MTP `eh_proj` cost 115 ms per 1536-token ubatch until 0016 reshaped it to 2D ([finding](../../findings/llama-cpp-sycl-batched-mul-mat-loops-per-slice.md), [raw op profile](../../models/qwen3.8-flash-next/benchmarks/raw/2026-09-26-prefill-op-profile.csv)).

## Sparse flash attention

- **Upstream, single token.** [#28796](https://github.com/ggml-org/llama.cpp/pull/28796), merged as `cd74ef627`, adds sparse flash attention for qwen4exp's QSA (Qwen sparse attention) layers, enabled with `GGML_SYCL_SPARSE_FA=1`. It only handles one query row, so with MTP on it never engages on the target's 4-token verify passes. It also needs a GGUF with correct `compress_ratios` ([quant note](../../models/qwen3.8-flash-next/research/2026-09-24-quants-and-uncensored-variants.md)).
- **Our patch 0006, multi-token.** Extends it to up to 32 query rows by gathering each row's cells and dispatching rows as sequences to the unchanged TILE kernel. At 135K context, decode median went from 22.2 to 34.5 tok/s ([experiment](../../models/qwen3.8-flash-next/experiments/2026-09-25-sparse-fa-multi-token.md)).
- **Our patch 0009, draft QSA.** The MTP draft head attends with QSA too. The GGUF converter wrote compress ratio 0 for the draft layer, so the draft scanned the whole context densely; at load it now gets the trunk's ratio (4), the draft context gets an attention-only hybrid memory with its own indexer cache and pooled keys (0007), and one-token draft decodes take the sparse path. 10-turn session at 135K: 70.5 ms per MTP step, 41.3 tok/s overall and 0.92 s prompt median, against 72.7-73.4 ms, 40.9-42.3 tok/s and 1.24-1.27 s in the two dense-draft sessions of the same day ([raw](../../models/qwen3.8-flash-next/benchmarks/raw/2026-09-26-lcbench-sessions-summary.csv), `lc-draftqsa` against `lc-mtpwin-0-allrows` and `lc-mtpwin-0-rowsonly`). A paired A/B in one session (12 turns, each decoded with and without the file `/tmp/llama-mtp-qsa-off`) gave acceptance 67.6% against 65.7%, 70.3 against 71.5 ms per step and 43.4 against 41.9 tok/s ([raw](../../models/qwen3.8-flash-next/benchmarks/raw/2026-09-25-pairprobe-draft-qsa.csv); [experiment](../../models/qwen3.8-flash-next/experiments/2026-09-25-mtp-draft-qsa.md)). `LLAMA_MTP_QSA=0` restores the dense draft.
- **Prompt batches stay dense.** Batches over 32 rows take the masked dense kernel. `GGML_SYCL_SPARSE_FA_DEBUG=2` (our patch 0017) logs how many cells tiles of consecutive prompt rows select together: at n_kv 34,304 one row keeps 2,051 cells, a 16-row tile 7,649 (max 9,251), 32 rows 10,863, 64 rows 15,834 and 128 rows 21,499 ([raw](../../models/qwen3.8-flash-next/benchmarks/raw/2026-09-26-fa-selection-union.csv)). With attention put at about 12% of a whole 135K cold read (session estimate, UNVERIFIED; a fit of the archived progress lines suggests a larger share) a tile-union sparse prompt kernel was judged worth about 3-5% reusing the existing kernels and about 10% with a custom kernel (estimates), and was not built ([research note](../../models/qwen3.8-flash-next/research/2026-09-26-sparse-prompt-attention-tile-union.md)).

Kernel time at 262,144 KV cells (`test-backend-ops perf`, head size 256, f16 KV): 3421.66 µs dense vs 54.48 µs sparse for 1 query row, and 3969.57 vs 209.18 µs for 4 rows.

## Speculative decoding

Native MTP (multi-token prediction) through PR #28243: `--spec-type draft-mtp --spec-draft-model <mtp gguf> --spec-draft-n-max 3`, with `-devd` to place the draft on one card. Use `-fit off`: the memory fitter cannot size the qwen4exp draft, and at `-ub 2048` it overflowed the last card ([quant note](../../models/qwen3.8-flash-next/research/2026-09-24-quants-and-uncensored-variants.md)). Draft lengths other than 3 gave no gain ([experiment](../../models/qwen3.8-flash-next/experiments/2026-09-24-mtp-draft-length-sweep.md)).

## Environment variables

| Variable | Default | From | Effect |
|---|---|---|---|
| `ZES_ENABLE_SYSMAN` | 0 | upstream | free-memory query; recommended with `-sm layer` |
| `UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS` | 0 | upstream | allows device allocations over 4 GiB |
| `GGML_SYCL_SPARSE_FA` | 0 | upstream #28796 | enables sparse flash attention |
| `GGML_SYCL_SPARSE_FA_MARGIN` | 256 | upstream #28796 | extra cells gathered past the hint |
| `GGML_SYCL_SPARSE_FA_DEBUG` | 0 | upstream #28796; `2` from our patch 0017 | `1`: upstream's debug logging. `2`: also logs selection-union sizes of prompt batches for tiles of 1, 16, 32, 64 and 128 rows, one call in 12 (diagnostic) |
| `GGML_SYCL_OP_DEVPROF` | 0 | our patches 0020 and 0023 | `<seconds>`: device-side op times from barrier timestamps, printed every `<seconds>` while `/tmp/ggml-sycl-op-devprof` exists. Per-op mode doubles the decode step, so op times turn host-paced; write `coarse` into the file for one span per graph at no measurable cost. Creates the queues with profiling on only when set |
| `LLAMA_SPEC_OVERRIDE` | unset | our patch 0018 | a file holding `n_max p_min`, re-read by the MTP draft on every draft call (capped at `--spec-draft-n-max`), for A/B tests within one server session |
| `GGML_SYCL_XMX_GATHER_TYPES` | all bits | upstream #29245 (copied as 0012); bits 9-12 from our patch 0013 | bitmask of expert formats that take the grouped XMX GEMM for `MUL_MAT_ID` batches over 8 tokens: bits 1-256 the IQ formats, 512 Q4_K, 1024 Q5_K, 2048 Q5_1, 4096 Q8_0. `0` restores the per-expert oneDNN loop for A/B tests |
| `LLAMA_MTP_QSA` | on | our patch 0009 | `0`: dense attention in the MTP draft head; the file `/tmp/llama-mtp-qsa-off` turns draft QSA off at runtime for A/B tests |
| `LLAMA_MTP_WINDOW` | unset | our patch 0008 | `=<tokens>`: sliding-window mask for the draft; disables draft QSA. 2048 and 8192 saved 4-5 ms per step but cost 4-8 points of acceptance, so leave it unset ([raw](../../models/qwen3.8-flash-next/benchmarks/raw/2026-09-26-lcbench-sessions-summary.csv), `lc-mtpwin-*`) |
| `LLAMA_HOST_PROF` | 0 | our patch 0008 | `=<seconds>`: host-side time per `process_ubatch` phase and per graph-input type, by context (target or draft) and batch-size class, printed every N seconds (diagnostic) |
| `GGML_SYCL_SPARSE_FA_MAX_Q` | 32 | our patch 0006 | most query rows the sparse path takes; the file `/tmp/ggml-sycl-sparse-fa-off` turns sparse attention off at runtime for A/B tests |
| `GGML_SYCL_MMID_MULTITOKEN` | 1 | our patch 0003 | fused per-token MUL_MAT_ID for small batches |
| `GGML_SYCL_MMID_MULTITOKEN_MAX` | 8 | our patch 0004 | largest batch for that path; 512 was slower on ~400-token prompt turns ([experiment](../../models/qwen3.8-flash-next/experiments/2026-09-25-mmid-multitoken-prompt-turns.md)) |
| `GGML_SYCL_OP_PROFILE` | 0 | our patch 0004 | `=<seconds>`: per-op timing printed every N seconds while `/tmp/ggml-sycl-op-profile` exists. It waits on the queue after each op, so read shares, not absolute times |
| `LLAMA_QSA_POOLED` | on | our patch 0007 | pooled QSA indexer-key cache; `0` disables it, and `/tmp/llama-qsa-pooled-off` toggles it at runtime |
| `GGML_SYCL_ENABLE_GRAPH` | – | upstream | no effect with more than one GPU |
| `GGML_SYCL_PRIORITIZE_DMMV` | 0 | upstream | leave off: it disables the shared-expert GLU fusion and gains nothing for Q8_0 |

## Performance summary

Qwen3.8-Flash-Next huihui-ai abliterated UD-Q4_K_XL, 262K context, MTP n-max 3, `-ub 1536`, sparse flash attention and pooled key cache on, 230 W per card, single stream. Full details in the [model card](../../models/qwen3.8-flash-next/README.md), the [production fnbench benchmark](../../models/qwen3.8-flash-next/benchmarks/2026-09-25-production-build.md) and the [cold-read progression](../../models/qwen3.8-flash-next/benchmarks/2026-09-26-cold-read-135k-by-build.md).

The fnbench rows are from build `20260925-f47a6a5f3` (2026-09-25). The later builds up to `20260926-2b84213a4` changed prompt processing and the draft head. Only the tree of `0f23f62c8` has one fnbench run (46.3 / 59.4 / 54.9 tok/s short / ~10K / ~39K, [raw](../../models/qwen3.8-flash-next/benchmarks/raw/2026-09-25-draft-qsa-fnbench.csv)); the other builds' fnbench numbers are UNVERIFIED.

| Workload | Build | Decode tok/s | Prompt tok/s |
|---|---|---|---|
| Short prompt | f47a6a5f3 | 48.6 (46.4-49.9) | – |
| ~10K prompt | f47a6a5f3 | 60.6 | 589.8 |
| ~39K prompt | f47a6a5f3 | 53.6 | 577.2 |
| ~219K prompt | f47a6a5f3 | 40.6 | 268.3 |
| Agent session at 135K (cached prefix, ~60-400 new tokens per turn) | f47a6a5f3 | median 39.1-48.5 | [A/B experiment](../../models/qwen3.8-flash-next/experiments/2026-09-25-pooled-qsa-key-cache.md) |
| Cold read of a 135K-token code context (lcbench) | a890bf8b0 (the production code without the diagnostic patch 0017) | – | 284 s wall: 283.66 s in the lcbench run ([raw](../../models/qwen3.8-flash-next/benchmarks/raw/2026-09-26-cold-read-windows.csv), row `bcp5hgsyz`) and 283.10 s at its server end line (476 tok/s; [raw](../../models/qwen3.8-flash-next/benchmarks/raw/2026-09-26-cold-read-progress.csv)); 285.88 s on 2b84213a4 with `-b 3072` (472 tok/s, same raw file). Progression: 455 s (993baf141, 2026-09-25, windows CSV row `br63o3b14`; its only server end line, 447.01 s, is from a profiling run), 382 s (2c88bdf19), 312-313 s (517eb833d and 85b26781b, N=3 including one profiling run; a second profiling read: 295 s), 283-286 s (a890bf8b0 and 2b84213a4, N=2), progress CSV |
| Follow-up turns at 135K after the cold read (~60-470 new tokens) | a890bf8b0 | median 43.7 over 10 turns ([raw](../../models/qwen3.8-flash-next/benchmarks/raw/2026-09-26-cold-read-windows.csv), row `bcp5hgsyz`) | prompt median about 0.76 s per turn: 0.90 s before patch 0015, 0.77 s after it and 0.76 s after 0016 (same raw file, rows `bhc79ftn4`, `b2pfxuo2i`, `bcp5hgsyz`; 0016 did not move it), and about 1.24 s before 0009 ([raw](../../models/qwen3.8-flash-next/benchmarks/raw/2026-09-26-lcbench-sessions-summary.csv), `lc-mtpwin-0-allrows`) |

For comparison, the dense Qwen3.8-27B on vLLM at TP4 on the same four cards decodes at 129.1 tok/s single-stream ([benchmark](../../models/qwen3.8-27b/benchmarks/2026-09-23-tp4-quad-b70.md)). That is a different model; tensor parallelism is what lets it read weights on all four cards at once.

## Notes

- Pinning each card's minimum GPU clock to its maximum while serving speeds up layer-split decode ([finding](../../findings/gpu-clock-floor-speeds-layer-split.md)).
- Never run a second llama-server next to a loaded one on `xe`: VRAM overcommit spills silently into host RAM that cannot be swapped ([finding](../../findings/xe-vram-overcommit-spills-into-host-ram.md)).
- Open gaps (upstream status checked 2026-09-26): no async cross-device copies ([#29398](https://github.com/ggml-org/llama.cpp/issues/29398), open feature request); the grouped MoE GEMM for K-quant, Q5_1 and Q8_0 experts exists only locally (patches 0012-0015), while upstream [#29245](https://github.com/ggml-org/llama.cpp/pull/29245) (IQ types) is still open at `950f1c4aa`; Q8_0 kernel PRs [#29186](https://github.com/ggml-org/llama.cpp/pull/29186) and [#29337](https://github.com/ggml-org/llama.cpp/pull/29337) still open; tensor split for qwen4exp [#28569](https://github.com/ggml-org/llama.cpp/pull/28569) still open. A service restart empties the prompt cache; slot save/restore with the draft (0010) and `--cache-ram` cover it ([finding](../../findings/llama-server-restart-drops-prefix-cache.md)).

## Host-to-device copies (2026-09-26)

Upstream's `ggml_backend_sycl_buffer_set_tensor` copies every call through a fresh `malloc` plus `memcpy` (a workaround for mmap'd sources on PVC). For multi-GiB copies the page faults dominate. Loading a 135K-token Flash-Next state back from the host-RAM prompt cache took 2.7 s, and a slot restore from disk 3.2 s. Our patch 0022 stages the copy through two reusable pinned 32 MiB buffers per device. That brought the load to about 0.43 s and the restore to 1.0 s, and cut service start to ready from about 75 s to 33 s, because model weights take the same path ([experiment](../../models/qwen3.8-flash-next/experiments/2026-09-26-prompt-cache-swap-speedup.md)). Device-to-host copies into already-touched memory ran at about 9 GB/s and were not changed.
