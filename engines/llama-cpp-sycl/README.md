# llama.cpp (SYCL backend)

Tested on Arc Pro B70 since 2026-09-24, on host [quad-b70-5800x-pex88096](../../hardware/hosts/quad-b70-5800x-pex88096.md). It is the primary engine for [Qwen3.8-Flash-Next](../../models/qwen3.8-flash-next/), because upstream vLLM refuses that model on XPU ([deep dive](../../models/qwen3.8-flash-next/research/2026-09-25-vllm-xpu-deep-dive.md)).

## Build

| Item | Value |
|---|---|
| Tree | [ggml-org/llama.cpp PR #28243](https://github.com/ggml-org/llama.cpp/pull/28243) (qwen4exp + MTP, open) at `6fcaa16`, plus our patches 0001-0007 ([list](../../models/qwen3.8-flash-next/configs/patches/)) |
| Frozen production build | `20260925-f47a6a5f3` |
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
| `-sm layer` | Works. The cards run one after another, so each card's compute engine is busy well under half the time (16-41% measured, by card and workload) and speed is bounded by the sum of per-card time | [decode analysis](../../models/qwen3.8-flash-next/research/2026-09-24-decode-bottleneck-and-speed-plan.md) |
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
| more than 8 | copies expert ids to the host and waits, then runs one oneDNN GEMM per expert. On Flash-Next: 144 host syncs and about 73.7K GEMMs per micro-batch | yes |

The slow path dominates MTP verify passes (stock) and prompt processing. See [the prefill finding](../../findings/llama-cpp-sycl-moe-prefill-host-sync.md). Reordered MoE weights exist only for Q4_K, Q5_K and Q6_K (`opt_for_reorder_id`, `ggml-sycl.cpp:4730`); IQ-type experts use the fused path without reordering.

## Sparse flash attention

- **Upstream, single token.** [#28796](https://github.com/ggml-org/llama.cpp/pull/28796), merged as `cd74ef627`, adds sparse flash attention for qwen4exp's QSA (Qwen sparse attention) layers, enabled with `GGML_SYCL_SPARSE_FA=1`. It only handles one query row, so with MTP on it never engages on the target's 4-token verify passes. It also needs a GGUF with correct `compress_ratios` ([quant note](../../models/qwen3.8-flash-next/research/2026-09-24-quants-and-uncensored-variants.md)).
- **Our patch 0006, multi-token.** Extends it to up to 32 query rows by gathering each row's cells and dispatching rows as sequences to the unchanged TILE kernel. At 135K context, decode median went from 22.2 to 34.5 tok/s ([experiment](../../models/qwen3.8-flash-next/experiments/2026-09-25-sparse-fa-multi-token.md)).

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
| `GGML_SYCL_SPARSE_FA_DEBUG` | 0 | upstream #28796 | debug logging |
| `GGML_SYCL_SPARSE_FA_MAX_Q` | 32 | our patch 0006 | most query rows the sparse path takes; the file `/tmp/ggml-sycl-sparse-fa-off` turns sparse attention off at runtime for A/B tests |
| `GGML_SYCL_MMID_MULTITOKEN` | 1 | our patch 0003 | fused per-token MUL_MAT_ID for small batches |
| `GGML_SYCL_MMID_MULTITOKEN_MAX` | 8 | our patch 0004 | largest batch for that path; 512 was slower on ~400-token prompt turns ([experiment](../../models/qwen3.8-flash-next/experiments/2026-09-25-mmid-multitoken-prompt-turns.md)) |
| `GGML_SYCL_OP_PROFILE` | 0 | our patch 0004 | `=<seconds>`: per-op timing printed every N seconds while `/tmp/ggml-sycl-op-profile` exists. It waits on the queue after each op, so read shares, not absolute times |
| `LLAMA_QSA_POOLED` | on | our patch 0007 | pooled QSA indexer-key cache; `0` disables it, and `/tmp/llama-qsa-pooled-off` toggles it at runtime |
| `GGML_SYCL_ENABLE_GRAPH` | – | upstream | no effect with more than one GPU |
| `GGML_SYCL_PRIORITIZE_DMMV` | 0 | upstream | leave off: it disables the shared-expert GLU fusion and gains nothing for Q8_0 |

## Performance summary

Qwen3.8-Flash-Next huihui-ai abliterated UD-Q4_K_XL, 262K context, MTP n-max 3, `-ub 1536`, sparse flash attention and pooled key cache on, 230 W per card, single stream. Full details in the [model card](../../models/qwen3.8-flash-next/README.md) and [production benchmark](../../models/qwen3.8-flash-next/benchmarks/2026-09-25-production-build.md).

| Workload | Decode tok/s | Prompt tok/s |
|---|---|---|
| Short prompt | 48.6 (46.4-49.9) | – |
| ~10K prompt | 60.6 | 589.8 |
| ~39K prompt | 53.6 | 577.2 |
| ~219K prompt | 40.6 | 268.3 |
| Agent session at 135K (cached prefix, ~60-400 new tokens per turn) | median 39.1-48.5 | [A/B experiment](../../models/qwen3.8-flash-next/experiments/2026-09-25-pooled-qsa-key-cache.md) |

For comparison, the dense Qwen3.8-27B on vLLM at TP4 on the same four cards decodes at 129.1 tok/s single-stream ([benchmark](../../models/qwen3.8-27b/benchmarks/2026-09-23-tp4-quad-b70.md)). That is a different model; tensor parallelism is what lets it read weights on all four cards at once.

## Notes

- Pinning each card's minimum GPU clock to its maximum while serving speeds up layer-split decode ([finding](../../findings/gpu-clock-floor-speeds-layer-split.md)).
- Never run a second llama-server next to a loaded one on `xe`: VRAM overcommit spills silently into host RAM that cannot be swapped ([finding](../../findings/xe-vram-overcommit-spills-into-host-ram.md)).
- Open gaps: no async cross-device copies ([#29398](https://github.com/ggml-org/llama.cpp/issues/29398)), no grouped MoE GEMM for K-quant experts ([#29245](https://github.com/ggml-org/llama.cpp/pull/29245) covers IQ types only), and Q8_0 kernel PRs [#29186](https://github.com/ggml-org/llama.cpp/pull/29186) and [#29337](https://github.com/ggml-org/llama.cpp/pull/29337) not yet merged.
