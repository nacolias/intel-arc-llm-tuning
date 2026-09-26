# Baseline: llama.cpp SYCL across 4 cards, 262K context with the MTP draft

| | |
|---|---|
| **Date** | 2026-09-24 |
| **Model / checkpoint** | unsloth `UD-Q4_K_XL` (stock) + unsloth MTP draft `mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf` |
| **Host** | [quad-b70-5800x-pex88096](../../../hardware/hosts/quad-b70-5800x-pex88096.md) |
| **Status** | kept (service default from 2026-09-24) |
| **Baseline** | first runs of the day, same file and build: see the table below |
| **Result** | [benchmarks/2026-09-24-baseline-262k-mtp.md](../benchmarks/2026-09-24-baseline-262k-mtp.md) |
| **Related** | [research: quants and uncensored variants](../research/2026-09-24-quants-and-uncensored-variants.md), [finding: xe VRAM overcommit spills into host RAM](../../../findings/xe-vram-overcommit-spills-into-host-ram.md) |

## Hypothesis

With `-ub 2048` and the memory fitter, 262K context plus the MTP (multi-token prediction) draft overflowed the last card and decode fell to 0.6 tok/s. The fitter leaves the draft out of its sizing. At `-ub 2048` the QSA (Qwen sparse attention) indexer takes about 7 GiB of compute buffer per card, and that buffer scales with the ubatch. Halving the ubatch and splitting layers explicitly should free enough VRAM for the draft on the last card, and MTP should then beat MTP off at depth.

## Change

```diff
- -c 262144 -b 2048 -ub 2048                        --spec-type draft-mtp --spec-draft-model <mtp> --spec-draft-n-max 3
+ -c 262144 -b 2048 -ub 1024 -ts 13,13,13,10 -fit off --spec-type draft-mtp --spec-draft-model <mtp> --spec-draft-n-max 3 -devd SYCL3
```

Common flags in every run: `-dev SYCL0,SYCL1,SYCL2,SYCL3 -sm layer -ngl 99 -ot per_layer_token_embd=CPU -np 1 -fa on -ctk f16 -ctv f16 -t 8 --jinja`, environment `ZES_ENABLE_SYSMAN=1 UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1`.

`-ts` takes whole numbers that sum to 49, which gives exact layer counts: 48 layers plus the output layer.

## Environment

| Component | Version / value |
|---|---|
| Engine and image (tag and digest) | llama.cpp, no container. [ggml-org/llama.cpp PR #28243](https://github.com/ggml-org/llama.cpp/pull/28243) at `6fcaa16`, unpatched |
| Build flags | `GGML_SYCL=ON GGML_SYCL_TARGET=INTEL GGML_SYCL_F16=ON GGML_SYCL_DNN=ON GGML_SYCL_GRAPH=ON GGML_SYCL_SUPPORT_LEVEL_ZERO_API=ON GGML_NATIVE=ON`, `CMAKE_BUILD_TYPE=Release`, icx/icpx from oneAPI 2026.1 (DPC++ 2026.1.1) |
| compute-runtime / IGC | 26.35.39758.11 / IGC 2.41.9; Level Zero loader 1.32.0 |
| Kernel / `xe` driver / GuC firmware | 7.0.0-34-generic / in-tree `xe` / GuC 70.58.0 |
| oneCCL | not used |
| PyTorch XPU | not used |
| Power cap per card | 230 W |
| Tensor / pipeline parallel | layer split (`-sm layer`) across 4 cards; no tensor parallel |
| GPU clock floor | none (driver default) |

## Procedure

1. Stop every other GPU user. A second tenant on the cards spills VRAM into host RAM on `xe`.
2. Start `llama-server` with the flags above and a host-RAM watchdog logging per-card VRAM and driver-held memory every few seconds.
3. Run [`tools/fnbench.py`](../../../tools/fnbench.py): a sanity question, 3 short-context runs of 512 tokens, then prompts of 9,749 and 39,119 tokens with 256 tokens decoded. For the 262K configs, also `LONG_TOKENS=200000` (a 219,217-token prompt). Greedy, `ignore_eos`, no prompt cache.
4. One run per config.

## Results

All decode figures are single-stream tok/s. Raw rows: [raw/2026-09-24-baseline-fnbench.csv](../benchmarks/raw/2026-09-24-baseline-fnbench.csv), [raw/2026-09-24-baseline-memory.csv](../benchmarks/raw/2026-09-24-baseline-memory.csv).

| Config | Short | ~10k | ~39k | 219K | Prompt tok/s (10k / 39k / 219K) | Busiest card, peak | Driver-held host RAM, peak |
|---|---|---|---|---|---|---|---|
| 262K, `-ub 2048`, fitter, no MTP | 33.0-33.9 | 31.8 | 23.1 | not run | 711.7 / 700.1 / - | 30,949 MiB | 26.8 GiB |
| 262K, `-ub 2048`, fitter, MTP | 0.6 (thrashing, stopped) | - | - | - | - | 32,636 MiB (GPU3) | 57.8 GiB |
| 64K, `-ub 1024`, fitter, no MTP | 33.6-33.9 | 32.1 | 23.3 | - | 572.7 / 569.8 / - | 23,531 MiB | 8.1 GiB |
| 64K, `-ub 1024`, fitter, MTP | 35.2-36.3 | 41.3 | 39.6 | - | 535.5 / 523.2 / - | 24,589 MiB | 8.6 GiB |
| 262K, `-ub 1024 -ts 12,12,12,13`, no MTP | 33.1-33.5 | 31.4 | 23.4 | 7.2 | 568.8 / 570.6 / 274.8 | 26,115 MiB | 16.8 GiB |
| **262K, `-ub 1024 -ts 13,13,13,10 -fit off`, MTP** | **33.8-36.9** | **43.3** | **33.4** | **16.1** | **531.6 / 523.3 / 254.5** | **27,420 MiB** | **17.5 GiB** |

| Metric | MTP off (`-ts 12,12,12,13`) | MTP on (`-ts 13,13,13,10`) | Delta |
|---|---|---|---|
| Single-stream tok/s, short (median of 3) | 33.4 | 35.9 | +7% |
| Single-stream tok/s, ~10k | 31.4 | 43.3 | +38% |
| Single-stream tok/s, ~39k | 23.4 | 33.4 | +43% |
| Single-stream tok/s, 219K | 7.2 | 16.1 | +124% |
| Prompt tok/s, 219K | 274.8 | 254.5 | -7% |
| Spec-decode acceptance | - | 58-65% short; 88% at 10k; 86% at 39k; 73% at 219K | |
| Aggregate at c8 / c16 | not applicable: one slot (`-np 1`) | | |

- The ~10k and ~39k prompts repeat one paragraph, and the 219K prompt is structured records. Both flatter MTP. Short-prompt acceptance (58-65%) is the more realistic figure.
- VRAM stays flat as the context fills, because the KV cache is allocated at load.
- A 219K-token prompt takes 13 to 15 minutes to read.

## Correctness

- All runs answered the sanity question correctly (155 minutes).
- MTP drafts are verified by the target model, so they should not change the output. We did not compare outputs token by token.
- The long-prompt answers were not graded.
- The thrashing run is the failure mode in [findings/xe-vram-overcommit-spills-into-host-ram.md](../../../findings/xe-vram-overcommit-spills-into-host-ram.md): VRAM overcommit does not fail on `xe`, it spills into host RAM that cannot be swapped.

## Decision

Kept. `-ub 1024 -ts 13,13,13,10 -fit off` with MTP became the service default on 2026-09-24. It fits 262K with MTP on every card, with the busiest card at 27,420 MiB. MTP is worth +38% to +124% at depth on these prompts.

## Follow-ups

- [x] Decode is host-bound: see [research: decode bottleneck and speed plan](../research/2026-09-24-decode-bottleneck-and-speed-plan.md) and the [fused MUL_MAT_ID experiment](2026-09-24-fused-mul-mat-id-mtp-verify.md).
- [x] Re-tune the ubatch after the speed work: [ubatch 1536](2026-09-25-ubatch-1536.md).
- [x] Long-context decode: [sparse FA for multi-token batches](2026-09-25-sparse-fa-multi-token.md).
