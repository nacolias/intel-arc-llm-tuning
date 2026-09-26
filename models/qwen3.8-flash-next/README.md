# Qwen3.8-Flash-Next

| | |
|---|---|
| **Status** | production |
| **Upstream** | [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) |
| **License** | Qwen Community License 1.0 (`qwen-community-1.0` on the Hugging Face card); terms not reviewed here |
| **Host** | [quad-b70-5800x-pex88096](../../hardware/hosts/quad-b70-5800x-pex88096.md), 4x Arc Pro B70 behind a PCIe switch |
| **Engine** | [llama.cpp SYCL](../../engines/llama-cpp-sycl/), PR #28243 at `6fcaa16` plus [patches](configs/patches/README.md); build `20260925-f47a6a5f3` |
| **Last updated** | 2026-09-25 |

On four B70s this 125B-parameter MoE model serves the full 262,144-token context at about 49 tok/s on short prompts and about 40 tok/s at 135K-219K, single stream. It started at 34-37 and 16 tok/s on 2026-09-24. vLLM does not run it on XPU yet, so everything here is llama.cpp.

## Architecture

Read from the GGUF headers and a tensor dump on our host, unless marked otherwise. The research notes on [long-context QSA](research/2026-09-25-long-context-qsa-design.md) and [decode speed](research/2026-09-24-decode-bottleneck-and-speed-plan.md) have the details.

| Property | Value |
|---|---|
| Parameters (total / active) | about 125B in the main model plus a 51B-parameter per-layer n-gram embedding table (PLE); about 6B active per token. From the [Qwen](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) and [unsloth](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) model cards (upstream-documented) |
| Layers and layer types | 48 layers. 12 full-attention layers with QSA (Qwen sparse attention) at layers 3, 7, 11, ..., 47 (compress ratio 4); the other 36 are Gated DeltaNet (linear attention) |
| Hidden size / heads / KV heads | 2560 / 24 query heads / 2 KV heads, head dim 256 |
| QSA indexer | 4 heads x 128, top-k 2048 tokens (512 blocks of 4); llama.cpp selects over expanded cells with width 2051 |
| Experts | 512 routed experts, top-10, plus a shared expert, in every layer |
| Residual stream | DeepSeek-V4-style hyper-connections (llama.cpp `DSV4_HC` ops) |
| PLE n-gram table | about 51B parameters; 27 GiB as IQ4_NL in `UD-Q4_K_XL`; read lazily from CPU memory (`GET_ROWS` on the host) |
| Vocabulary | 248,320 |
| Native context length | 262,144 |
| Native speculative head (MTP) | yes: one MTP (multi-token prediction) layer, shipped as a separate draft GGUF |
| Vision | yes, through a separate `mmproj` projector GGUF |

**Why the architecture matters on Arc:**

- **MoE on SYCL.** `MUL_MAT_ID` has a fused single-token kernel. Anything with more than one token fell back to a loop that copies expert ids to the host, waits, and runs one matmul per expert: 144 host syncs per pass. That made MTP verify expensive (fixed for 2-8 tokens by [patch 0003](configs/patches/0003-sycl-fused-mul-mat-id-2-8-tokens.diff)) and still bounds prefill and prompt turns. See [findings/llama-cpp-sycl-moe-prefill-host-sync.md](../../findings/llama-cpp-sycl-moe-prefill-host-sync.md).
- **Layer split only.** qwen4exp is excluded from `-sm tensor`, the SYCL all-reduce supports only 2 devices, and `-sm row` cannot split MoE tensors. With `-sm layer` the four cards run one after another, so decode is host- and launch-bound: about 4,000 kernel launches per token, main thread at 96% of one core, each card's compute engine 16-18% busy ([raw/2026-09-24-decode-profile.csv](benchmarks/raw/2026-09-24-decode-profile.csv)). A token reads 6.33 GB of weights plus about 0.45 GB of Gated DeltaNet state; at an assumed 608 GB/s per card (UNVERIFIED) sequential layer split caps decode at about 90 tok/s, or about 127 with MTP ([roofline](research/2026-09-24-decode-bottleneck-and-speed-plan.md#roofline-under-layer-split)).
- **No SYCL graphs.** Graphs are refused with more than one device and with any `MUL_MAT_ID` ([finding](../../findings/llama-cpp-sycl-graphs-off-with-multiple-gpus.md)).
- **QSA needs sparse attention kernels.** Dense flash attention scans the whole cache even though only about 2,051 cells are selected. Upstream SYCL sparse FA covered single-token queries only; [patch 0006](configs/patches/0006-sycl-sparse-fa-multi-token.diff) extends it to MTP verify batches.
- **VRAM overcommit is silent.** `xe` spills overcommitted VRAM into host RAM that cannot be swapped, and llama.cpp's memory fitter cannot size the MTP draft. The split must be explicit (`-fit off -ts ...`). See [findings/xe-vram-overcommit-spills-into-host-ram.md](../../findings/xe-vram-overcommit-spills-into-host-ram.md).

## Checkpoints

| Checkpoint | Quant method | Bits / group | Size on disk | XPU kernel path | Status |
|---|---|---|---|---|---|
| [huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-GGUF](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-GGUF) `UD-Q4_K_XL` | unsloth dynamic quant with abliteration applied to the Q8_0 tensors only | routed experts mostly Q4_K (most down projections Q5_1), trunk and shared expert Q8_0, PLE IQ4_NL | 111.33 GB, 4 shards | SYCL MMVQ with the fused MoE path (Q4_K, Q5_K, Q5_1, Q8_0) | **best**, in production since 2026-09-24 |
| [unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) `UD-Q4_K_XL` | unsloth dynamic quant | same tensor layout and types as above | 111.33 GB, 4 shards | same | tried: baseline benchmarks; stock (censored) fallback |
| unsloth `MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf` | Q8_0; borrows `token_embd` and `output` from the main model | 8 | 2.79 GB | SYCL MMVQ, runs on the last card (`-devd SYCL3`) | best (MTP draft) |
| huihui-ai `mmproj-model-bf16.gguf` | BF16 vision projector | 16 | 0.91 GB | SYCL | in production (vision) |
| IQ4_XS and other uncensored quants (heretic-2, orcarouter, apetersson LoRA, ...) | various | | 94-140 GB | IQ experts miss the reordered MoE kernel | not tried; see [research: quants](research/2026-09-24-quants-and-uncensored-variants.md) |

Sizes are the file sizes listed on Hugging Face. The two `UD-Q4_K_XL` files have identical tensor layouts (1,224 tensors, same types) and the same VRAM use and speed on our host.

On a 10-prompt mild refusal smoke test (one sample each), stock refused 2/10 and hedged 2/10; the abliterated file refused and hedged 0/10. Both passed 5/5 simple capability checks ([raw/2026-09-24-refusal-smoke-probe.csv](benchmarks/raw/2026-09-24-refusal-smoke-probe.csv)). This is a smoke test, not a quality benchmark.

## Current best config

- Config: [configs/llama-server-262k-mtp.sh](configs/llama-server-262k-mtp.sh)
- Patches: [configs/patches/](configs/patches/README.md)
- Benchmarks: [production build](benchmarks/2026-09-25-production-build.md), [agent session at 135K](benchmarks/2026-09-25-agent-session-135k.md)

| Setting | Value |
|---|---|
| Engine | llama.cpp SYCL: PR #28243 at `6fcaa16` + #28931 + 0002 + 0003 + 0004 + #28796 + 0006 + 0007 (build `20260925-f47a6a5f3`) |
| Parallelism | `-sm layer` across 4 cards, `-ts 12,13,13,11`, `-fit off`; no tensor parallel |
| Context and batching | `-c 262144 -np 1 -b 2048 -ub 1536` |
| KV cache | f16 K and V, allocated at load (about 6 GiB at 262K) |
| Speculative decoding | MTP draft, `--spec-draft-n-max 3`, draft on the last card |
| Attention | `-fa on`, `GGML_SYCL_SPARSE_FA=1` (multi-token sparse FA), pooled QSA key cache on (patch 0007 default) |
| PLE table | CPU memory (`-ot per_layer_token_embd=CPU`, redundant but kept) |
| Graph mode | eager |
| Power and clocks | 230 W cap per card; GPU clock floor `min_freq = rp0` (2800 MHz) while serving ([`tools/gpu-clock-floor.sh`](../../tools/gpu-clock-floor.sh)) |
| Idle power | PCIe ASPM L1 on the card links ([`tools/bmg-aspm-l1.sh`](../../tools/bmg-aspm-l1.sh)) |

## Headline numbers

Single stream, greedy. Short, ~10k and ~39k are the [`tools/fnbench.py`](../../tools/fnbench.py) points (19, 9,749 and 39,119 prompt tokens; the long prompts are repetitive and flatter MTP). The 135K figure is a code workload from [`tools/lcbench.py`](../../tools/lcbench.py).

| Metric | Value | Benchmark |
|---|---|---|
| Single-stream decode, short prompt | 48.6-50.5 tok/s (3 sessions) | [production build](benchmarks/2026-09-25-production-build.md) |
| Single-stream decode, ~10k context | 60.6-67.2 tok/s | [production build](benchmarks/2026-09-25-production-build.md) |
| Single-stream decode, ~39k context | 53.6-55.2 tok/s | [production build](benchmarks/2026-09-25-production-build.md) |
| Single-stream decode, 135K agent session | about 39 tok/s (medians 39.1 and 48.5; acceptance 62.5% and 82.7%); MTP step about 73 ms | [agent session](benchmarks/2026-09-25-agent-session-135k.md) |
| Single-stream decode, 219K context | 40.6 tok/s | [production build](benchmarks/2026-09-25-production-build.md) |
| Prompt processing, 10k-39k | 568-645 tok/s | [production build](benchmarks/2026-09-25-production-build.md) |
| Prompt processing, 219K (cold) | 268.3 tok/s, 816.9 s | [production build](benchmarks/2026-09-25-production-build.md) |
| Follow-up turn, ~60 new tokens on a cached 135K prefix | 1.04-1.32 s | [agent session](benchmarks/2026-09-25-agent-session-135k.md) |
| Aggregate at c8 / c16 | not measured: single slot | |
| Max context served | 262,144 tokens | [production build](benchmarks/2026-09-25-production-build.md) |
| Peak VRAM, busiest card, at 219K | 28,957 MiB | [production build](benchmarks/2026-09-25-production-build.md) |
| Idle power, 4 cards, model loaded | 185 W; 25-27 W with ASPM L1 | [finding](../../findings/bmg-aspm-l1-idle-power.md) |

## Progress

| Stage | Date | Short | ~10k | ~39k | 219K | 135K | Experiment |
|---|---|---|---|---|---|---|---|
| Baseline: unpatched, `-ub 1024 -ts 13,13,13,10`, MTP | 2026-09-24 | 33.8-36.9 | 43.3 | 33.4 | 16.1 | - | [baseline](experiments/2026-09-24-baseline-llamacpp-mtp-262k.md) |
| + fused `MUL_MAT_ID` for MTP verify, gettid cache, #28931 | 2026-09-24 | 42-44 | 52-54 | 44-45 | - | - | [fused MUL_MAT_ID](experiments/2026-09-24-fused-mul-mat-id-mtp-verify.md) |
| + GPU clock floor 2800 MHz | 2026-09-24 | 50 | 64-66 | 47-49 | - | - | [clock floor](experiments/2026-09-24-gpu-clock-floor.md) |
| + `-ub 1536 -ts 12,13,13,11` | 2026-09-25 | 50.3 | 63.6 | 52.6 | 18.4 | 22.3-23.3 | [ubatch 1536](experiments/2026-09-25-ubatch-1536.md) |
| + multi-token sparse FA | 2026-09-25 | 49.3 | 63.1 | 49.7 | - | 31.3-34.5 | [sparse FA](experiments/2026-09-25-sparse-fa-multi-token.md) |
| + pooled QSA key cache (production) | 2026-09-25 | 48.6-50.5 | 60.6-67.2 | 53.6-55.2 | 40.6 | 39.1-48.5 | [pooled cache](experiments/2026-09-25-pooled-qsa-key-cache.md) |

Decode in tok/s. A dash means not measured at that stage. The full log, including what was reverted, is in [experiments/README.md](experiments/README.md).

## What worked and what did not

- **Worked:**
  - MTP with an explicit split: +38% to +124% over MTP off at depth once `-ub 1024 -ts 13,13,13,10 -fit off` made room for the draft.
  - Fused `MUL_MAT_ID` for 2-8-token batches, shipped together with the `gettid` cache and #28931: +14% to +29% decode, the largest single gain at short and medium context.
  - GPU clock floor at 2800 MHz: +4% to +27% decode between range extremes (+8% to +23% between range midpoints), about 0 W at true idle.
  - Multi-token sparse flash attention: +41% to +55% decode at 135K.
  - Pooled QSA key cache: MTP step at 135K from 95.8 to about 73 ms; with sparse FA, 219K decode from 18.4 to 40.6 tok/s.
  - `-ub 1536`: +4.5% to +6% prefill at depth.
  - PCIe ASPM L1: idle power of the four cards from 185 W to 25-27 W, with no decode or prefill cost ([finding](../../findings/bmg-aspm-l1-idle-power.md)).
- **Did not work:**
  - SYCL graphs: no effect, refused with more than one device.
  - Longer MTP drafts (n-max 4; n-max 6 with p-min 0.75): no consistent gain.
  - Pipeline-parallel prefill: no gain, +2.6 to +2.9 GiB VRAM per card.
  - Per-token fused `MUL_MAT_ID` for batches up to 512 tokens made prompt turns of about 400 tokens 16-33% slower.
  - `-ub 2048` with MTP at 262K: the last card overflowed and decode fell to 0.6 tok/s.
  - vLLM: upstream v0.30.0 raises `NotImplementedError` for Qwen4Exp on XPU; not pursued ([research](research/2026-09-25-vllm-xpu-deep-dive.md)).

## Open questions

- [ ] The MTP draft layer still attends densely over the whole context: 9.4% of decode op time at 135K, 1.87 ms per call. The draft GGUF ships indexer weights, but its compress ratio for that layer is 0.
- [ ] Follow-up turns of about 60 tokens take about 1.1 s at 135K, and prefill runs at about 580 tok/s. Both are MoE-bound. A grouped multi-token `MUL_MAT_ID` kernel for SYCL (each expert read once for its routed tokens, ids kept on the device) is the next lever.
- [ ] llama.cpp's QSA top-k over expanded cells can attend up to 3 cells of a 513th block, unlike the reference. Fix it (changes output, matches the reference) or leave it?
- [ ] vLLM route: a community fork reports 52.5 tok/s single stream and 626-634 tok/s at 16 streams on another 4x B70 host (UNVERIFIED), but runs dense attention without prefix caching, so every agent turn would re-read the whole context. Revisit when Intel llm-scaler PR #660 or upstream XPU support lands.
- [ ] No long-context quality evaluation of the abliterated checkpoint beyond the refusal smoke test and teacher-forced KL checks between builds.
- [ ] Rerun the draft-length sweep (n-max 2, p-min 0.5-0.85) on the agent-session workload.
- [ ] Newer card firmware (GSC 31.1062) is on LVFS and not applied; it mentions better low-power state entry with no display attached.

## Index

- [Research notes](research/README.md)
- [Experiments log](experiments/README.md)
- [Benchmarks](benchmarks/README.md)
- [Configs](configs/README.md)
