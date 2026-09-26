# Qwen3.8-Flash-Next

| | |
|---|---|
| **Status** | production |
| **Upstream** | [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) |
| **License** | Qwen Community License 1.0 (`qwen-community-1.0` on the Hugging Face card); terms not reviewed here |
| **Host** | [quad-b70-5800x-pex88096](../../hardware/hosts/quad-b70-5800x-pex88096.md), 4x Arc Pro B70 behind a PCIe switch |
| **Engine** | [llama.cpp SYCL](../../engines/llama-cpp-sycl/), PR #28243 at `6fcaa16` plus patches 0002-0017 ([list](configs/patches/README.md)); build `20260926-2b84213a4`, in production since 2026-09-26 |
| **Last updated** | 2026-09-26 |

On four B70s this 125B-parameter mixture-of-experts (MoE) model serves the full 262,144-token context, single stream. A cold read of a 135K-token code context takes about 284 s on the production code (one read at the default flags), down from 455 s on 2026-09-25; follow-up turns of about 60 new tokens take 0.76-0.77 s; decode at 135K runs at 43.7-48.7 tok/s, greedy ([cold read by build](benchmarks/2026-09-26-cold-read-135k-by-build.md)). Short prompts decode at 48.6-50.5 tok/s on build `20260925-f47a6a5f3`, not re-measured since ([production build](benchmarks/2026-09-25-production-build.md)). It started at 34-37 tok/s short and 16 tok/s at 219K on 2026-09-24 ([baseline](experiments/2026-09-24-baseline-llamacpp-mtp-262k.md)). vLLM does not run it on XPU yet, so everything here is llama.cpp.

## Architecture

Read from the GGUF headers and a tensor dump on our host, unless marked otherwise. The research notes on [long-context QSA (Qwen sparse attention)](research/2026-09-25-long-context-qsa-design.md) and [decode speed](research/2026-09-24-decode-bottleneck-and-speed-plan.md) have the details.

| Property | Value |
|---|---|
| Parameters (total / active) | about 125B in the main model plus a 51B-parameter per-layer n-gram embedding table (PLE); about 6B active per token. From the [Qwen](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) and [unsloth](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) model cards (upstream-documented) |
| Layers and layer types | 48 layers. 12 full-attention layers with QSA (Qwen sparse attention) at layers 3, 7, 11, ..., 47 (compress ratio 4); the other 36 are Gated DeltaNet (linear attention) |
| Hidden size / heads / KV (key/value) heads | 2560 / 24 query heads / 2 KV heads, head dim 256 |
| QSA indexer | 4 heads x 128, top-k 2048 tokens (512 blocks of 4); llama.cpp selects over expanded cells with width 2051 |
| Experts | 512 routed experts, top-10, plus a shared expert, in every layer |
| Residual stream | DeepSeek-V4-style hyper-connections (llama.cpp `DSV4_HC` ops) |
| PLE n-gram table | about 51B parameters; 27 GiB as IQ4_NL in `UD-Q4_K_XL`; read lazily from CPU memory (`GET_ROWS` on the host) |
| Vocabulary | 248,320 |
| Native context length | 262,144 |
| Native speculative head (MTP) | yes: one MTP (multi-token prediction) layer, shipped as a separate draft GGUF |
| Vision | yes, through a separate `mmproj` projector GGUF |

**Why the architecture matters on Arc:**

- **MoE on SYCL.** `MUL_MAT_ID` has a fused single-token kernel. Anything with more than one token fell back to a loop that copies expert ids to the host, waits, and runs one matmul per expert: 144 host syncs per pass. That made MTP verify expensive (fixed for 2-8 tokens by [patch 0003](configs/patches/0003-sycl-fused-mul-mat-id-2-8-tokens.diff)) and bounded prefill. See [findings/llama-cpp-sycl-moe-prefill-host-sync.md](../../findings/llama-cpp-sycl-moe-prefill-host-sync.md). Since patches 0012-0015, batches over 8 tokens run one grouped XMX (Xe Matrix Extensions) GEMM (general matrix multiply) launch per call instead: on one 1,536-token ubatch in `test-backend-ops`, 2.7-2.9x faster for the Q4_K, Q5_K and Q5_1 experts and 1.5x for Q8_0 (other batch sizes not benchmarked). The expert ids still go to the host once per call ([grouped MoE GEMM](experiments/2026-09-26-grouped-moe-xmx-gemm.md)).
- **Layer split only.** qwen4exp is excluded from `-sm tensor`, the SYCL all-reduce supports only 2 devices, and `-sm row` cannot split MoE tensors. With `-sm layer` the four cards run one after another, so decode is host- and launch-bound: about 4,000 kernel launches per token, main thread at 96% of one core, each card's compute engine 16-18% busy ([raw/2026-09-24-decode-profile.csv](benchmarks/raw/2026-09-24-decode-profile.csv)). A token reads 6.33 GB of weights plus about 0.45 GB of Gated DeltaNet state; at an assumed 608 GB/s per card (UNVERIFIED) sequential layer split caps decode at about 90 tok/s, or about 127 with MTP ([roofline](research/2026-09-24-decode-bottleneck-and-speed-plan.md#roofline-under-layer-split)).
- **No SYCL graphs.** Graphs are refused with more than one device and with any `MUL_MAT_ID` ([finding](../../findings/llama-cpp-sycl-graphs-off-with-multiple-gpus.md)).
- **QSA needs sparse attention kernels.** Dense flash attention scans the whole cache even though only about 2,051 cells are selected. Upstream SYCL sparse FA (flash attention) covered single-token queries only; [patch 0006](configs/patches/0006-sycl-sparse-fa-multi-token.diff) extends it to MTP verify batches, and [patch 0009](configs/patches/0009-qwen4exp-mtp-draft-qsa.diff) gives the MTP draft layer QSA too. Prompt batches over 32 rows still attend densely: 32.8% of listed op time at about 100-120K context on build `85b26781b` ([prefill op profile](benchmarks/2026-09-26-prefill-op-profile.md)).
- **VRAM overcommit is silent.** `xe` spills overcommitted VRAM into host RAM that cannot be swapped, and llama.cpp's memory fitter cannot size the MTP draft. The split must be explicit (`-fit off -ts ...`). See [findings/xe-vram-overcommit-spills-into-host-ram.md](../../findings/xe-vram-overcommit-spills-into-host-ram.md).

## Checkpoints

| Checkpoint | Quant method | Bits / group | Size on disk | XPU kernel path | Status |
|---|---|---|---|---|---|
| [huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-GGUF](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-GGUF) `UD-Q4_K_XL` | unsloth dynamic quant with abliteration applied to the Q8_0 tensors only | routed experts mostly Q4_K (most down projections Q5_1), trunk and shared expert Q8_0, PLE IQ4_NL | 111.33 GB, 4 shards | SYCL MMVQ with the fused MoE path (Q4_K, Q5_K, Q5_1, Q8_0); grouped XMX GEMM for prompt batches (patches 0012-0015) | **best**, in production since 2026-09-24 |
| [unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) `UD-Q4_K_XL` | unsloth dynamic quant | same tensor layout and types as above | 111.33 GB, 4 shards | same | tried: baseline benchmarks; stock (censored) fallback |
| unsloth `MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf` | Q8_0; borrows `token_embd` and `output` from the main model | 8 | 2.79 GB | SYCL MMVQ, runs on the last card (`-devd SYCL3`); QSA with its own indexer since patch 0009 | best (MTP draft) |
| huihui-ai `mmproj-model-bf16.gguf` | BF16 vision projector | 16 | 0.91 GB | SYCL | in production (vision) |
| IQ4_XS and other uncensored quants (heretic-2, orcarouter, apetersson LoRA, ...) | various | | 94-140 GB | IQ experts miss the reordered MoE kernel | not tried; see [research: quants](research/2026-09-24-quants-and-uncensored-variants.md) |

Sizes are the file sizes listed on Hugging Face. The two `UD-Q4_K_XL` files have identical tensor layouts (1,224 tensors, same types) and the same VRAM use and speed on our host.

On a 10-prompt mild refusal smoke test (one sample each), stock refused 2/10 and hedged 2/10; the abliterated file refused and hedged 0/10. Both passed 5/5 simple capability checks ([raw/2026-09-24-refusal-smoke-probe.csv](benchmarks/raw/2026-09-24-refusal-smoke-probe.csv)). This is a smoke test, not a quality benchmark.

## Current best config

- Config: [configs/llama-server-262k-mtp.sh](configs/llama-server-262k-mtp.sh), with [configs/llama-server-slot-cache.sh](configs/llama-server-slot-cache.sh) for the prompt cache across restarts ([configs README](configs/README.md#prompt-cache-across-restarts))
- Patches: [configs/patches/](configs/patches/README.md)
- Benchmarks: [cold read by build](benchmarks/2026-09-26-cold-read-135k-by-build.md), [production build `20260925-f47a6a5f3`](benchmarks/2026-09-25-production-build.md) (fnbench), [agent session at 135K](benchmarks/2026-09-25-agent-session-135k.md), [grouped MoE kernel](benchmarks/2026-09-26-mul-mat-id-kernel-perf.md), [prefill op profile](benchmarks/2026-09-26-prefill-op-profile.md)

| Setting | Value |
|---|---|
| Engine | llama.cpp SYCL: PR #28243 at `6fcaa16` + #28931 + 0002 + 0003 + 0004 + #28796 + 0006-0017 (build `20260926-2b84213a4`, production since 2026-09-26; 0017 is diagnostics only) |
| Parallelism | `-sm layer` across 4 cards, `-ts 12,13,13,11`, `-fit off`; no tensor parallel |
| Context and batching | `-c 262144 -np 1 -b 2048 -ub 1536` (`-b 3072` tested on 2026-09-26: no gain) |
| KV cache | f16 K and V, allocated at load (about 6 GiB at 262K) |
| Speculative decoding | MTP draft, `--spec-draft-n-max 3`, draft on the last card. The draft attends only its output rows (patch 0008) and uses QSA with its own indexer (patch 0009, on by default; `LLAMA_MTP_QSA=0` gives the dense draft). `LLAMA_MTP_WINDOW` stays unset |
| Attention | `-fa on`, `GGML_SYCL_SPARSE_FA=1` (multi-token sparse FA), pooled QSA key cache on (patch 0007 default), QSA score expansion by broadcast (patch 0011, automatic when cells sit in position order) |
| MoE prompt path | grouped MoE XMX GEMM for the Q4_K, Q5_K, Q5_1 and Q8_0 experts (patches 0012-0015), on by default: `GGML_SYCL_XMX_GATHER_TYPES` unset means all bits set; `0` restores the per-expert loop |
| Prompt cache | `--slot-save-path` with save on stop and restore on start through [llama-server-slot-cache.sh](configs/llama-server-slot-cache.sh), only into an identical setup (signature file); patch 0010 saves the MTP draft's context too. `--cache-ram 24576` (`CACHE_RAM`; upstream default 8192 MiB). Restart only when the instance has served no requests |
| Sampling (server defaults) | `--temp 1.0 --top-p 0.95 --top-k 20`; every benchmark on this card is greedy |
| PLE table | CPU memory (`-ot per_layer_token_embd=CPU`, redundant but kept) |
| Graph mode | eager |
| Power and clocks | 230 W cap per card; GPU clock floor `min_freq = rp0` (2800 MHz) while serving ([`tools/gpu-clock-floor.sh`](../../tools/gpu-clock-floor.sh)) |
| Idle power | PCIe ASPM (Active State Power Management) L1 on the card links ([`tools/bmg-aspm-l1.sh`](../../tools/bmg-aspm-l1.sh)) |

## Headline numbers

Single stream, greedy. The 135K rows are one code corpus: the 134,862-token C++ context of [`tools/lcbench.py`](../../tools/lcbench.py); the build is named in each row. The short, ~10k, ~39k and 219K rows are [`tools/fnbench.py`](../../tools/fnbench.py) points (19, 9,749, 39,119 and about 219K prompt tokens; the long prompts are repetitive and flatter MTP) from build `20260925-f47a6a5f3`. They were not re-measured on the 2026-09-26 builds; the one fnbench run of `20260925-0f23f62c8` is in the [draft QSA experiment](experiments/2026-09-25-mtp-draft-qsa.md). Production samples at temperature 1.0 ([launch script](configs/llama-server-262k-mtp.sh)); decode speed and draft acceptance under that sampler were not measured. Follow-up-turn times are lcbench's per-turn prompt median: the upper median over all follow-up turns, where every third turn adds 358-416 tokens, so with 6 turns it is the slowest of the four ~60-token turns and with 10 turns the second slowest of seven ([`tools/lcbench.py`](../../tools/lcbench.py)).

| Metric | Value | Benchmark |
|---|---|---|
| Cold read of the 134,862-token lcbench context | about 284 s: 283.10 s server-side and 283.66 s client-side on `a890bf8b0` (the production code without the diagnostic patch 0017), 285.88 / 286.44 s on `2b84213a4` with `-b 3072`; about 475 tok/s. From 455.35 s on `993baf141` (2026-09-25), 382.49 s on `2c88bdf19` and 312.58-312.71 s after the grouped MoE GEMM. One read per build after the baseline | [cold read by build](benchmarks/2026-09-26-cold-read-135k-by-build.md), [raw progress](benchmarks/raw/2026-09-26-cold-read-progress.csv), [raw windows](benchmarks/raw/2026-09-26-cold-read-windows.csv) |
| Follow-up turn, ~60 new tokens on a cached 135K prefix | 0.76-0.77 s median (`85b26781b`, `a890bf8b0`); 0.92-0.96 s on the evening 2026-09-25 builds with the draft QSA; 1.04-1.32 s per turn on `20260925-f47a6a5f3` | [raw windows](benchmarks/raw/2026-09-26-cold-read-windows.csv); [agent session](benchmarks/2026-09-25-agent-session-135k.md) |
| Single-stream decode, 135K agent session | 43.7-48.7 tok/s (medians of 6-10 greedy turns per build, `2c88bdf19` to `a890bf8b0`; acceptance 63.8-72.2%); MTP step 67.1-68.4 ms. Decode follows acceptance, not the build | [raw windows](benchmarks/raw/2026-09-26-cold-read-windows.csv), [cold read by build](benchmarks/2026-09-26-cold-read-135k-by-build.md) |
| Service restart with a 135,760-token conversation in the slot | 88 s restart (model load included) + 3.92 s first turn, then 0.78-0.80 s per turn, instead of a cold re-read (403-472 s on the 2026-09-25 builds, about 284 s on the current code) | [slot save/restore](experiments/2026-09-25-slot-save-restore-and-ram-cache.md), [raw](benchmarks/raw/2026-09-26-slot-cache-timings.csv) |
| Single-stream decode, short prompt | 48.6-50.5 tok/s (3 sessions, `20260925-f47a6a5f3`) | [production build](benchmarks/2026-09-25-production-build.md) |
| Single-stream decode, ~10k context | 60.6-67.2 tok/s (`20260925-f47a6a5f3`) | [production build](benchmarks/2026-09-25-production-build.md) |
| Single-stream decode, ~39k context | 53.6-55.2 tok/s (`20260925-f47a6a5f3`) | [production build](benchmarks/2026-09-25-production-build.md) |
| Single-stream decode, 219K context | 40.6 tok/s (`20260925-f47a6a5f3`) | [production build](benchmarks/2026-09-25-production-build.md) |
| Prompt processing, 10k-39k | 568-645 tok/s (`20260925-f47a6a5f3`) | [production build](benchmarks/2026-09-25-production-build.md) |
| Prompt processing, 219K (cold) | 268.3 tok/s, 816.9 s (`20260925-f47a6a5f3`) | [production build](benchmarks/2026-09-25-production-build.md) |
| Aggregate at c8 / c16 | not measured: single slot | |
| Max context served | 262,144 tokens | [production build](benchmarks/2026-09-25-production-build.md) |
| Peak VRAM, busiest card, at 219K | 28,957 MiB (`20260925-f47a6a5f3`); 29,278 MiB in use after a 135K session on `a890bf8b0` (session log, not archived) | [production build](benchmarks/2026-09-25-production-build.md) |
| Idle power, 4 cards, model loaded | 185 W; 25-27 W with ASPM L1 | [finding](../../findings/bmg-aspm-l1-idle-power.md) |

The `-b 3072` read is not the production configuration; production keeps `-b 2048`, and the two reads agree within run-to-run noise ([`-b 3072`](experiments/2026-09-26-batch-3072.md)).

## Progress

| Stage | Date | Short | ~10k | ~39k | 219K | 135K | 135K cold read | Experiment |
|---|---|---|---|---|---|---|---|---|
| Baseline: unpatched, `-ub 1024 -ts 13,13,13,10`, MTP | 2026-09-24 | 33.8-36.9 | 43.3 | 33.4 | 16.1 | - | - | [baseline](experiments/2026-09-24-baseline-llamacpp-mtp-262k.md) |
| + fused `MUL_MAT_ID` for MTP verify, gettid cache, #28931 | 2026-09-24 | 42-44 | 52-54 | 44-45 | - | - | - | [fused MUL_MAT_ID](experiments/2026-09-24-fused-mul-mat-id-mtp-verify.md) |
| + GPU clock floor 2800 MHz | 2026-09-24 | 50 | 64-66 | 47-49 | - | - | - | [clock floor](experiments/2026-09-24-gpu-clock-floor.md) |
| + `-ub 1536 -ts 12,13,13,11` | 2026-09-25 | 50.3 | 63.6 | 52.6 | 18.4 | 22.3-23.3 | - | [micro-batch (ubatch) 1536](experiments/2026-09-25-ubatch-1536.md) |
| + multi-token sparse FA | 2026-09-25 | 49.3 | 63.1 | 49.7 | - | 31.3-34.5 | - | [sparse FA](experiments/2026-09-25-sparse-fa-multi-token.md) |
| + pooled QSA key cache (`20260925-f47a6a5f3`) | 2026-09-25 | 48.6-50.5 | 60.6-67.2 | 53.6-55.2 | 40.6 | 39.1-48.5 | 442.6 s (sandbox through 0007) | [pooled cache](experiments/2026-09-25-pooled-qsa-key-cache.md), [agent session](benchmarks/2026-09-25-agent-session-135k.md) |
| + MTP draft: output-rows-only attention and QSA with its own indexer (patches 0008-0009, `20260925-0f23f62c8`) | 2026-09-25 | 46.3 (1 run) | 59.4 | 54.9 | - | 41.1; paired A/B 43.4 against 41.9 with the dense draft | 436.80 s | [draft QSA](experiments/2026-09-25-mtp-draft-qsa.md), [rows only and window](experiments/2026-09-25-mtp-draft-rows-only-and-window.md) |
| + slot save/restore with the draft context (patch 0010, `20260925-993baf141`) | 2026-09-25 | - | - | - | - | 39.1 (2 turns) | 455.35 s; a restart now keeps the conversation: 88 s + 3.92 s first turn | [slot save/restore](experiments/2026-09-25-slot-save-restore-and-ram-cache.md) |
| + QSA score expansion by broadcast (patch 0011, `20260926-2c88bdf19`) | 2026-09-26 | - | - | - | - | 43.9 | 382.49 s | [broadcast](experiments/2026-09-26-qsa-score-expansion-broadcast.md) |
| + grouped MoE XMX GEMM (patches 0012-0015, `517eb833d` / `85b26781b`) | 2026-09-26 | - | - | - | - | 48.3 / 48.7 | 312.58 / 312.71 s | [grouped MoE GEMM](experiments/2026-09-26-grouped-moe-xmx-gemm.md) |
| + MTP `eh_proj` as one 2D product (patch 0016, `a890bf8b0`; production `20260926-2b84213a4` adds the diagnostic 0017) | 2026-09-26 | - | - | - | - | 43.7 | 283.66 s | [eh_proj](experiments/2026-09-26-mtp-eh-proj-2d-product.md) |

Decode in tok/s. The 135K column is the lcbench median over the follow-up turns of one greedy session per build (2-12 turns); it moves with draft acceptance, which differs with the generated text (63.1-72.2% on the 2026-09-25 evening and 2026-09-26 builds), not with the build. The cold-read column is lcbench's client-side first turn; after the pooled-cache row it is one read per build, so differences under about 3% are not established; the earlier builds read the same context in 403-472 s over ten sessions ([cold read by build](benchmarks/2026-09-26-cold-read-135k-by-build.md), [slot save/restore](experiments/2026-09-25-slot-save-restore-and-ram-cache.md)). Short prompt turns of about 60 tokens went from 1.24-1.27 s to 0.92 s with the draft QSA and to 0.77 s with patch 0015 ([cold read by build](benchmarks/2026-09-26-cold-read-135k-by-build.md)). A dash means not measured at that stage: short, ~10k, ~39k and 219K were not re-measured after `20260925-0f23f62c8`. The full log, including what was reverted, is in [experiments/README.md](experiments/README.md).

## What worked and what did not

- **Worked:**
  - MTP with an explicit split: +38% to +124% over MTP off at depth once `-ub 1024 -ts 13,13,13,10 -fit off` made room for the draft.
  - Fused `MUL_MAT_ID` for 2-8-token batches, shipped together with the `gettid` cache and #28931: +14% to +29% decode, the largest single gain at short and medium context.
  - GPU clock floor at 2800 MHz: +4% to +27% decode between range extremes (+8% to +23% between range midpoints), about 0 W at true idle.
  - Multi-token sparse flash attention: +41% to +55% decode at 135K.
  - Pooled QSA key cache: MTP step at 135K from 95.8 to about 73 ms; with sparse FA, 219K decode from 18.4 to 40.6 tok/s.
  - `-ub 1536`: +4.5% to +6% prefill at depth.
  - PCIe ASPM L1: idle power of the four cards from 185 W to 25-27 W, with no decode or prefill cost ([finding](../../findings/bmg-aspm-l1-idle-power.md)).
  - MTP draft attends only its output rows (patch 0008): exact by construction; MTP step at 135K from 73.4 to 72.7 ms, one session per arm ([experiment](experiments/2026-09-25-mtp-draft-rows-only-and-window.md)).
  - QSA in the MTP draft head with its own indexer (patch 0009): in a paired A/B at 135K (one session, 12 paired turns), decode +3.6% (43.4 against 41.9 tok/s), acceptance +1.9 points, step 1.2 ms shorter. Prompt turns of about 60 tokens fell from 1.24-1.27 s to 0.92 s, because the draft's per-turn checkpoints stopped copying its whole KV cache. Costs 96 MiB on the last card ([experiment](experiments/2026-09-25-mtp-draft-qsa.md)).
  - Slot save/restore across restarts, with the draft context (patch 0010) and a setup signature: 135,760 tokens saved in 1.5 s (3.7 GiB + 303 MiB for the draft, as `du -h` reports them); a restart costs 88 s plus a 3.92 s first turn instead of a 403-472 s cold re-read on the 2026-09-25 builds. `--cache-ram 24576` brought a displaced conversation of about 40K tokens back in 0.58 s (session log, not archived) ([experiment](experiments/2026-09-25-slot-save-restore-and-ram-cache.md), [finding](../../findings/llama-server-restart-drops-prefix-cache.md)).
  - QSA score expansion by broadcast (patch 0011): a graph change with no new kernel and no VRAM cost took the 135K cold read from 455 to 382 s (-16% against the median of 6 baseline reads; one read after the change, so about 5-17% given the baseline's spread, estimate). Teacher-forced KL (Kullback-Leibler divergence) stayed within the same-build noise floor ([experiment](experiments/2026-09-26-qsa-score-expansion-broadcast.md)).
  - Grouped MoE XMX GEMM, upstream #29245 plus our Q4_K, Q5_K, Q5_1 and Q8_0 decoders (patches 0012-0015): one expert matmul over a 1,536-token ubatch from about 11 ms to 3.8-4.4 ms (7.4 ms for Q8_0); cold read 382 to 313 s (-18%; one clean read per build); prompt turns 0.90 to 0.77 s with the in-place I/O of 0015. KL within noise ([experiment](experiments/2026-09-26-grouped-moe-xmx-gemm.md), [kernel benchmark](benchmarks/2026-09-26-mul-mat-id-kernel-perf.md), [finding: lazily reordered K-quant experts](../../findings/llama-cpp-sycl-kquant-experts-reordered-lazily.md)).
  - MTP `eh_proj` as one 2D product (patch 0016): a 3D input made SYCL run one product per token, 115 ms per 1,536-token ubatch. The reshape took the cold read from about 313 to 284 s (-4% to -10% against the baseline's three reads, one read of the new build) ([experiment](experiments/2026-09-26-mtp-eh-proj-2d-product.md), [finding](../../findings/llama-cpp-sycl-batched-mul-mat-loops-per-slice.md)).
- **Did not work:**
  - SYCL graphs: no effect, refused with more than one device.
  - Longer MTP drafts (n-max 4; n-max 6 with p-min 0.75): no consistent gain.
  - Pipeline-parallel prefill: no gain, +2.6 to +2.9 GiB VRAM per card.
  - Per-token fused `MUL_MAT_ID` for batches up to 512 tokens made prompt turns of about 400 tokens 16-33% slower.
  - `-ub 2048` with MTP at 262K: the last card overflowed and decode fell to 0.6 tok/s.
  - vLLM: upstream v0.30.0 raises `NotImplementedError` for Qwen4Exp on XPU; not pursued ([research](research/2026-09-25-vllm-xpu-deep-dive.md)).
  - A sliding window for the MTP draft (`LLAMA_MTP_WINDOW` 2048 and 8192): step 4.2-5.0 ms shorter, but acceptance 4.1-8.0 points lower; decode 41.9-43.1 against 42.3 tok/s, a wash. Left off ([experiment](experiments/2026-09-25-mtp-draft-rows-only-and-window.md)).
  - `-b 3072` (no 512-token remainder ubatches): cold read 286.44 against 283.66 s, no gain; reverted. That session's 62.4 tok/s and 93.1% acceptance are not comparable: the model repeated the same answer from turn 2 on ([experiment](experiments/2026-09-26-batch-3072.md)).
  - Not built: sparse attention for prompt batches over the union of a tile's selections. At 34K context, 16 consecutive prompt rows select 3.7x one row's 2,051 cells and 64 rows 7.7x. The session put the gain at about 3-5% of a cold read reusing the existing kernels and about 10% with a custom kernel (estimates), so it was not built ([research](research/2026-09-26-sparse-prompt-attention-tile-union.md)).

## Open questions

Resolved since 2026-09-25: the MTP draft layer no longer attends densely (QSA in the draft, [patch 0009](experiments/2026-09-25-mtp-draft-qsa.md)), and the grouped multi-token `MUL_MAT_ID` kernel is built ([patches 0012-0015](experiments/2026-09-26-grouped-moe-xmx-gemm.md)).

- [ ] How large is attention's share of a whole 135K cold read? Dense prompt attention was 32.8% of listed op time at about 100-120K context ([op profile](benchmarks/2026-09-26-prefill-op-profile.md)); the session estimated about 12% of the whole read (session estimate, UNVERIFIED; [tile union](research/2026-09-26-sparse-prompt-attention-tile-union.md)), but the per-segment read speed still falls to 0.45-0.64 of the first 30,720-token segment's speed by the last logged segment on every build ([cold read by build](benchmarks/2026-09-26-cold-read-135k-by-build.md)). The answer decides whether sparse prompt attention is worth building.
- [ ] Remaining prefill levers, all estimates: a custom sparse prompt attention kernel (about 10% of a cold read), more MoE kernel work (about 5-8%), shared routing and B pack between the gate and up calls (about 2%) ([grouped MoE GEMM](experiments/2026-09-26-grouped-moe-xmx-gemm.md), [tile union](research/2026-09-26-sparse-prompt-attention-tile-union.md)).
- [ ] Upstream [#29245](https://github.com/ggml-org/llama.cpp/pull/29245) was still open on 2026-09-26 (head `950f1c4aa`); our K-quant, Q5_1 and Q8_0 decoders sit on top of it and are local only.
- [ ] llama.cpp's QSA top-k over expanded cells can attend up to 3 cells of a 513th block, unlike the reference. Fix it (changes output, matches the reference) or leave it?
- [ ] vLLM route: a community fork reports 52.5 tok/s single stream and 626-634 tok/s at 16 streams on another 4x B70 host (UNVERIFIED), but runs dense attention without prefix caching, so every agent turn would re-read the whole context. Revisit when Intel llm-scaler PR #660 or upstream XPU support lands.
- [ ] No long-context quality evaluation of the abliterated checkpoint beyond the refusal smoke test and teacher-forced KL checks between builds.
- [ ] Newer card firmware (GSC 31.1062) is on LVFS and not applied; it mentions better low-power state entry with no display attached.

From the [2026-09-26 review of what was missed](research/2026-09-26-review-what-was-missed.md), highest value first:

- [ ] Reasoning-token share: how much of a real agent turn's decode is reasoning, and whether a thinking policy or budget pays. Not measured ([review](research/2026-09-26-review-what-was-missed.md)).
- [ ] Step-time jump near 106-112K: the decode step grew by 20-28 ms there on the 2026-09-25 builds ([QSA design note](research/2026-09-25-long-context-qsa-design.md), from session notes); the agent works past that depth and no later build was measured across it ([review](research/2026-09-26-review-what-was-missed.md)).
- [ ] Re-sweep MTP draft shaping at 135K (n-max, p-min, MTP off) with greedy and with the production sampler at temperature 1.0; every draft number here is greedy. Replaces the old plan to rerun the draft-length sweep on the agent workload ([review](research/2026-09-26-review-what-was-missed.md)).
- [ ] Decompose the decode step of the current builds (67.1-68.4 ms at 135K, [raw windows](benchmarks/raw/2026-09-26-cold-read-windows.csv)) into launches, host time and per-op GPU time before choosing decode levers ([review](research/2026-09-26-review-what-was-missed.md)).
- [ ] Outage causes and the slot signature: key the signature on the state layout instead of the build path, save periodically because a crash cannot save, and check client timeout and cancel behaviour during a cold re-read ([review](research/2026-09-26-review-what-was-missed.md), [finding](../../findings/llama-server-restart-drops-prefix-cache.md)).
- [ ] Serialize the pooled QSA key rows with the saved slot: the first turn after a restore takes 3.92 s because the pooled cache re-pools the whole context; later turns take 0.78-0.80 s ([review](research/2026-09-26-review-what-was-missed.md), [slot save/restore](experiments/2026-09-25-slot-save-restore-and-ram-cache.md)).
- [ ] Non-model time per turn: how much of the 0.76 s short turn ([raw windows](benchmarks/raw/2026-09-26-cold-read-windows.csv)) falls outside the server's prompt time, measured on the chat-completions path ([review](research/2026-09-26-review-what-was-missed.md)).
- [ ] Image-path hazard: one image in a conversation takes the pooled key cache and the draft QSA off their fast paths for the affected positions and blocks slot save; how often the client sends images is unknown ([review](research/2026-09-26-review-what-was-missed.md), [patch scope](configs/patches/README.md#scope-and-limits)).
- [ ] Prefill host gap: part of a cold read's time lies outside the profiled SYCL ops; profile the host side during a read (lazy page faults on the CPU-resident PLE table are one suspect, UNVERIFIED) ([review](research/2026-09-26-review-what-was-missed.md)).

## Index

- [Research notes](research/README.md)
- [Experiments log](experiments/README.md)
- [Benchmarks](benchmarks/README.md)
- [Configs](configs/README.md)
