# Qwen3.8-27B

| | |
|---|---|
| **Status** | active tuning |
| **Host** | [dual-b70-5800x](../../hardware/hosts/dual-b70-5800x.md), 2x Arc Pro B70 |
| **Engine** | [vLLM XPU](../../engines/vllm-xpu/) 0.28.0 |
| **Last updated** | 2026-09-12 |

## Architecture

| Property | Value |
|---|---|
| Parameters | 27B dense |
| Layers and layer types | 64 layers: 48 Gated DeltaNet (linear attention) + 16 full attention, 3:1 |
| Hidden size | 5120 |
| Vocabulary | 248,320 |
| Context served | 262,144 tokens |
| Native speculative head (MTP) | yes, 1 layer |

Why it matters on Arc: the Gated DeltaNet layers need XPU GDN kernels that are still being patched, and the hybrid layout is what exposes the prefix-cache plus MTP corruption bugs. The large vocabulary makes the LM head a real share of each draft pass, which is why quantizing the draft-side head pays off.

## Checkpoints

| Checkpoint | Quant method | Bits / group | XPU kernel path | Status |
|---|---|---|---|---|
| `Qwen3.8-27B-Uncensored-GPTQ-Int4-sym-G128-MTP-BF16` | GPTQ, symmetric | INT4 / 128, BF16 MTP head | `XPUwNa16LinearKernel` | in production |
| `SergiioB/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16` | GPTQ, symmetric | INT4 / 128, BF16 MTP head | `XPUwNa16LinearKernel` | community-measured reference |
| `Marcin116/...-Uncensored-W4A16-AutoRound-embed-int4` | AutoRound | INT4 | `XPUwNa16LinearKernel` | candidate for accuracy |
| FP8 W8A16 | FP8 | 8 | Xe2 small-M kernel | rejected: bandwidth-bound, slower than INT4 on TP2 |
| INT8 weight-only | | 8 | none on XPU | not available |

## Current best config

- Config: [configs/production-mtp2-compose.yml](configs/production-mtp2-compose.yml)
- Settings that matter: TP2, GPTQ INT4 G128 with BF16 MTP head, MTP depth 2, XPU graphs, per-worker Level Zero affinity, `CCL_TOPO_P2P_ACCESS=1`, SYCL simple-path collectives, FP8 KV, block size 32, prefix caching on, 7 startup patches.

## Headline numbers

| Metric | Value |
|---|---|
| Single-stream decode | 78 to 84 tok/s |
| Aggregate at c8 | 291.6 tok/s |
| Aggregate at c16 | 405 to 442 tok/s |
| 150K-token cold TTFT | about 183 s |
| Max context served | 262,144 tokens |

## What worked and what did not

- **Worked:** per-worker `ZE_AFFINITY_MASK` for TP2 graphs (about 2x decode), enabling PCIe P2P (c16 356 to 442 tok/s, prefill 213 s to 183 s), SYCL simple-path collectives, native MTP with a BF16 draft head.
- **Did not work:** DFlash and DFlash2 external drafters (0% acceptance), FP8 weights (bandwidth-bound).

## Next steps

From the [performance gains report](research/2026-09-12-performance-gains-report.md). The realistic stacked target is 105 to 120 tok/s single stream.

| Lever | Expected single-stream effect | Status |
|---|---|---|
| Power cap 150 W to 230 W per card | +18% to +30% if still at 150 W | planned |
| MTP depth 2 to 4 | +12% to +27% | planned |
| `CCL_SYCL_ALLREDUCE_LL=twoshots` | lower per-collective latency | planned |
| Whole-graph compile, `FULL_DECODE_ONLY` capture sizes to 320, block size 64 | concurrency scaling, avoids MTP phantom-token bug | planned |
| INT4 draft LM head (TP2-safe lab implementation) | +24% on top of MTP4 | planned |
| vLLM 0.29.0 upgrade | small; drops the pointer-overflow patch | planned |
| compute-runtime 26.31 or master | 0% to a few %, better P2P stability | planned |
| Prefix-cache plus MTP correctness fixes | correctness only | planned |
| DFlash2 retest on vLLM main at or after `d61b6e18` | unknown | blocked |

## Open questions

- [ ] What is MTP acceptance on this uncensored fine-tune at depth 4? Drop to depth 3 if under about 85%.
- [ ] Which host latency class is dual-b70-5800x in? Run the host probe and allreduce census.
- [ ] Does the prefix-cache plus MTP combination corrupt output on this config?
- [ ] Is 140 tok/s reachable outside code-heavy, short-context prompts?

## Index

- [Research notes](research/README.md)
- [Experiments log](experiments/README.md)
- [Benchmarks](benchmarks/README.md)
- [Configs](configs/README.md)

## Note, 2026-09-23: what the production service actually ran

An audit on the upgraded box, [quad-b70-5800x-pex88096](../../hardware/hosts/quad-b70-5800x-pex88096.md), found the production vLLM service serving a locally quantized DavidAU Qwen3.8-27B merge (GPTQ INT4, symmetric, group 128) with MTP depth 3, not `Qwen3.8-27B-Uncensored-GPTQ-Int4-sym-G128-MTP-BF16` with MTP depth 2. The checkpoint is described in the [TP4 benchmark](benchmarks/2026-09-23-tp4-quad-b70.md). When the service changed was not recorded. The headline numbers above come from the tuning blueprint and are left as recorded.

Two more corrections from the same audit:

- Prefix caching is on by default in vLLM 0.28.0, also for this hybrid model ([note in the finding](../../findings/prefix-cache-mtp-corruption-hybrid.md)).
- The per-worker affinity patch does not engage in vLLM 0.28.0 as launched ([note in the finding](../../findings/tp-xpu-graphs-need-per-worker-affinity.md)).
