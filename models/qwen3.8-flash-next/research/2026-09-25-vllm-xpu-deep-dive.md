# Can vLLM serve Flash-Next on four B70s? Routes, checkpoints and risks

| | |
|---|---|
| **Date read** | 2026-09-25 |
| **Source** | vLLM v0.30.0 and nightly source and images; community ports ([devan-carlin/vllm](https://github.com/devan-carlin/vllm/tree/xpu-qwen4exp) branch `xpu-qwen4exp`, [devan-carlin/electric-sheep](https://github.com/devan-carlin/electric-sheep), [imryanpurdy/Qwen3.8-Flash-Next-4x-Intel-B70s](https://github.com/imryanpurdy/Qwen3.8-Flash-Next-4x-Intel-B70s), [steveseguin/b70-optimization-lab](https://github.com/steveseguin/b70-optimization-lab)); Intel [llm-scaler PR #660](https://github.com/intel/llm-scaler/pull/660); vLLM [PR #55068](https://github.com/vllm-project/vllm/pull/55068); Hugging Face checkpoint headers |
| **Author / org** | this repository (AI-assisted review, then an adversarial critique that corrected it) |
| **Type** | upstream source review and community labs |
| **Applies to** | Qwen3.8-Flash-Next (`Qwen4Exp`) on vLLM XPU, 4x Arc Pro B70, host [quad-b70-5800x-pex88096](../../../hardware/hosts/quad-b70-5800x-pex88096.md) |

## Summary

No vLLM build runs Qwen4Exp on XPU as of 2026-09-25. Upstream v0.30.0, the nightly `7f1a5398` image and main all raise `NotImplementedError("Qwen4Exp currently supports CUDA and ROCm only")` at [`vllm/models/qwen4_exp/__init__.py:30-31`](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/models/qwen4_exp/__init__.py#L30-L31). The fastest community port is a dense-attention fork that measured 52.5 tok/s single-stream on another 4x B70 host. It runs without prefix caching, so every turn of a long agent session would re-read the whole context. **Verdict: not pursued.** Our llama.cpp build already decodes at about 50 tok/s short and reuses the cached prefix on every turn.

Nothing in this note ran on our GPUs. Every performance number below is from someone else's host and is UNVERIFIED here.

## Routes

| Route | Attention | PLE (per-layer embedding n-gram table) | MTP | Reported performance | Main blockers |
|---|---|---|---|---|---|
| **A.** [devan-carlin fork](https://github.com/devan-carlin/vllm/tree/xpu-qwen4exp) `xpu-qwen4exp@a69fba21` with a community operations kit | Dense. The QSA (Qwen sparse attention) indexer weights are skipped | memory-mapped from host RAM, CPU gather. The fork ships a 95.37 GiB BF16 table | Drafter exists but was a net loss: 44.2 vs 53 tok/s at 34% acceptance ([electric-sheep](https://github.com/devan-carlin/electric-sheep) operations doc) | 52.5 tok/s single-stream, 338 at 8 streams, 626-634 at 16; about 2.78K tok/s prefill at 98K and 1.70K at 250K; 262,144 context with a 250,700-token needle pass ([kit repo](https://github.com/imryanpurdy/Qwen3.8-Flash-Next-4x-Intel-B70s), commit `743ecc5e`, kernel 6.17, GuC 70.65) | **No prefix caching** (PLE state lives outside vLLM's cache manager). Different model behaviour above about 2K tokens (dense instead of trained sparse attention). Base is a 2026-08-07 ancestor of v0.28.0 |
| **B.** Upstream `amd/` Triton path on the nightly image | Sparse QSA | `amd/` keeps PLE on the GPU in BF16 (`amd/ple_layer.py:227-232`), which does not fit; a host path must be ported | upstream `Qwen4ExpMTP` | Unmeasured upstream. Older lab builds of the same route on B70: 27.2 single / 299 at 16 streams, 98K ceiling (kit repo, legacy line); 46.85 single-sequence with FP8 + MTP1 at ≤4K ([steveseguin lab](https://github.com/steveseguin/b70-optimization-lab)) | Gate edit, five `.is_cuda` guards in `amd/ops/qsa.py`, a host sync in `amd/indexer_qsa.py:204`, host PLE path, Model Runner V2 graph bring-up on an XPU hybrid model. 4-15 days of work (estimate) |
| **C.** [Intel llm-scaler PR #660](https://github.com/intel/llm-scaler/pull/660) | Native ESIMD (explicit SIMD) sparse attention | pinned host USM lookup | MTP K4 | Commit messages only, internal container: TP4 INT4 eager 47.48 to 51.76 tok/s; MTP K4 57.74 to 63.81 tok/s; a 254,986-token request on TP8 FP8 | Kernels only; the vLLM integration is unpublished |
| **D.** vLLM [PR #55068](https://github.com/vllm-project/vllm/pull/55068) "add xpu support for qwen3.8-next" | Sparse via `deepklox` | GPU-resident FP8 | yes | none posted | Depends on [intel/DeepKLOX](https://github.com/intel/DeepKLOX), which held only boilerplate files and has no PyPI package (checked 2026-09-25) |

Every route needs all four GPUs.

### Correction: route A makes agent sessions much slower

The first draft of this analysis compared route A with llama.cpp on cold prompts only ("4-6x faster prefill"). The critique showed that is the wrong comparison for our workload.

- Our real workload is one coding-agent session at 126K-140K tokens, 23-555 new tokens per turn in 1-3.5 s, 19-33 tok/s (two readings of the 2026-09-25 journal). Every turn reuses the cached prefix and prefills only the new tokens ([agent-session benchmark](../benchmarks/2026-09-25-agent-session-135k.md)).
- Route A turns prefix caching off, so every turn would re-prefill about 130K tokens: roughly 50-75 s to first token at 1.7-2.8K tok/s (estimate), against 1-3 s today.
- For this workload, route A is 20-50x worse per-turn latency, and its concurrency advantage goes unused.
- The fix would be to make the fork's PLE conv state and n-gram history cache-managed, as upstream does (`amd/ple_layer.py:507-510`, SHORT_CONV MambaSpec), then use prefix caching in align mode. Effort unknown.

Other corrections from the critique:

- No source backs "much faster decode past 100K" for route A. The community kit published short-prompt decode and long-prompt time to first token only.
- The host-RAM budget came from our Qwen3.8-27B runs with `--enable-sleep-mode` (the xpumem allocator), which route A's launch line does not use. The fork does contain `vllm/device_allocator/xpumem.py`, so the A/B is possible on it.
- The fork's PLE runs on the CPU in Python per token, repeated in all four ranks. Our host has 16 threads against the reference rig's 12-core Threadripper PRO.
- The fork base likely predates fixes for four vLLM security advisories (GHSA-v5gm-qgmv-gc6c, GHSA-wpww-v874-ph2p, GHSA-85xf-c7hm-whqw, GHSA-p6g9-7v3x-m8mv; version ranges only, fork code not diffed).
- Vision (`--mmproj` in our llama.cpp config) is untested on the fork's XPU path, and `--language-model-only` would drop it.

## Checkpoint

The candidate checkpoint was [densohax/Qwen3.8-Flash-Next-heretic2-W4A16-FP8PLE](https://huggingface.co/densohax/Qwen3.8-Flash-Next-heretic2-W4A16-FP8PLE) (read from the Hugging Face API and safetensors headers on 2026-09-25):

| Item | Value |
|---|---|
| Size | 120.07 GiB (129.0 GB download) |
| Routed experts | INT4 symmetric group-128 AutoRound, 58.45 GiB |
| PLE table | FP8 E4M3, 47.68 GiB, one global scale |
| Rest of the trunk | BF16 |
| MTP draft | only in `runtime/mtp-int4-g32`; the stock head |
| Uncensoring | quantized from [trohrbaugh/Qwen3.8-Flash-Next-heretic-2](https://huggingface.co/trohrbaugh/Qwen3.8-Flash-Next-heretic-2) (0/100 refusals, KL 0.0818 on the BF16 model per its card); not re-measured after quantization |

Checkpoints with a 95.37 GiB BF16 PLE table cannot stay resident in 121 GiB of host RAM next to everything else, so they were rejected. The fork needs two small patches for this checkpoint: dequantize FP8 PLE rows in the gather, and memory-map the safetensors shards instead of the table-prep script, which allocates the whole table in RAM.

## Memory budget (estimates)

**VRAM per card**, TP4 with expert parallelism, MTP off, language model only. vLLM sees 30.3 GiB per card (measured on our Qwen3.8-27B TP4 run).

| Item | GiB per card |
|---|---|
| Routed experts, 128 of 512 per card | 14.61 |
| Hyper-connections / Gated DeltaNet / embed+lm_head / attention / shared expert+router+PLE projection | 1.19 / 0.97 / 0.59 / 0.29 / 0.29 |
| **Weights** | **about 18.0** |
| Startup plus non-torch memory | about 4.0 (measured 1.6 + 2.43 on the 27B at TP4) |
| Activations / XPU graphs / margin | 0.8-1.5 / 1.5-3.2 / 1.0 |
| **KV cache left** | **2.6-5.0** |

Dense attention allows an fp8 KV cache at 6 KiB per token per card (12 layers x 1 KV head x 256 x 2 bytes), so 2.6-5.0 GiB holds about 440K-870K tokens. The community kit measured an 886,567-token pool. Size the KV cache with `--kv-cache-memory-bytes`, not `--gpu-memory-utilization`: on the 27B, vLLM overshot to 18.7 GiB of KV and 31.19 GiB used per card, and allocations past about 31.2 GiB spill into host RAM ([finding](../../../findings/xe-vram-overcommit-spills-into-host-ram.md)).

**Host RAM** (MemTotal 121.46 GiB):

| Item | GiB |
|---|---|
| Driver-held RAM | measured 15-23 on the 27B TP4 with the sleep-mode allocator; budget 15-30 (estimate, see correction above) |
| Anonymous memory | measured 12-19 on the 27B; budget 18-25 |
| FP8 PLE page cache | up to 47.7, reclaimable |
| Ceiling for non-movable allocations | about 104, because 17.31 GiB is kernel CMA scratch that TTM (the kernel's GPU memory manager) cannot use |

Rejected host layouts: a pinned FP8 table (about 64 GiB once the allocator rounds each pinned block up to a power of two), a BF16 page-cached table (does not stay resident), and PLE on the GPUs (FP8 adds 11.92 GiB per card, leaving no KV room).

## Claims to verify

| Claim | Reported number | Their setup | Status |
|---|---|---|---|
| Upstream refuses Qwen4Exp on XPU | `NotImplementedError` at `__init__.py:30-31` | v0.30.0, nightly `7f1a5398`, main | upstream-documented (read in source and in the images) |
| Dense fork single-stream decode on 4x B70 | 52.5 tok/s (fork author: 53.4) | community host, kernel 6.17, GuC 70.65 | UNVERIFIED |
| Dense fork aggregate | 626-634 tok/s at 16 streams | same | UNVERIFIED |
| Dense fork prefill | about 2.78K tok/s at 98K, 1.70K at 250K | same | UNVERIFIED |
| Level Zero wedge every 2-6 h on the fork | – | community operators | UNVERIFIED |
| MoE top-10 kernel faults on GuC 70.58 | – | [vllm-xpu-kernels #559](https://github.com/vllm-project/vllm-xpu-kernels/issues/559) (open) | UNVERIFIED; our host runs GuC 70.58 |
| llm-scaler #660 decode | 51.76 tok/s eager, 63.81 with MTP K4 | Intel internal container, TP4 INT4 | UNVERIFIED (commit messages) |
| Route B lab builds | 27.2 single, 299 at 16 streams | community legacy line | UNVERIFIED |

## Relevance

- vLLM would win on concurrency (hundreds of tok/s aggregate) and cold prefill. Neither matters for one agent session that reuses its prefix.
- Single-stream decode at short context comes out about even with our llama.cpp build (52.5 vs about 50 tok/s).
- Route A changes the model's attention above about 2K tokens, so its output is not comparable to the trained model without a fidelity check.

## Actions

- [ ] Watch llm-scaler #660 for a published vLLM integration
- [ ] Watch vLLM for a lifted XPU gate in `vllm/models/qwen4_exp/__init__.py`
- [ ] If revisited: boot route A with `--load-format dummy` first to test platform, memory and stability before a 129 GB download, and replay a real 130K agent transcript turn by turn against llama.cpp

## Update 2026-09-26

Route B's host-PLE blocker is partly cleared upstream; the verdict is unchanged. vLLM [#57497](https://github.com/vllm-project/vllm/pull/57497), merged 2026-09-25, keeps the PLE table in pinned host memory behind a UVA (unified virtual addressing) view on ROCm, and moves the storage classes to `qwen4_exp/common/`. On `main` (read 2026-09-26), `amd/ple_layer.py:250-254` picks `Qwen4ExpPLEPinnedHostEmbedding` when `engram_config.cpu_offload` is set, so the `amd/ple_layer.py:227-232` reference in the Routes table is stale. XPU is still excluded. The gate at [`__init__.py:30-31`](https://github.com/vllm-project/vllm/blob/main/vllm/models/qwen4_exp/__init__.py) still raises on `main` `379e9a1ea8`, and `amd/ops/qsa.py` still has five `.is_cuda` guards (lines 604, 680, 829, 970, 1015); `amd/indexer_qsa.py:204` was not re-read. The steveseguin lab's [patch 0004](https://github.com/steveseguin/b70-optimization-lab/blob/main/patches/qwen38-flash-next-fp8-b70/vllm/0004-Enable-Qwen4Exp-model-dispatch-on-XPU.patch) (2026-08-26) already takes this route: it sends XPU to the `amd/` model with the `nvidia/` host-PLE layer. Its [results](https://github.com/steveseguin/b70-optimization-lab/blob/main/results/qwen38-flash-next-fp8-b70/README.md) show sparse QSA, MTP1 and UVA host PLE running together at 46.854 tok/s, but only at a 4,352-token capacity (33,280 in a depth ladder) with prefix caching off (UNVERIFIED here). The [community survey](2026-09-26-flash-next-on-vllm-intel-community.md) has the sources and the other corrections to this note: the fork runs with prefix caching on by default, with hit correctness untested; a BF16 PLE table need not stay resident for decode; plugins cannot bypass the gate; `--engram-config` replaces `VLLM_PLE_CPU_OFFLOAD`; the newest nightly image is `ddd6fbca`; and the fork also runs on a 128 GB host with 64 GiB of swap.
