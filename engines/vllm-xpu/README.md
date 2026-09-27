# vLLM on XPU

## Images

| Image | vLLM | Notes |
|---|---|---|
| `vllm/vllm-openai-xpu:v0.28.0` | 0.28.0 | current production base for [Qwen3.8-27B](../../models/qwen3.8-27b/) |
| `vllm/vllm-openai-xpu:v0.29.0` | 0.29.0 | released 2026-09-08; upgrade candidate |
| `vllm/vllm-openai-xpu:v0.30.0` | 0.30.0 | released 2026-09-22. torch 2.13.0+xpu, vllm-xpu-kernels 0.1.14.1, oneCCL 2022.0.0 (read from the image). Has the Gated DeltaNet fixes vllm-xpu-kernels #537 and #544 and pipeline-parallel support for the MTP draft. Refuses Qwen4Exp on XPU (`vllm/models/qwen4_exp/__init__.py:30-31`). Default model runner on XPU is V2 |
| `vllm/vllm-openai-xpu:nightly-7f1a5398e9610d96c473931a26c0e12bbe0d0423` | 0.30.1rc1.dev48+g7f1a5398e | checked 2026-09-24. torch 2.14.0+xpu, vllm-xpu-kernels 0.1.15.4, oneCCL 2022.1.2, oneAPI runtime 2026.1.1. Same Qwen4Exp refusal at `__init__.py:30-31` |
| `vllm/vllm-openai-xpu:nightly` (tag `nightly-ddd6fbca…`) | not read | pushed 2026-09-26T06:31Z, after `nightly-29468dde…` (2026-09-25); not pulled. The gate file has had one commit ever (`e126687a9a`, 2026-08-31), so this image still refuses Qwen4Exp (inferred) ([Docker Hub tags](https://hub.docker.com/r/vllm/vllm-openai-xpu/tags), checked 2026-09-26) |
| community lab images (steveseguin) | 0.27.2rc1 + rebuilt vllm-xpu-kernels and oneCCL | reference for the 112.9 tok/s dual-B70 result |

## Quantization on XPU

| Format | Kernel path | Decode speed on B70 | Notes |
|---|---|---|---|
| GPTQ INT4 (`uint4b8`) | `XPUwNa16LinearKernel` (oneDNN `int4_gemm_w4a16`) | fastest measured | the default choice |
| AutoRound INT4 (GPTQ packing) | `XPUwNa16LinearKernel` | same as GPTQ | equal or better logprob parity in community tests |
| AutoRound / INC with `VLLM_XPU_INC_WNA16_BACKEND=w4a8` | W4A8 | prefill only, at 512 or more tokens per call | 0.29.0 and later |
| FP8 W8A16 | Xe2 small-M kernel (rebuilt package) | slower than INT4 on TP2 | |
| INT8 weight-only | none | not available on XPU | |
| AWQ, compressed-tensors, GGUF | varies | unproven or strips MTP on this stack | |

## Features

| Feature | Status | Notes |
|---|---|---|
| Tensor parallel | works | needs per-worker `ZE_AFFINITY_MASK` for graph capture |
| Tensor parallel 4 on four B70s behind a PCIe switch | works | Qwen3.8-27B GPTQ-INT4, MTP 3: 129.1 tok/s single-stream, 969.0 aggregate at 16 streams on v0.28.0 ([benchmark](../../models/qwen3.8-27b/benchmarks/2026-09-23-tp4-quad-b70.md)). Set `ZE_AFFINITY_MASK=0,1,2,3`, size the KV cache with `--kv-cache-memory-bytes`, and watch driver-held host RAM ([finding](../../findings/xe-vram-overcommit-spills-into-host-ram.md)) |
| Pipeline parallel with MTP | fails on 0.28.0 | the MTP draft `Qwen3_5MTP` lacks `SupportsPP`; vLLM #46994 adds it in 0.30.0, untested on XPU |
| Qwen3.8-Flash-Next (Qwen4Exp) | blocked on XPU (re-checked 2026-09-26) | upstream raises `NotImplementedError` in 0.30.0, nightly and main; community ports and llm-scaler #660 reviewed in the [deep dive](../../models/qwen3.8-flash-next/research/2026-09-25-vllm-xpu-deep-dive.md) and the [community survey](../../models/qwen3.8-flash-next/research/2026-09-26-flash-next-on-vllm-intel-community.md); status below |
| XPU graphs (`VLLM_XPU_ENABLE_XPU_GRAPH=1`) | works | doubles decode throughput versus eager. On by default on `main` since [#51600](https://github.com/vllm-project/vllm/pull/51600) (merged 2026-09-24) |
| Native MTP speculative decoding | works | depth 4 is the single-user sweet spot on Qwen3.8-27B |
| DFlash / DFlash2 drafters | broken on released versions | non-causal mask dropped by the XPU attention wrapper; RoPE layout fix only on main |
| Prefix caching with MTP on hybrid models | silent corruption risk | three open upstream issues: vllm#53919, #48375, #53505. Prefix caching is **on by default** in 0.28.0, also for hybrid models: with no prefix-caching flag, our Qwen3.8-27B server logged `enable_prefix_caching=True` and `Mamba cache mode is set to 'align' for Qwen3_5ForConditionalGeneration by default when prefix caching is enabled`. Pass `--no-enable-prefix-caching` to turn it off ([finding](../../findings/prefix-cache-mtp-corruption-hybrid.md)) |
| Sleep mode with graphs | needs patch | graphs must be released before sleep |

## Qwen3.8-Flash-Next status (2026-09-26)

Upstream vLLM still refuses the model on XPU; people who run it on Arc use forks or private builds. For our single long-context agent session the verdict is unchanged: stay on llama.cpp (the bar an engine must clear is in [engines/README.md](../README.md)). Third-party numbers are UNVERIFIED here; the [community survey](../../models/qwen3.8-flash-next/research/2026-09-26-flash-next-on-vllm-intel-community.md) has every source.

| Item | State on 2026-09-26 | Source |
|---|---|---|
| Upstream gate | `NotImplementedError` at `vllm/models/qwen4_exp/__init__.py:30-31` on `main` `379e9a1ea8` (2026-09-26T16:28Z) and v0.30.0; nightly `ddd6fbca` not pulled (the gate is unchanged in source) | [file on main](https://github.com/vllm-project/vllm/blob/main/vllm/models/qwen4_exp/__init__.py) |
| XPU enablement PRs | [#57535](https://github.com/vllm-project/vllm/pull/57535) (skips the Qwen4Exp registry test on XPU instead of lifting the gate) and [#55068](https://github.com/vllm-project/vllm/pull/55068) (XPU model code, last commit 2026-09-04) are drafts | PRs |
| Dense port (devan-carlin, packaged by Ryan Purdy's kit) | 52.5-54.2 tok/s single stream and 634 tok/s at 16 streams on 4x Arc Pro B70. Dense attention instead of the trained QSA (Qwen sparse attention); MTP (multi-token prediction) off; prefix caching on by default, hit correctness untested | [kit](https://github.com/imryanpurdy/Qwen3.8-Flash-Next-4x-Intel-B70s), [Hugging Face discussion #3](https://huggingface.co/Intel/Qwen3.8-Flash-Next-W4A16-AutoRound/discussions/3) |
| steveseguin route: upstream `amd/` model with the `nvidia/` host PLE (per-layer embedding) layer | sparse QSA and MTP1 at 46.854 tok/s on 4x B70, 4,352-token capacity, prefix caching off | [results README](https://github.com/steveseguin/b70-optimization-lab/blob/main/results/qwen38-flash-next-fp8-b70/README.md), [patch 0004](https://github.com/steveseguin/b70-optimization-lab/blob/main/patches/qwen38-flash-next-fp8-b70/vllm/0004-Enable-Qwen4Exp-model-dispatch-on-XPU.patch) |
| PLE storage | [#57497](https://github.com/vllm-project/vllm/pull/57497) merged 2026-09-25: ROCm keeps the PLE table in pinned host memory behind a UVA (unified virtual addressing) view; XPU still excluded. [#58815](https://github.com/vllm-project/vllm/pull/58815) (`host_file_gather`) opened 2026-09-26 and is CUDA-gated, but TSUMUGI-XE ran it on 2x B70 with local patches at PP=2 (pipeline parallel): 38.1 ms per token against 38.3 with [#54129](https://github.com/vllm-project/vllm/pull/54129), which closed unmerged the same day | PRs, [#58815 comment](https://github.com/vllm-project/vllm/pull/58815#issuecomment-5845125914) |
| Platform | [#56013](https://github.com/vllm-project/vllm/pull/56013) moved XPU to PyTorch 2.14 (merged 2026-09-23; the dense port and Ryan's kit run torch 2.13). [#57291](https://github.com/vllm-project/vllm/pull/57291) (vllm-xpu-kernels 0.1.15.1 pin) is open. vllm-xpu-kernels [v0.1.15](https://github.com/vllm-project/vllm-xpu-kernels/releases/tag/v0.1.15) (2026-09-22) has the expert-parallel negative-expert-id fix [#578](https://github.com/vllm-project/vllm-xpu-kernels/pull/578), which 0.1.12-0.1.14.1 lack. [#58415](https://github.com/vllm-project/vllm/pull/58415) (reset oneCCL's collective chain after XPU graph capture) is open | PRs, release |
| Model-code issues | [#58688](https://github.com/vllm-project/vllm/issues/58688): the `amd/` PLE could not load an FP8 PLE table on v0.29.0 and v0.30.0 (disputed for `main`). [#56457](https://github.com/vllm-project/vllm/issues/56457): the QSA prefill indexer's buffer grows per chunk on the `nvidia/` path only | issues |

## Patch catalog

Patches currently applied in production and their upstream status. The patch files live in each model's `configs/patches/`.

| Patch | Purpose | Upstream status |
|---|---|---|
| `patch_mtp_nightly.py` | BF16 MTP draft head with a quantized base | vllm#47828 open |
| `patch_mtp_boundary.py` | off-by-one in speculative verification at context boundaries | not upstream |
| `patch_gdn_mixed_split_v5.py` | Gated DeltaNet mixed prefill/decode split on Battlemage | fixed upstream in vllm-xpu-kernels 0.1.14.1 (#537) per the 2026-09-25 deep dive (see the v0.30.0 image row above); patch still applied on 0.28.0 |
| `patch_worker_affinity.py` | per-rank `ZE_AFFINITY_MASK` at worker spawn | not upstream |
| `patch_mtp_ptr_wrap.py` | Mamba state pointer overflow | fixed in 0.29.0 (`f936a267`, #48109) |
| `patch_sleep_graphs_v5.py` | release and recapture graphs around sleep mode | not upstream |
| `patch_mem_report2.py` | memory diagnostics RPC | local tooling only |
