# vLLM on XPU

## Images

| Image | vLLM | Notes |
|---|---|---|
| `vllm/vllm-openai-xpu:v0.28.0` | 0.28.0 | current production base for [Qwen3.8-27B](../../models/qwen3.8-27b/) |
| `vllm/vllm-openai-xpu:v0.29.0` | 0.29.0 | released 2026-09-08; upgrade candidate |
| `vllm/vllm-openai-xpu:v0.30.0` | 0.30.0 | released 2026-09-22. torch 2.13.0+xpu, vllm-xpu-kernels 0.1.14.1, oneCCL 2022.0.0 (read from the image). Has the Gated DeltaNet fixes vllm-xpu-kernels #537 and #544 and pipeline-parallel support for the MTP draft. Refuses Qwen4Exp on XPU (`vllm/models/qwen4_exp/__init__.py:30-31`). Default model runner on XPU is V2 |
| `vllm/vllm-openai-xpu:nightly-7f1a5398e9610d96c473931a26c0e12bbe0d0423` | 0.30.1rc1.dev48+g7f1a5398e | checked 2026-09-24. torch 2.14.0+xpu, vllm-xpu-kernels 0.1.15.4, oneCCL 2022.1.2, oneAPI runtime 2026.1.1. Same Qwen4Exp refusal at `__init__.py:30-31` |
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
| Qwen3.8-Flash-Next (Qwen4Exp) | blocked on XPU | upstream raises `NotImplementedError` in 0.30.0, nightly and main; community ports and llm-scaler #660 reviewed in the [deep dive](../../models/qwen3.8-flash-next/research/2026-09-25-vllm-xpu-deep-dive.md) |
| XPU graphs (`VLLM_XPU_ENABLE_XPU_GRAPH=1`) | works | doubles decode throughput versus eager |
| Native MTP speculative decoding | works | depth 4 is the single-user sweet spot on Qwen3.8-27B |
| DFlash / DFlash2 drafters | broken on released versions | non-causal mask dropped by the XPU attention wrapper; RoPE layout fix only on main |
| Prefix caching with MTP on hybrid models | silent corruption risk | three open upstream issues: vllm#53919, #48375, #53505. Prefix caching is **on by default** in 0.28.0, also for hybrid models: with no prefix-caching flag, our Qwen3.8-27B server logged `enable_prefix_caching=True` and `Mamba cache mode is set to 'align' for Qwen3_5ForConditionalGeneration by default when prefix caching is enabled`. Pass `--no-enable-prefix-caching` to turn it off ([finding](../../findings/prefix-cache-mtp-corruption-hybrid.md)) |
| Sleep mode with graphs | needs patch | graphs must be released before sleep |

## Patch catalog

Patches currently applied in production and their upstream status. The patch files live in each model's `configs/patches/`.

| Patch | Purpose | Upstream status |
|---|---|---|
| `patch_mtp_nightly.py` | BF16 MTP draft head with a quantized base | vllm#47828 open |
| `patch_mtp_boundary.py` | off-by-one in speculative verification at context boundaries | not upstream |
| `patch_gdn_mixed_split_v5.py` | Gated DeltaNet mixed prefill/decode split on Battlemage | not upstream |
| `patch_worker_affinity.py` | per-rank `ZE_AFFINITY_MASK` at worker spawn | not upstream |
| `patch_mtp_ptr_wrap.py` | Mamba state pointer overflow | fixed in 0.29.0 (`f936a267`, #48109) |
| `patch_sleep_graphs_v5.py` | release and recapture graphs around sleep mode | not upstream |
| `patch_mem_report2.py` | memory diagnostics RPC | local tooling only |
