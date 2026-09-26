# Inference engines

XPU-specific notes per engine: which images and versions work, which features are supported on Arc, required patches, and known gaps. Model-specific configs stay in `models/<model>/configs/`.

| Engine | Folder | Used for | Status on Arc Pro B70 |
|---|---|---|---|
| vLLM (XPU backend) | [vllm-xpu/](vllm-xpu/) | production serving, tensor parallel, MTP (multi-token prediction) | primary for [Qwen3.8-27B](../models/qwen3.8-27b/). Refuses Qwen3.8-Flash-Next on XPU (re-checked 2026-09-26) |
| Intel llm-scaler (vLLM fork) | [llm-scaler/](llm-scaler/) | Intel-patched vLLM builds | reference for patches. No Qwen3.8-Flash-Next image (2026-09-26) |
| llama.cpp (SYCL backend) | [llama-cpp-sycl/](llama-cpp-sycl/) | GGUF models; Qwen3.8-Flash-Next on four cards with layer split and MTP | tested; primary for Qwen3.8-Flash-Next |
| OpenVINO / OpenVINO Model Server | [openvino/](openvino/) | Intel's own inference stack | not yet tested. No Qwen3.8-Flash-Next support (2026-09-26) |
| SGLang (XPU backend) | none yet | not used here | not tested here. Functional Qwen3.8-Flash-Next support on XPU merged upstream 2026-09-24 ([#37213](https://github.com/sgl-project/sglang/pull/37213)), tested by Intel on Arc Pro B60 only, in no release as of 2026-09-26 |

For each engine, record:

- Image or build, with tag and digest.
- Supported quantization formats and the XPU kernel each one lands on.
- Speculative decoding support (MTP, EAGLE, DFlash, n-gram).
- Graph capture and compile modes that work.
- Multi-GPU support and collective backend.
- Patches we apply, and the upstream PR or issue each one tracks.

## Routes for Qwen3.8-Flash-Next on Arc (2026-09-26)

Only llama.cpp runs Qwen3.8-Flash-Next on Arc with the model's sparse attention, MTP and prefix reuse together; that is why it serves this model here. Every vLLM route is a fork or a private build, Intel's own work is not runnable, and SGLang runs it without a prefix cache. Details and every source are in the [community survey](../models/qwen3.8-flash-next/research/2026-09-26-flash-next-on-vllm-intel-community.md). Third-party numbers are UNVERIFIED here.

| Route | State on 2026-09-26 | Attention / MTP / prefix cache | Reported speed | Source |
|---|---|---|---|---|
| vLLM upstream (v0.30.0, nightly `ddd6fbca`, `main` `379e9a1ea8`) | refuses Qwen4Exp on XPU (`NotImplementedError` at `__init__.py:30-31`); XPU PRs #57535 and #55068 are drafts | none | none | [gate on main](https://github.com/vllm-project/vllm/blob/main/vllm/models/qwen4_exp/__init__.py), [vllm-xpu/](vllm-xpu/README.md) |
| vLLM fork: devan-carlin's dense port, packaged by Ryan Purdy's kit | public fork `a69fba21` on a 2026-08-07 base | dense, not the trained QSA (Qwen sparse attention); MTP off; prefix caching on by default, hit correctness untested | 52.5-54.2 tok/s single stream, 634 tok/s at 16 streams (4x B70) | [kit](https://github.com/imryanpurdy/Qwen3.8-Flash-Next-4x-Intel-B70s), [Hugging Face discussion #3](https://huggingface.co/Intel/Qwen3.8-Flash-Next-W4A16-AutoRound/discussions/3) |
| vLLM fork: steveseguin lab (upstream `amd/` model with the `nvidia/` host PLE (per-layer embedding) layer) | public patches; campaign closed 2026-09-13 | sparse QSA; MTP1; prefix caching off | 46.854 tok/s at a 4,352-token capacity; 44.05 tok/s at 32K input (4x B70, FP8) | [results README](https://github.com/steveseguin/b70-optimization-lab/blob/main/results/qwen38-flash-next-fp8-b70/README.md) |
| Intel llm-scaler | no Flash-Next image; newest release `vllm-0.26.0-b2` (2026-09-09); PR #660 is kernels only, the vLLM integration is unpublished | ESIMD (explicit SIMD) sparse QSA; MTP K4; pinned-host PLE (kernels only) | commit messages: 47.48-51.76 tok/s at TP4 (tensor parallel over 4 cards) INT4, 63.81 with MTP K4; TPOT (time per output token) 25.94 and 26.55 ms at batch 1 on the one commit naming Intel B70 | [#660](https://github.com/intel/llm-scaler/pull/660), [llm-scaler/](llm-scaler/README.md) |
| SGLang XPU | merged 2026-09-24 (`5c44214d6a`); in no release (`lmsysorg/sglang:v0.5.20-xpu` was built 2026-09-18) | attention path not recorded; MTP/NEXTN included; `--disable-radix-cache` mandatory, so no prefix cache; PLE on the GPU | none: correctness only, Arc Pro B60 (one card and TP=4) | [#37213](https://github.com/sgl-project/sglang/pull/37213) |
| llama.cpp SYCL or Vulkan | `qwen4exp` upstream since [#27742](https://github.com/ggml-org/llama.cpp/pull/27742); SYCL sparse FA (flash attention) merged in [#28796](https://github.com/ggml-org/llama.cpp/pull/28796) on 2026-09-25; MTP out of tree in [#28243](https://github.com/ggml-org/llama.cpp/pull/28243) | sparse QSA on SYCL (`GGML_SYCL_SPARSE_FA=1`); MTP from #28243; prefix reuse in `llama-server`. Vulkan not tested here | ours at 135K: decode medians 43.7-48.7 tok/s across the 2026-09-26 trees; on tree `a890bf8b0` follow-up turns 0.76 s and a cold read of 283.66 s | [llama-cpp-sycl/](llama-cpp-sycl/README.md), [cold read by build](../models/qwen3.8-flash-next/benchmarks/2026-09-26-cold-read-135k-by-build.md) |
| IPEX-LLM | archived (last push 2026-01-28) | none | no Flash-Next support | [intel/ipex-llm](https://github.com/intel/ipex-llm) |
| OpenVINO | no Flash-Next support: Intel's Qwen3.8 listing on Hugging Face has only a Qwen3.8-27B OpenVINO model (created 2026-09-24) | none | none | [Hugging Face listing](https://huggingface.co/api/models?author=Intel&search=Qwen3.8), [openvino/](openvino/README.md) |
| KTransformers | no Flash-Next support; failed to reach generation on one B70 (kotoba-lang, 2026-08-28/29) | none | none | [kotoba-lang/inference](https://github.com/kotoba-lang/inference/commits/main) |

Private vLLM builds (First-Ad-117, r1nzl3r99, TSUMUGI-XE), the vLLM GGUF plugin and two small SYCL engines are listed in the community survey. Ollama, MLC-LLM and ExLlama were not checked for this model on Arc.

**The bar for our workload.** An engine replaces llama.cpp for this repository's workload only if it clears every point below, and as of 2026-09-26 no route does ([community survey](../models/qwen3.8-flash-next/research/2026-09-26-flash-next-on-vllm-intel-community.md)). The workload is one coding-agent session at about 135K tokens that reuses its prefix on every turn. The engine must hit the prefix cache at about 135K and give correct output on the hit, checked with the prefix-cache step of the [correctness check](../benchmarks/README.md#correctness-check). It must answer a follow-up turn in under our about 0.8 s (0.76 s median for about 60 new tokens on tree `a890bf8b0`) and decode at or above our 43.7-48.7 tok/s at 135K ([cold read by build](../models/qwen3.8-flash-next/benchmarks/2026-09-26-cold-read-135k-by-build.md)). It must hold 262,144 tokens of context ([launch script](../models/qwen3.8-flash-next/configs/llama-server-262k-mtp.sh)). It must keep the model's behaviour: the trained sparse QSA, not dense attention, which changes behaviour above about 2K tokens ([deep dive](../models/qwen3.8-flash-next/research/2026-09-25-vllm-xpu-deep-dive.md)). Cold-read speed and concurrency do not count toward the bar: the dense fork reads a cold 135K prompt about 5x faster (estimate) and serves 16 streams, but neither helps one session that reuses its prefix.
