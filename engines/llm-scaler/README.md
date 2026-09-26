# Intel llm-scaler

Intel's vLLM distribution for Arc and Arc Pro, with its own patches and a platform installer.

| Item | Value |
|---|---|
| Latest image seen | `intel/llm-scaler-vllm:0.26.0-b2` (2026-09), based on vLLM 0.26 |
| Platform bundle | Intel RDC "multi-arc-bmg-offline-installer" (26.18.8.2): kernel, GuC firmware, compute-runtime, oneCCL, `xpu-smi`, `ze_peer` |

## Why it matters

- It is the only place Intel ships a prefix-cache plus MTP fix today. Read its patch set before writing our own fix.
- It supports `method: qwen3_5_mtp`, and DFlash v1 for Qwen3.6-27B and 35B, but only in eager mode with `VLLM_USE_V2_MODEL_RUNNER=1`.
- It is older than our vLLM base, so it is a patch reference rather than a faster image for single-stream decode.
- When people say "Intel's custom driver" for B60 or B70, they usually mean the platform bundle.

## Qwen3.8-Flash-Next: PR #660 (status 2026-09-25)

Intel's only Flash-Next work is [PR #660](https://github.com/intel/llm-scaler/pull/660), "Enabe qwen3.8 flash next". It is open, and nothing in it is runnable yet.

| Item | Value |
|---|---|
| State | open, not draft, no review comments; base branch `upgrade/v0.26.0` |
| Last commit | `7d73f30617`, 2026-09-16 (67 commits); the PR was touched 2026-09-23 with no visible code change |
| What it adds | kernels only, all under `vllm/custom-esimd-kernels-vllm`: ESIMD (explicit SIMD) kernels for QSA (Qwen sparse attention) with exact selection, indexer norm + RoPE and FP16 packed row stores; a pinned-host USM (unified shared memory) lookup for the PLE (per-layer embedding) n-gram table, 2 banks for TP4; ESIMD hyper-connection kernels; INT4 MoE (mixture of experts) for 512 experts / top-10 and an FP8 MoE; a Gated DeltaNet speculative-decode kernel with packed MTP (multi-token prediction) state |
| What is missing | the vLLM model integration. The branch's `vllm_for_multi_arc.patch` has no Qwen4Exp code, and no release or Docker Hub image mentions Flash-Next. The newest release seen was `vllm-0.26.0-b2` (2026-09-09) |
| Reported speed | commit messages only, on an internal container (UNVERIFIED): TP4 INT4 eager 47.48 to 51.76 tok/s at 1K in / 512 out; MTP K4 57.74 to 63.81 tok/s; TP4 weights 17.38 GiB per card; a 254,986-token request served on TP8 FP8 |

Why it matters: it is the only route that keeps the trained QSA sparse attention and MTP on XPU without porting the upstream `amd/` Triton path by hand. See the [vLLM deep dive](../../models/qwen3.8-flash-next/research/2026-09-25-vllm-xpu-deep-dive.md). Re-check when a Flash-Next image or the integration patch is published.

## Notes
