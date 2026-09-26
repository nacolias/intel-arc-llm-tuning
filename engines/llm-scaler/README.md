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

## Notes
