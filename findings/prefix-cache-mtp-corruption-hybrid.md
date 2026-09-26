# Prefix caching plus MTP can silently corrupt hybrid models

| | |
|---|---|
| **Date** | 2026-09 |
| **Confidence** | upstream-documented |
| **Applies to** | vLLM with `--enable-prefix-caching` and MTP speculative decoding on hybrid (linear attention plus attention) models such as Qwen3.8 |
| **Area** | engine |

## Finding

Three open upstream vLLM issues describe silent output corruption when prefix caching and MTP are combined on hybrid models: vllm#53919, vllm#48375 and vllm#53505 (all open as of 2026-09-05). The production Qwen3.8-27B config runs exactly this combination.

## Impact

No speed effect. The risk is wrong output that looks plausible.

## What to do

- Run the prefix-cache part of the correctness check in [`benchmarks/README.md`](../benchmarks/README.md).
- Read Intel llm-scaler's prefix-cache plus MTP fix (see [engines/llm-scaler](../engines/llm-scaler/)) before writing a local patch.
- If the check fails, disable prefix caching until a fix is applied.

## Still open

Not yet reproduced on our hardware. Track the three issues on each vLLM upgrade.

## Note, 2026-09-25: prefix caching is on by default in vLLM v0.28.0

Omitting `--enable-prefix-caching` does not keep a server out of this bug. vLLM v0.28.0 turns prefix caching on by default, including for hybrid models, where it selects the "align" Mamba cache mode. Our Qwen3.8-27B servers on `vllm/vllm-openai-xpu:v0.28.0` (TP2 and TP4 runs on [quad-b70-5800x-pex88096](../hardware/hosts/quad-b70-5800x-pex88096.md), 2026-09-24) were launched with no prefix-caching flag and logged:

```
enable_prefix_caching=True
Mamba cache mode is set to 'align' for Qwen3_5ForConditionalGeneration by default when prefix caching is enabled
```

So a production config that relies on "we removed the flag" is running prefix caching with MTP. Pass `--no-enable-prefix-caching` explicitly to stay out of the bug until the correctness check passes. The align-mode block-copy path is also the one that vllm-xpu-kernels [#544](https://github.com/vllm-project/vllm-xpu-kernels/pull/544) (Gated DeltaNet conv-state layout in the speculative-decode kernel, merged 2026-08-25) fixes; vllm-xpu-kernels 0.1.13.2 in the v0.28.0 image does not have it, v0.30.0 (0.1.14.1) does ([engines/vllm-xpu](../engines/vllm-xpu/)).
