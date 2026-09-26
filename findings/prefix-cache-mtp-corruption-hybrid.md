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
