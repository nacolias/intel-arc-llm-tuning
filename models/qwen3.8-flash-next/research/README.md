# Research notes

Reading notes, upstream issues, external reports and long-form analysis for Qwen3.8-Flash-Next. Start new notes from [`templates/research-note.md`](../../../templates/research-note.md).

| Date | Note | Source type | Key claim | Status |
|---|---|---|---|---|
| 2026-09-24 | [Decode bottleneck and ranked speed plan](2026-09-24-decode-bottleneck-and-speed-plan.md) | source review and `perf` profiling | layer-split decode is host- and launch-bound (about 4K launches per token); MTP verify took a host-synced MoE slow path; bandwidth cap about 95 tok/s with MTP | levers #1, #3, #4, #5 and the clock floor measured and kept; the rest not tried |
| 2026-09-24 | [Quants and uncensored variants](2026-09-24-quants-and-uncensored-variants.md) | Hugging Face cards and GGUF headers | `-ub`, not the quant, decides whether 262K + MTP fits; huihui-ai abliterated UD-Q4_K_XL is a drop-in swap | fit and refusal smoke test measured here; other files UNVERIFIED |
| 2026-09-25 | [Prefill bottleneck](2026-09-25-prefill-bottleneck.md) | profiling and source review | MUL_MAT_ID above 8 tokens syncs the host 144 times per micro-batch and runs about 73.7K per-expert GEMMs; pipeline parallelism gains 0% | measured here; fixes not built |
| 2026-09-25 | [vLLM on XPU deep dive](2026-09-25-vllm-xpu-deep-dive.md) | upstream source and community labs | upstream vLLM refuses Qwen4Exp on XPU; the dense community fork has no prefix caching, so agent turns would re-read the whole context | not pursued; community numbers UNVERIFIED |
| 2026-09-25 | [Long-context QSA design](2026-09-25-long-context-qsa-design.md) | reference implementations and op profiles | at 135K, multi-token sparse flash attention lifts decode from 22 to 31-35 tok/s, and a pooled indexer-key cache cuts about 21 ms per MTP step | two of four designs built and kept |
