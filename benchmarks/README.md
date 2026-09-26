# Benchmark methodology

Shared rules so numbers are comparable across models, configs and months. Per-model results live in `models/<model>/benchmarks/`; this folder holds the method, the workloads and a cross-model summary.

## Metrics

| Metric | Definition |
|---|---|
| Single-stream decode | output tokens per second for one request, excluding time to first token |
| Aggregate throughput | total output tokens per second across all concurrent requests |
| TTFT | time to first token; say whether the prefix cache was cold or warm |
| TPOT | time per output token after the first |
| Acceptance rate | `vllm:spec_decode_num_accepted_tokens_total / vllm:spec_decode_num_draft_tokens_total` from `/metrics` |

## Standard run

1. **Warm up.** Send at least 3 requests and discard them. The first requests after start pay for JIT and graph capture.
2. **Concurrency sweep.** 1, 2, 4, 8, 16, 32. Stop early if the config caps `max-num-seqs` lower.
3. **Context sweep.** Single-stream decode and cold TTFT at 2K, 32K and 128K input tokens, plus the maximum context the config serves.
4. **Workload classes.** Run prose, code and structured-docs prompts separately. Speculative decoding acceptance differs a lot by class: one dual-B70 lab measured 111 tok/s on prose and 183 tok/s on Python with the same server.
5. **Repeat.** At least 3 runs per point; report the median and note the spread.
6. **Record the environment.** Use the table in [`templates/benchmark.md`](../templates/benchmark.md). Include the power cap per card.
7. **Save raw output** to `models/<model>/benchmarks/raw/`.

## Correctness check

Run this whenever a change touches kernels, graphs, collectives, speculative decoding or caching.

- Greedy decode (temperature 0) of a fixed prompt set, compared token by token against a reference run with speculative decoding off.
- Repeat under concurrency (c4 and c16): outputs must match the single-stream outputs.
- With prefix caching on, send the same prompt twice and compare.
- Watch logs for NaN, repetition loops and garbled output.

A speed gain that changes greedy output is recorded as `reverted` unless the difference is explained.

## Things that silently skew results

- Power cap left at the 150 W default on one card.
- Thermal throttling during long runs; log temperatures and clocks.
- Prefix cache hits making TTFT look fast.
- Different sampling settings between baseline and change.
- A stale compile cache from a different image.

## Workloads

Standard prompt sets live in [`workloads/`](workloads/).

## Cross-model summary

Best measured configuration per model and host. Update when a model card's headline numbers change.

| Model | Host | Config | Single-stream | c8 aggregate | c16 aggregate | Date |
|---|---|---|---|---|---|---|
| Qwen3.8-27B GPTQ INT4 | dual-b70-5800x (2x B70) | TP2, MTP2, XPU graphs, FP8 KV | 78 to 84 tok/s | 291.6 tok/s | 405 to 442 tok/s | 2026-09 |
