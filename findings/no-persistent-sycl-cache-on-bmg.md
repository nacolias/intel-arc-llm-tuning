# Do not use a persistent SYCL kernel cache on Battlemage

| | |
|---|---|
| **Date** | 2026-09 |
| **Confidence** | reproduced-community |
| **Applies to** | Battlemage with `SYCL_CACHE_PERSISTENT=1` |
| **Area** | runtime |

## Finding

A four-B70 field report (chriswagner-ai) recorded kernel cache poisoning across restarts on Battlemage, which produced corrupt output after a restart.

## Impact

Leaving the cache off costs about 30 seconds of JIT warm-up per start.

## What to do

Do not set `SYCL_CACHE_PERSISTENT=1`. The vLLM compile cache (`/root/.cache/vllm`) is a different cache and is fine to persist.

## Still open

Not reproduced on our hardware. Re-test after compute-runtime upgrades, with the greedy correctness check from [`benchmarks/README.md`](../benchmarks/README.md).
