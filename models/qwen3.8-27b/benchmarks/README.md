# Benchmarks

One file per measured run, named `YYYY-MM-DD-<slug>.md`. Start from [`templates/benchmark.md`](../../../templates/benchmark.md) and follow the shared [methodology](../../../benchmarks/README.md). Raw output goes in [`raw/`](raw/).

| Date | Benchmark | Config | Single-stream | Aggregate (c16) | Notes |
|---|---|---|---|---|---|
| 2026-09 | production baseline (from the blueprint) | [production-mtp2-compose.yml](../configs/production-mtp2-compose.yml) | 78 to 84 tok/s | 405 to 442 tok/s | measured with `bench_vllm.py`, 200 output tokens; re-run with the standard methodology before the first experiment |
