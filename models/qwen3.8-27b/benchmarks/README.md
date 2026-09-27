# Benchmarks

One file per measured run, named `YYYY-MM-DD-<slug>.md`. Start from [`templates/benchmark.md`](../../../templates/benchmark.md) and follow the shared [methodology](../../../benchmarks/README.md). Raw output goes in [`raw/`](raw/).

| Date | Benchmark | Config | Single-stream | Aggregate (c16) | Notes |
|---|---|---|---|---|---|
| 2026-09 | production baseline (from the blueprint) | [production-mtp2-compose.yml](../configs/production-mtp2-compose.yml) | 78 to 84 tok/s | 405 to 442 tok/s | measured with `bench_vllm.py`, 200 output tokens; re-run with the standard methodology before the first experiment |
| 2026-09-23 | [TP4 vs TP2 vs two TP2 lanes, quad-B70 host](2026-09-23-tp4-quad-b70.md) | production flags at TP4, MTP3, sleep-mode allocator (inline in the file) | 129.1 tok/s (TP4); 86.0 (TP2) | 969.0 tok/s (TP4); 756.9 (TP2) | host [quad-b70-5800x-pex88096](../../../hardware/hosts/quad-b70-5800x-pex88096.md), DavidAU merge GPTQ INT4; synthetic single-prompt bench, 512 tokens, one pass, PCIe uplink at Gen1; not comparable with the row above |
