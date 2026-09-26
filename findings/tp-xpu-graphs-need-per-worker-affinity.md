# Tensor-parallel XPU graphs need per-worker Level Zero affinity

| | |
|---|---|
| **Date** | 2026-09 |
| **Confidence** | verified-here (2026-09; see the 2026-09-23 note below) |
| **Applies to** | vLLM XPU 0.28.0 with `--tensor-parallel-size` greater than 1 and `VLLM_XPU_ENABLE_XPU_GRAPH=1` |
| **Area** | engine / Level Zero |

## Finding

With tensor parallelism, each spawned vLLM worker initializes Level Zero across every visible device. That races on memory pool tracking and SYCL command queue capture, and graph capture fails or crashes. Setting `ZE_AFFINITY_MASK` to the worker's own rank inside each worker, before Level Zero or PyTorch initializes, fixes it.

## Evidence

After the affinity patch, graphs captured cleanly (piecewise 14 of 14, full 8 of 8). See the Qwen3.8-27B [tuning blueprint](../models/qwen3.8-27b/research/tuning-blueprint.md), section 3.

## Impact

Decode throughput roughly doubled versus eager mode: single-stream went from 28 to 35 tok/s to 55 to 75 tok/s on dual B70.

## What to do

Apply a spawn-time affinity patch (`patch_worker_affinity.py`) and run the container with `ipc: host` and `SYS_PTRACE`. The setting must happen in the worker, not in the parent environment.

## Still open

Not upstream as of vLLM 0.29.0. Re-check on each vLLM upgrade.

## Correction, 2026-09-23: the affinity patch does not engage in vLLM 0.28.0 as launched

- In vLLM v0.28.0, per-worker device isolation on XPU applies only when `--device-ids` is passed. Without it, `_resolve_device_ids()` returns None and every worker keeps every visible device ([engine/utils.py, v0.28.0](https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/v1/engine/utils.py)). Our launch does not pass it, so `patch_worker_affinity.py` has no effect.
- On [quad-b70-5800x-pex88096](../hardware/hosts/quad-b70-5800x-pex88096.md), the `xe` fdinfo of every TP=4 worker showed DRM clients on all four cards. XPU graphs still captured, and the server decoded 129.1 tok/s single-stream ([benchmark](../models/qwen3.8-27b/benchmarks/2026-09-23-tp4-quad-b70.md)).
- So graph capture with tensor parallelism on 0.28.0 did not depend on this patch on that host. What fixed the earlier capture failures is not established. Treat "What to do" above as unconfirmed until it is retested with and without the patch.
