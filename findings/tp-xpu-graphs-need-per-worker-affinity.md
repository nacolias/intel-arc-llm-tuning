# Tensor-parallel XPU graphs need per-worker Level Zero affinity

| | |
|---|---|
| **Date** | 2026-09 |
| **Confidence** | verified-here |
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
