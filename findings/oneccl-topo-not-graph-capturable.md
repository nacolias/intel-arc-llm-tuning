# oneCCL topo algorithms cannot be captured in SYCL graphs

| | |
|---|---|
| **Date** | 2026-09 |
| **Confidence** | verified-here |
| **Applies to** | oneCCL 2022.0.0 as shipped in PyTorch 2.13 XPU wheels |
| **Area** | oneCCL |

## Finding

oneCCL's `topo` scheduler uses worker threads to run transfers, and those cannot be recorded inside a SYCL graph. The SYCL kernel collectives can. Forcing every message size onto the SYCL "simple" kernel path keeps all collectives capturable.

## Evidence

Graph record errors with the topo path; clean capture with the settings below. See the Qwen3.8-27B [tuning blueprint](../models/qwen3.8-27b/research/tuning-blueprint.md), section 5.

## What to do

```bash
CCL_ENABLE_SYCL_KERNELS=1
CCL_ENABLE_TOPO_ALGO=1
CCL_WORKER_COUNT=1
CCL_SYCL_ALLREDUCE_SIMPLE_THRESHOLD=4294967296
CCL_SYCL_REDUCE_SCATTER_SIMPLE_THRESHOLD=4294967296
CCL_SYCL_ALLGATHERV_SIMPLE_THRESHOLD=4294967296
CCL_SYCL_ALLTOALL_TMP_BUF=1
```

The 4 GiB threshold keeps every message size up to 4 GB on the capturable path.

## Still open

- The low-latency protocol choice (`CCL_SYCL_ALLREDUCE_LL=ring` versus `twoshots`) has not been measured on two ranks here.
- A community lab found the pinned `libccl` failed a graph-replay exactness test at the default LL threshold. Run [`tools/xccl_allreduce_census.py`](../tools/xccl_allreduce_census.py) after any oneCCL change.
