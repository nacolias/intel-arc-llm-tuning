# Tools

Probes and scripts that work across models. Model-specific launch scripts live in `models/<model>/configs/`.

| Tool | What it measures |
|---|---|
| [`xccl_allreduce_census.py`](xccl_allreduce_census.py) | two-rank XCCL allreduce exactness (correctly rounded fp16 sum, row-count invariance, repeat equality) and latency at 2, 32, 64 and 900 rows of `[rows, 5120]` fp16 |
| [`host_submission_latency_probe.py`](host_submission_latency_probe.py) | per-launch host cost on one card (async launch and launch plus sync, in microseconds); tells you the host latency class |

Run the census inside the engine container with both cards visible, once with the default protocol and once with `CCL_SYCL_ALLREDUCE_LL=twoshots`:

```bash
python -m torch.distributed.run --standalone --nproc_per_node=2 xccl_allreduce_census.py /tmp/ar.json
```

Run it again after any oneCCL, compute-runtime or graph-mode change.

Both probes are copied verbatim from `steveseguin/b70-optimization-lab`, which is public domain (Unlicense).
