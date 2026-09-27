# Experiments log

One file per change attempt, named `YYYY-MM-DD-<slug>.md`. Start from [`templates/experiment.md`](../../../templates/experiment.md).

## History before this repo

Reconstructed from the [tuning blueprint](../research/tuning-blueprint.md). No per-experiment files exist for these.

| Stage | Change | Single-stream | c16 aggregate | Status |
|---|---|---|---|---|
| Baseline | single GPU, FP8, eager | 25 to 45 tok/s | about 180 tok/s | superseded |
| TP2 | two cards, no graphs, oneCCL defaults | 28 to 35 tok/s | about 193 tok/s | superseded |
| Graphs | per-worker affinity patch plus XPU graphs, P2P off | 55 to 75 tok/s | about 356 tok/s | kept |
| Production | P2P on plus MTP2 | 78 to 84 tok/s | 405 to 442 tok/s | kept |

## Log

| Date | Experiment | Change | Result | Status |
|---|---|---|---|---|
| 2026-09-23 | [TP4 and two TP2 lanes on the quad-B70 host](2026-09-23-tp4-and-dual-tp2-quad-b70.md) | `--tensor-parallel-size 4`, all four cards visible, `--enable-sleep-mode` | 129.1 tok/s single-stream vs 86.0 at TP2 (synthetic bench); earlier attempts without the sleep-mode allocator exhausted host RAM | inconclusive |
| | power cap 230 W per card | `xpu-smi config --powerlimit 230` | | planned |
| | MTP depth 4 | `num_speculative_tokens: 4` | | planned |
| | oneCCL twoshots | `CCL_SYCL_ALLREDUCE_LL=twoshots`, `CCL_SEND/RECV=direct` | | planned |
| | whole-graph compile, block size 64 | `splitting_ops: []`, `FULL_DECODE_ONLY`, capture sizes to 320 | | planned |
| | lab image A/B | [`configs/lab-mtp4-int4-draft-head.sh`](../configs/lab-mtp4-int4-draft-head.sh) | | planned |
| | INT4 draft LM head | `VLLM_XPU_DRAFT_LM_HEAD_INT4=1` | | planned |
| | vLLM 0.29.0 | image upgrade | | planned |
| | compute-runtime 26.31 | container `.deb` swap | | planned |
