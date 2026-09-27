# llama.cpp SYCL graphs never engage with more than one GPU or with MoE

| | |
|---|---|
| **Date** | 2026-09-24 |
| **Confidence** | upstream-documented (source) and verified-here (measured no effect) |
| **Applies to** | llama.cpp SYCL backend built with `GGML_SYCL_GRAPH=ON`: PR #28243 at `6fcaa16` and master at `171e8846b` (checked 2026-09-25); any multi-GPU split, and any model with MUL_MAT_ID (MoE) or CONCAT nodes |
| **Area** | engine |

## Finding

`GGML_SYCL_ENABLE_GRAPH=1` does nothing on a multi-GPU llama.cpp setup. The SYCL backend's `check_graph_compatibility` returns false as soon as more than one SYCL device exists, before it looks at the graph. It also returns false for any MUL_MAT_ID (mixture-of-experts matmul) or CONCAT node, because those ops block on the host. On four B70s serving Qwen3.8-Flash-Next, decode measured 36.47 tok/s with the variable set, the same as without it.

## Evidence

`ggml/src/ggml-sycl/ggml-sycl.cpp:6099-6121` at `6fcaa16` (line 6172 onward on master `171e8846b`); excerpt from llama.cpp, MIT license:

```cpp
static bool check_graph_compatibility(ggml_cgraph * cgraph) {
    if (ggml_sycl_info().device_count > 1) {
        // A sycl_ex::command_graph object can only be created for a single device
        GGML_LOG_INFO("%s: disabling SYCL graphs due to multiple devices\n", __func__);
        return false;
    }
    ...
            case GGML_OP_CONCAT:
                [[fallthrough]];
            case GGML_OP_MUL_MAT_ID:
                // ggml_sycl_mul_mat_id() does a blocking host wait ...
                return false;
```

With graphs enabled, the server log prints `disabling SYCL graphs due to multiple devices` when a graph is computed.

Measured on [quad-b70-5800x-pex88096](../hardware/hosts/quad-b70-5800x-pex88096.md), Qwen3.8-Flash-Next, `-sm layer` across 4 cards, MTP on, 2026-09-24: 36.47 tok/s with `GGML_SYCL_ENABLE_GRAPH=1`, the same as without ([experiment](../models/qwen3.8-flash-next/experiments/2026-09-24-sycl-graphs-multi-gpu.md)).

## Impact

No speed effect either way. The cost is misplaced effort: graph capture is not a lever for multi-GPU or MoE llama.cpp on SYCL today, so the host launch overhead (about 4,000 kernel launches per token on Flash-Next) stays ([decode analysis](../models/qwen3.8-flash-next/research/2026-09-24-decode-bottleneck-and-speed-plan.md)).

## What to do

- Do not A/B `GGML_SYCL_ENABLE_GRAPH` on multi-GPU setups; check for the log line above instead.
- Upstream [#28725](https://github.com/ggml-org/llama.cpp/pull/28725) ("Add SYCL graph record and replay", open) drops the CONCAT exclusion but keeps the multi-device and MUL_MAT_ID ones, so it does not change this.
- Enabling graphs here would need a per-split device check instead of the global device count, a MUL_MAT_ID path without host waits (see [the MoE host-sync finding](llama-cpp-sycl-moe-prefill-host-sync.md)), and no oneMKL calls inside recorded graphs. The expected gain is unmeasured.

## Still open

Re-check when #28725 merges or when the device-count check becomes per split.
