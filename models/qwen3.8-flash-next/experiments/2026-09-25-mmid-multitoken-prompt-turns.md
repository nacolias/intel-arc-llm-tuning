# Per-token fused MUL_MAT_ID for prompt turns up to 512 tokens

| | |
|---|---|
| **Date** | 2026-09-25 |
| **Model / checkpoint** | huihui-ai abliterated `UD-Q4_K_XL` + unsloth MTP draft shared-Q8_0 + mmproj BF16 |
| **Host** | [quad-b70-5800x-pex88096](../../../hardware/hosts/quad-b70-5800x-pex88096.md) |
| **Status** | reverted (slower) |
| **Baseline** | [benchmarks/2026-09-25-agent-session-135k.md](../benchmarks/2026-09-25-agent-session-135k.md), runs `lc-base` and `lc-lc1` |
| **Result** | [raw/2026-09-25-agent-session-135k.csv](../benchmarks/raw/2026-09-25-agent-session-135k.csv), run `lc-mmid512` |
| **Related** | [fused MUL_MAT_ID](2026-09-24-fused-mul-mat-id-mtp-verify.md), [patch 0004](../configs/patches/0004-sycl-op-profile.diff), [finding: MoE prefill host sync](../../../findings/llama-cpp-sycl-moe-prefill-host-sync.md) |

## Hypothesis

In an agent session at 135K context, each turn appends 60-400 tokens, and those prompt turns take 1.1 to 2.9 s. Batches over 8 tokens take the host-synchronised per-expert `MUL_MAT_ID` loop. Sending batches of up to 512 tokens through the per-token fused path from [patch 0003](../configs/patches/0003-sycl-fused-mul-mat-id-2-8-tokens.diff) removes those host waits, so prompt turns should get faster.

## Change

```diff
+ Environment=GGML_SYCL_MMID_MULTITOKEN_MAX=512
```

The variable comes from [patch 0004](../configs/patches/0004-sycl-op-profile.diff); the default is 8.

## Environment

| Component | Version / value |
|---|---|
| Engine and image (tag and digest) | llama.cpp, no container. Sandbox build: `20260924-7e5cb8f13` plus the profiler and batch-limit code later committed as 0004 |
| Build flags | `GGML_SYCL=ON GGML_SYCL_TARGET=INTEL GGML_SYCL_F16=ON GGML_SYCL_DNN=ON GGML_SYCL_GRAPH=ON GGML_SYCL_SUPPORT_LEVEL_ZERO_API=ON GGML_NATIVE=ON`, Release, icx/icpx from oneAPI 2026.1 |
| compute-runtime / IGC | 26.35.39758.11 / IGC 2.41.9; Level Zero loader 1.32.0 |
| Kernel / `xe` driver / GuC firmware | 7.0.0-34-generic / in-tree `xe` / GuC 70.58.0 |
| oneCCL | not used |
| PyTorch XPU | not used |
| Power cap per card | 230 W |
| Tensor / pipeline parallel | layer split across 4 cards, `-ub 1536 -ts 12,13,13,11 -fit off`, MTP n-max 3 on `SYCL3`, dense attention |
| GPU clock floor | 2800 MHz |

## Procedure

[`tools/lcbench.py`](../../../tools/lcbench.py) with `LC_TOKENS=135000 LC_TURNS=6 LC_PREDICT=256`: one cold read of a fixed 134,862-token C++ context with `cache_prompt`, then 6 turns that each append the previous answer and a 120- or 1,500-character tool-output snippet (about 60 or 400 new tokens) and decode 256 tokens greedily.

## Results

| Metric | Baseline (`lc-base`, `lc-lc1`) | `GGML_SYCL_MMID_MULTITOKEN_MAX=512` | Delta |
|---|---|---|---|
| Prompt time, ~60-token turns | 1.11-1.35 s | 1.08-1.31 s | none |
| Prompt time, ~400-token turns | 2.73-2.94 s | 3.40-3.64 s | +16% to +33% slower |
| Decode median, turns 1-6 | 23.3 / 22.3 tok/s | 21.1 tok/s | within noise (acceptance 59.6% against 63-67%) |
| Cold read of 134,862 tokens | 448.8 / 405.9 s | 472.2 s | |
| Aggregate at c8 / c16 | not applicable (one slot) | | |

- ~60-token turns are not bound by the MoE slow path. For comparison, a 19-token prompt at short context takes about 0.3 s.
- A later op profile of ~60-token prompt-only turns on build `20260925-60a598ed8` put `MUL_MAT_ID` down, gate and up at 16.9%, 14.8% and 14.4% of op time (46% together), with the per-expert path in use ([raw/2026-09-25-op-profile-135k.csv](../benchmarks/raw/2026-09-25-op-profile-135k.csv)). So the MoE work matters. The likely reason the per-token path loses is that it reads every routed expert once per token, which costs more than the host syncs it removes. This was not profiled.

## Correctness

Not assessed: the change was reverted on speed alone.

## Decision

Reverted. The default limit stays at 8, which covers MTP verify batches (4 tokens).

## Follow-ups

- [ ] A grouped multi-token MoE kernel: one launch per projection that reads each expert once for all tokens routed to it, with expert ids kept on the device.
- [ ] Find where the rest of a ~60-token turn's 1.1 s goes. Dense attention on the 12 QSA layers is about 4 ms per call at 60 tokens, above the sparse path's 32-row limit.
